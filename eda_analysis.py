import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CITY_NAME = os.getenv("CITY_NAME", "Karachi")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
os.makedirs("eda_output", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def fetch_all_rows():
    all_rows, start, page_size = [], 0, 1000
    while True:
        resp = (
            supabase.table("aqi_features")
            .select("*")
            .eq("city", CITY_NAME)
            .order("ts")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


print("Fetching data from Supabase...")
df = fetch_all_rows()
print(f"Loaded {len(df)} rows spanning {df['ts'].min()} to {df['ts'].max()}")

# 1. AQI time series over the full period
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["ts"], df["aqi"], linewidth=0.5, color="#1F4E79")
ax.set_title(f"{CITY_NAME} AQI — Full Time Series ({df['ts'].min().date()} to {df['ts'].max().date()})")
ax.set_ylabel("AQI")
plt.tight_layout()
plt.savefig("eda_output/01_aqi_timeseries.png", dpi=150)
plt.close()

# 2. Monthly seasonal pattern (boxplot by month)
df["month_name"] = df["ts"].dt.strftime("%b")
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=df, x="month_name", y="aqi", order=month_order, ax=ax, color="#2E7D6B")
ax.set_title(f"{CITY_NAME} AQI Distribution by Month (Seasonal Pattern)")
plt.tight_layout()
plt.savefig("eda_output/02_monthly_seasonality.png", dpi=150)
plt.close()

# 3. Hourly pattern (average AQI by hour of day)
hourly_avg = df.groupby("hour")["aqi"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color="#B8860B")
ax.set_title(f"{CITY_NAME} Average AQI by Hour of Day")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Average AQI")
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig("eda_output/03_hourly_pattern.png", dpi=150)
plt.close()

# 4. Correlation heatmap
corr_cols = ["aqi", "temperature", "humidity", "wind_speed", "pressure",
             "precipitation", "pm2_5", "pm10", "co", "no2", "so2", "o3"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title(f"{CITY_NAME} Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_output/04_correlation_heatmap.png", dpi=150)
plt.close()

# 5. PM2.5 vs AQI scatter (validate the strongest expected relationship)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["pm2_5"], df["aqi"], alpha=0.15, s=8, color="#A6336B")
ax.set_xlabel("PM2.5")
ax.set_ylabel("AQI")
ax.set_title("PM2.5 vs AQI Relationship")
plt.tight_layout()
plt.savefig("eda_output/05_pm25_vs_aqi.png", dpi=150)
plt.close()

# 6. Wind speed vs AQI (test the dispersion hypothesis)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["wind_speed"], df["aqi"], alpha=0.15, s=8, color="#2E7D6B")
ax.set_xlabel("Wind Speed")
ax.set_ylabel("AQI")
ax.set_title("Wind Speed vs AQI (Dispersion Effect)")
plt.tight_layout()
plt.savefig("eda_output/06_windspeed_vs_aqi.png", dpi=150)
plt.close()

# 7. AQI category distribution
def aqi_to_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy (Sensitive)"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

df["category"] = df["aqi"].apply(aqi_to_category)
cat_order = ["Good","Moderate","Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"]
cat_counts = df["category"].value_counts().reindex(cat_order).fillna(0)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97","#7e0023"]
ax.bar(cat_counts.index, cat_counts.values, color=colors)
ax.set_title(f"{CITY_NAME} — Distribution of Hours by AQI Category")
ax.set_ylabel("Number of hours")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("eda_output/07_category_distribution.png", dpi=150)
plt.close()

# ---- Print summary statistics / trend findings ----
print("\n=== EDA SUMMARY ===")
print(f"Mean AQI: {df['aqi'].mean():.1f} | Median: {df['aqi'].median():.1f} | Std: {df['aqi'].std():.1f}")
print(f"Min AQI: {df['aqi'].min():.1f} | Max AQI: {df['aqi'].max():.1f}")
worst_month = df.groupby("month_name")["aqi"].mean().reindex(month_order).idxmax()
best_month = df.groupby("month_name")["aqi"].mean().reindex(month_order).idxmin()
print(f"Worst month (highest avg AQI): {worst_month}")
print(f"Best month (lowest avg AQI): {best_month}")
worst_hour = hourly_avg.idxmax()
best_hour = hourly_avg.idxmin()
print(f"Worst hour of day: {worst_hour}:00 UTC (avg AQI {hourly_avg[worst_hour]:.1f})")
print(f"Best hour of day: {best_hour}:00 UTC (avg AQI {hourly_avg[best_hour]:.1f})")
print(f"PM2.5-AQI correlation: {corr.loc['pm2_5','aqi']:.3f}")
print(f"Wind speed-AQI correlation: {corr.loc['wind_speed','aqi']:.3f}")
print(f"\nCategory breakdown:\n{cat_counts}")
print("\nAll plots saved to eda_output/")