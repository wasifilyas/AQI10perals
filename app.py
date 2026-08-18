import os
import math
import joblib
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CITY_NAME = os.getenv("CITY_NAME", "Karachi")
CITY_LAT = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON = float(os.getenv("CITY_LON", "67.0011"))

HORIZONS = {"24h": 24, "48h": 48, "72h": 72}

NUMERIC_FEATURES = [
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate",
    "pm2_5_roll6", "pm2_5_roll24", "aqi_roll6", "aqi_roll24",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "day_of_week",
    "wind_dir_sin", "wind_dir_cos", "dispersion_index",
    "future_temperature", "future_humidity", "future_wind_speed",
    "future_pressure", "future_precipitation",
    "future_wind_dir_sin", "future_wind_dir_cos",
]

AQI_CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy (Sensitive)": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def aqi_to_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


@st.cache_data(ttl=1800)
def fetch_recent_features(_supabase, limit=48):
    resp = (
        _supabase.table("aqi_features")
        .select("*")
        .eq("city", CITY_NAME)
        .order("ts", desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(resp.data)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def add_engineered_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    doy = df["ts"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction"] / 360)
    df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction"] / 360)
    df["dispersion_index"] = df["pm2_5"] / (df["wind_speed"] + 1)

    df["pm2_5_roll6"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm2_5_roll24"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["aqi_roll6"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["aqi_roll24"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    return df


@st.cache_data(ttl=1800)
def fetch_forecast_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,winddirection_10m,surface_pressure,precipitation",
        "timezone": "UTC",
        "forecast_days": 4,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(data["time"], utc=True),
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "wind_speed": data["windspeed_10m"],
        "wind_direction": data["winddirection_10m"],
        "pressure": data["surface_pressure"],
        "precipitation": data["precipitation"],
    })
    return df


def get_future_weather_avg(forecast_df, target_time, horizon_hours):
    """Average forecasted weather over the 24h window ending at target_time,
    matching the daily-average target the models were trained on."""
    window_start = target_time - pd.Timedelta(hours=24)
    window_end = target_time
    mask = (forecast_df["ts"] > window_start) & (forecast_df["ts"] <= window_end)
    window = forecast_df[mask]
    if window.empty:
        forecast_df = forecast_df.copy()
        forecast_df["diff"] = (forecast_df["ts"] - target_time).abs()
        return forecast_df.sort_values("diff").iloc[0]
    return window.mean(numeric_only=True)


@st.cache_resource
def load_active_model(_supabase, horizon_label, model_type):
    resp = (
        _supabase.table("model_registry")
        .select("*")
        .eq("city", CITY_NAME)
        .eq("horizon", horizon_label)
        .eq("model_type", model_type)
        .eq("is_active", True)
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None, None

    record = resp.data[0]
    storage_path = record["storage_path"]
    local_path = f"/tmp/{model_type}_{horizon_label}.joblib"

    file_bytes = _supabase.storage.from_("models").download(storage_path)
    with open(local_path, "wb") as f:
        f.write(file_bytes)

    model = joblib.load(local_path)
    return model, record


def build_feature_row(latest_row, future_weather_row):
    row = latest_row.copy()
    row["future_temperature"] = future_weather_row["temperature"]
    row["future_humidity"] = future_weather_row["humidity"]
    row["future_wind_speed"] = future_weather_row["wind_speed"]
    row["future_pressure"] = future_weather_row["pressure"]
    row["future_precipitation"] = future_weather_row["precipitation"]
    row["future_wind_dir_sin"] = np.sin(2 * np.pi * future_weather_row["wind_direction"] / 360)
    row["future_wind_dir_cos"] = np.cos(2 * np.pi * future_weather_row["wind_direction"] / 360)
    return pd.DataFrame([row])[NUMERIC_FEATURES]


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse 800px 400px at 20% -10%, rgba(255,255,255,0.06), transparent),
            radial-gradient(ellipse 600px 300px at 80% 5%, rgba(148,163,184,0.08), transparent),
            radial-gradient(ellipse 900px 500px at 50% 0%, rgba(100,116,139,0.10), transparent),
            radial-gradient(circle at 15% 20%, rgba(226,232,240,0.04) 0%, transparent 8%),
            radial-gradient(circle at 70% 15%, rgba(226,232,240,0.03) 0%, transparent 12%),
            radial-gradient(circle at 40% 35%, rgba(226,232,240,0.025) 0%, transparent 10%),
            linear-gradient(180deg, #0B1120 0%, #10182B 55%, #0B1120 100%);
        background-attachment: fixed;
        color: #E2E8F0;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 24px;
        backdrop-filter: blur(8px);
    }

    .aqi-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1;
        color: #F1F5F9;
    }
    .aqi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 8px;
    }

    .category-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        color: #0B1120;
        margin-top: 10px;
    }

    .day-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .day-label {
        font-family: 'IBM Plex Mono', monospace;
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .day-aqi {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 8px 0;
    }

    .spectrum-wrap {
        margin: 20px 0 8px 0;
    }
    .spectrum-bar {
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #00e400 0%, #ffff00 20%, #ff7e00 40%, #ff0000 60%, #8f3f97 80%, #7e0023 100%);
        position: relative;
    }
    .spectrum-marker {
        position: absolute;
        top: -6px;
        width: 3px;
        height: 26px;
        background: #F1F5F9;
        box-shadow: 0 0 8px rgba(255,255,255,0.8);
    }
    .spectrum-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #64748B;
        margin-top: 6px;
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    .stMetric, div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_spectrum_gauge(aqi_value, max_scale=300):
    position_pct = min(max(aqi_value / max_scale * 100, 0), 100)
    st.markdown(f"""
    <div class="spectrum-wrap">
        <div class="spectrum-bar">
            <div class="spectrum-marker" style="left:{position_pct}%;"></div>
        </div>
        <div class="spectrum-labels">
            <span>0 Good</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300+ Hazardous</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def build_shap_explainer(_model):
    try:
        preprocessor = _model.named_steps["preprocess"]
        estimator = _model.named_steps["model"]
        return shap.TreeExplainer(estimator), preprocessor
    except Exception:
        return None, None


def render_shap_explanation(model, feature_row, horizon_label):
    explainer, preprocessor = build_shap_explainer(model)
    if explainer is None:
        st.info("Feature importance isn't available for this model type.")
        return

    try:
        transformed = preprocessor.transform(feature_row)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance_df = pd.DataFrame({
            "feature": NUMERIC_FEATURES,
            "impact": shap_values[0],
        }).sort_values("impact", key=abs, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0B1120")
        ax.set_facecolor("#0B1120")

        colors = ["#ff7e00" if v > 0 else "#00e400" for v in importance_df["impact"]]
        ax.barh(importance_df["feature"], importance_df["impact"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Impact on predicted AQI", color="#E2E8F0")
        ax.tick_params(colors="#E2E8F0")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_title(f"What's driving the {horizon_label} prediction", color="#F1F5F9", fontsize=11)

        st.pyplot(fig)
        st.caption("🟠 Orange bars push AQI higher · 🟢 Green bars push AQI lower")
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")


def log_predictions(supabase, forecast_results, now):
    records = []
    for r in forecast_results:
        target_ts = now + pd.Timedelta(hours=HORIZONS[r["horizon"]])
        records.append({
            "city": CITY_NAME,
            "predicted_at": now.isoformat(),
            "target_ts": target_ts.isoformat(),
            "horizon": r["horizon"],
            "predicted_aqi": float(r["predicted_aqi"]) if r["predicted_aqi"] else None,
            "predicted_category": r["category"],
        })
    try:
        supabase.table("prediction_log").upsert(
            records, on_conflict="city,target_ts,horizon"
        ).execute()
    except Exception as e:
        st.warning(f"Could not log predictions: {e}")


def update_resolved_predictions(supabase):
    pending = (
        supabase.table("prediction_log")
        .select("*")
        .eq("city", CITY_NAME)
        .is_("actual_aqi", "null")
        .lte("target_ts", datetime.now(timezone.utc).isoformat())
        .execute()
    )
    if not pending.data:
        return

    for pred in pending.data:
        target_ts = pred["target_ts"]
        actual_resp = (
            supabase.table("aqi_features")
            .select("aqi, ts")
            .eq("city", CITY_NAME)
            .gte("ts", target_ts)
            .order("ts")
            .limit(1)
            .execute()
        )
        if not actual_resp.data:
            continue

        actual_aqi = actual_resp.data[0]["aqi"]
        if actual_aqi is None:
            continue

        actual_category = aqi_to_category(actual_aqi)
        was_correct = actual_category == pred["predicted_category"]

        supabase.table("prediction_log").update({
            "actual_aqi": actual_aqi,
            "actual_category": actual_category,
            "was_correct": was_correct,
        }).eq("id", pred["id"]).execute()


def get_accuracy_summary(supabase):
    resp = (
        supabase.table("prediction_log")
        .select("*")
        .eq("city", CITY_NAME)
        .not_.is_("actual_aqi", "null")
        .execute()
    )
    df = pd.DataFrame(resp.data)
    return df


def main():
    st.set_page_config(page_title=f"{CITY_NAME} AQI Forecast", page_icon="🌫️", layout="wide")
    inject_custom_css()
    st.markdown(f'<div class="hero-title">{CITY_NAME} Air Quality Station</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live readings and 3-day forecast (daily average), updated hourly</div>', unsafe_allow_html=True)

    supabase = get_supabase()

    with st.spinner("Loading latest data..."):
        recent_df = fetch_recent_features(supabase)
        recent_df = add_engineered_features(recent_df)
        latest_row = recent_df.iloc[-1]
        forecast_df = fetch_forecast_weather()

    current_aqi = latest_row["aqi"]
    current_category = aqi_to_category(current_aqi)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="aqi-label">Current AQI</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="aqi-number">{current_aqi:.0f}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="category-pill" style="background-color:{AQI_CATEGORY_COLORS[current_category]};">'
            f'{current_category}</div>',
            unsafe_allow_html=True,
        )
        render_spectrum_gauge(current_aqi)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="aqi-label">Station Info</div>', unsafe_allow_html=True)
        st.write(f"**Last updated:** {latest_row['ts'].strftime('%Y-%m-%d %H:%M UTC')}")
        st.write(f"**Location:** {CITY_NAME} ({CITY_LAT}, {CITY_LON})")
        st.write(f"**Source:** Open-Meteo · Supabase · scikit-learn")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("3-Day Forecast (Daily Average)")

    now = datetime.now(timezone.utc)
    cols = st.columns(3)
    forecast_results = []

    for i, (horizon_label, horizon_hours) in enumerate(HORIZONS.items()):
        target_time = now + pd.Timedelta(hours=horizon_hours)
        future_weather = get_future_weather_avg(forecast_df, target_time, horizon_hours)
        feature_row = build_feature_row(latest_row, future_weather)

        reg_model, reg_record = load_active_model(supabase, horizon_label, "regressor")
        clf_model, clf_record = load_active_model(supabase, horizon_label, "classifier")

        pred_aqi = reg_model.predict(feature_row)[0] if reg_model else None
        pred_category = clf_model.predict(feature_row)[0] if clf_model else aqi_to_category(pred_aqi)

        forecast_results.append({
            "horizon": horizon_label,
            "day": f"Day {i+1}",
            "date": target_time.strftime("%b %d"),
            "predicted_aqi": pred_aqi,
            "category": pred_category,
        })

        with cols[i]:
            st.markdown(f"""
            <div class="day-card">
                <div class="day-label">Day {i+1} · {target_time.strftime('%b %d')}</div>
                <div class="day-aqi">{f"{pred_aqi:.0f}" if pred_aqi else "N/A"}</div>
                <div class="category-pill" style="background-color:{AQI_CATEGORY_COLORS.get(pred_category, "#cccccc")};">
                    {pred_category}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Predicted avg AQI")
            if clf_record:
                st.caption(f"Model accuracy: {clf_record['accuracy']:.1%}")

    st.divider()
    st.subheader("🔍 Why these predictions? (Day 1 breakdown)")
    day1_label = list(HORIZONS.keys())[0]
    day1_target = now + pd.Timedelta(hours=HORIZONS[day1_label])
    day1_weather = get_future_weather_avg(forecast_df, day1_target, HORIZONS[day1_label])
    day1_features = build_feature_row(latest_row, day1_weather)
    day1_reg_model, _ = load_active_model(supabase, day1_label, "regressor")
    if day1_reg_model:
        render_shap_explanation(day1_reg_model, day1_features, day1_label)

    update_resolved_predictions(supabase)
    log_predictions(supabase, forecast_results, now)

    ALERT_GUIDANCE = {
        "Unhealthy (Sensitive)": "Sensitive groups (children, elderly, respiratory/heart conditions) should limit prolonged outdoor exertion.",
        "Unhealthy": "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid it entirely.",
        "Very Unhealthy": "Avoid outdoor activity. Keep windows closed. Sensitive groups should remain indoors.",
        "Hazardous": "Stay indoors. Avoid all outdoor exertion. Use an air purifier if available.",
    }

    concerning_days = [r for r in forecast_results if r["category"] not in ("Good", "Moderate")]

    st.divider()
    if concerning_days:
        st.markdown('<div class="glass-card" style="border-color: rgba(255,126,0,0.4);">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Air Quality Alert")
        for r in concerning_days:
            guidance = ALERT_GUIDANCE.get(r["category"], "")
            color = AQI_CATEGORY_COLORS.get(r["category"], "#cccccc")
            st.markdown(f"""
            <div style="border-left: 3px solid {color}; padding-left: 12px; margin: 12px 0;">
                <strong>{r['day']} ({r['date']}) — {r['category']}</strong> (Avg AQI {r['predicted_aqi']:.0f})<br>
                <span style="color:#94A3B8; font-size:0.9rem;">{guidance}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(0,228,0,0.3);">
            ✅ <strong>No air quality alerts</strong> — forecasted average AQI stays in the Good/Moderate range for the next 3 days.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Forecast Trend")
    chart_df = pd.DataFrame(forecast_results)
    st.line_chart(chart_df.set_index("day")["predicted_aqi"])

    st.divider()
    with st.expander("Recent historical readings"):
        st.dataframe(
            recent_df[["ts", "aqi", "pm2_5", "temperature", "wind_speed"]].tail(24).sort_values("ts", ascending=False),
            use_container_width=True,
        )

    st.divider()
    st.subheader("📊 Live Prediction Accuracy Tracker")
    accuracy_df = get_accuracy_summary(supabase)

    if accuracy_df.empty:
        st.info("No resolved predictions yet — check back after your first forecasted day has passed.")
    else:
        overall_accuracy = accuracy_df["was_correct"].mean()
        st.metric("Overall category accuracy (live, so far)", f"{overall_accuracy:.1%}", f"{len(accuracy_df)} predictions resolved")

        by_horizon = accuracy_df.groupby("horizon")["was_correct"].agg(["mean", "count"])
        st.dataframe(
            by_horizon.rename(columns={"mean": "accuracy", "count": "n_predictions"}).style.format({"accuracy": "{:.1%}"}),
            use_container_width=True,
        )

        with st.expander("See individual predictions vs actuals"):
            display_df = accuracy_df[[
                "target_ts", "horizon", "predicted_aqi", "actual_aqi",
                "predicted_category", "actual_category", "was_correct"
            ]].sort_values("target_ts", ascending=False)
            st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()