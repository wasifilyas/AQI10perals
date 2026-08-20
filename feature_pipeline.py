import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CITY_NAME = os.getenv("CITY_NAME")
CITY_LAT = float(os.getenv("CITY_LAT"))
CITY_LON = float(os.getenv("CITY_LON"))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,winddirection_10m,surface_pressure,precipitation,boundary_layer_height",
        "timezone": "UTC",
    }
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code in (502, 503, 504) and attempt < 4:
                wait = 5 * (attempt + 1)
                print(f"Server error, retrying in {wait}s... (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise

def fetch_air_quality(start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "UTC",
    }
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code in (502, 503, 504) and attempt < 4:
                wait = 5 * (attempt + 1)
                print(f"Server error, retrying in {wait}s... (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise


def build_dataframe(weather_json, air_json):
    w = weather_json["hourly"]
    a = air_json["hourly"]

    df = pd.DataFrame({
        "ts": w["time"],
        "temperature": w["temperature_2m"],
        "humidity": w["relative_humidity_2m"],
        "wind_speed": w["windspeed_10m"],
        "wind_direction": w["winddirection_10m"],
        "boundary_layer_height": w["boundary_layer_height"],
        "pressure": w["surface_pressure"],
        "precipitation": w["precipitation"],
        "pm2_5": a["pm2_5"],
        "pm10": a["pm10"],
        "co": a["carbon_monoxide"],
        "no2": a["nitrogen_dioxide"],
        "so2": a["sulphur_dioxide"],
        "o3": a["ozone"],
        "aqi": a["us_aqi"],
    })

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["city"] = CITY_NAME
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month

    df = df.sort_values("ts").reset_index(drop=True)
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1h"]

    return df

import math

def clean_record(record):
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def upsert_to_supabase(df):
    df["ts"] = df["ts"].apply(lambda x: x.isoformat())

    records = df.to_dict(orient="records")
    records = [clean_record(r) for r in records]

    batch_size = 200
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("aqi_features").upsert(batch, on_conflict="city,ts").execute()
        print(f"Upserted rows {i} to {i + len(batch)}")


def run_for_date_range(start_date, end_date):
    print(f"Fetching {start_date} to {end_date} for {CITY_NAME}...")
    weather = fetch_weather(start_date, end_date)
    air = fetch_air_quality(start_date, end_date)
    df = build_dataframe(weather, air)
    upsert_to_supabase(df)
    print(f"Done. {len(df)} rows processed.")

def run_backfill(days=365, chunk_days=30):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        run_for_date_range(str(current), str(chunk_end))
        current = chunk_end

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        run_backfill(days=730)
    else:
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        run_for_date_range(str(yesterday), str(today))