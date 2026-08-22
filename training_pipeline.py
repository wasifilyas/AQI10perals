import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CITY_NAME = os.getenv("CITY_NAME")
HORIZONS = {"24h": 24, "48h": 48, "72h": 72}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

NUMERIC_FEATURES = [
    "pm2_5_roll24", "doy_cos", "pm2_5_roll6", "future_wind_speed",
    "future_humidity", "doy_sin", "future_temperature", "future_pressure",
    "month_cos", "pm2_5", "pm2_5_log", "future_wind_dir_sin",
    "future_wind_dir_cos", "aqi_lag_24h", "aqi_roll24", "day_of_week",
    "aqi_roll_std_24", "pressure", "aqi_roll6", "humidity",
    "wind_dir_sin", "month_sin", "pm10_log", "dispersion_index",
]

FUTURE_WEATHER_RAW = ["temperature", "humidity", "wind_speed", "pressure", "precipitation","wind_direction","boundary_layer_height"]
FUTURE_WEATHER_FEATURES = [
    f"future_{col}" for col in FUTURE_WEATHER_RAW
]

def add_future_weather_features(df, horizon_hours):
    df = df.copy()
    for col in FUTURE_WEATHER_RAW:
        rolling_col = df[col].rolling(window=24, min_periods=18).mean()
        df[f"future_{col}"] = rolling_col.shift(-horizon_hours)
    df["future_wind_dir_sin"] = np.sin(2 * np.pi * df["future_wind_direction"] / 360)
    df["future_wind_dir_cos"] = np.cos(2 * np.pi * df["future_wind_direction"] / 360)
    return df

def fetch_all_rows():
    all_rows = []
    page_size = 1000
    start = 0
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
    return pd.DataFrame(all_rows)


def add_cyclical_encoding(df):
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    day_of_year = df["ts"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction"] / 360)
    df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction"] / 360)
    df["dispersion_index"] = df["pm2_5"] / (df["wind_speed"] + 1)

    df["pm2_5_roll6"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm2_5_roll24"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["aqi_roll6"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["aqi_roll24"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    df["aqi_roll_std_6"] = df["aqi"].rolling(window=6, min_periods=2).std().fillna(0)
    df["aqi_roll_std_24"] = df["aqi"].rolling(window=24, min_periods=2).std().fillna(0)

    df["pm2_5_log"] = np.log1p(df["pm2_5"])
    df["pm10_log"] = np.log1p(df["pm10"])
    df["co_log"] = np.log1p(df["co"])
    df["no2_log"] = np.log1p(df["no2"])
    df["so2_log"] = np.log1p(df["so2"])
    df["o3_log"] = np.log1p(df["o3"])

    return df


def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

def rolling_backtest(pipeline, data, feature_list, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = data[feature_list]
    y = data["target"]

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(pipeline)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2})

        print(f"    fold {fold}: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.3f} "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_r2 = np.mean([m["r2"] for m in fold_metrics])
    std_rmse = np.std([m["rmse"] for m in fold_metrics])

    return {"rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2, "rmse_std": std_rmse}
def train_and_evaluate(df, horizon_hours, horizon_label):
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_future_weather_features(df, horizon_hours)

    rolling_avg_aqi = df["aqi"].rolling(window=24, min_periods=18).mean()
    df["target"] = rolling_avg_aqi.shift(-horizon_hours)

    data = df.dropna(
        subset=["target"] + FUTURE_WEATHER_FEATURES + ["future_wind_dir_sin", "future_wind_dir_cos"]
    ).copy()

    split_idx = int(len(data) * 0.85)
    train_df = data.iloc[:split_idx]
    X_train, y_train = train_df[NUMERIC_FEATURES], train_df["target"]

    tscv = TimeSeriesSplit(n_splits=5)

    candidates = {
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}),
        "random_forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [6, 12]},
        ),
        "gradient_boosting": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__l2_regularization": [0.0, 0.5, 1.0],
            },
        ),
    }

    tuned_pipelines = {}
    print(f"  Selecting best model type via grid search...")
    for name, (model, param_grid) in candidates.items():
        pipe = build_pipeline(model)
        search = GridSearchCV(
            pipe, param_grid, cv=tscv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_pipelines[name] = search.best_estimator_
        print(f"    {name}: best_params={search.best_params_} cv_rmse={-search.best_score_:.2f}")

    # Build a Ridge + Gradient Boosting ensemble from the already-tuned pipelines
    ensemble = VotingRegressor([
        ("ridge", tuned_pipelines["ridge"]),
        ("gb", tuned_pipelines["gradient_boosting"]),
    ])

    all_final_candidates = {**tuned_pipelines, "ensemble_ridge_gb": ensemble}

    best_name, best_arch, best_rmse, best_metrics = None, None, float("inf"), None
    for name, arch in all_final_candidates.items():
        print(f"  Running rolling backtest for {name}...")
        metrics = rolling_backtest(arch, data, NUMERIC_FEATURES, n_splits=5)
        print(f"    [{horizon_label}] {name} BACKTEST AVG: RMSE={metrics['rmse']:.2f} "
              f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")
        if metrics["rmse"] < best_rmse:
            best_name, best_arch, best_rmse, best_metrics = name, arch, metrics["rmse"], metrics

    print(f"  Best overall: {best_name}. Refitting on full dataset for deployment...")
    final_pipeline = clone(best_arch) if best_name != "ensemble_ridge_gb" else VotingRegressor([
        ("ridge", clone(tuned_pipelines["ridge"])),
        ("gb", clone(tuned_pipelines["gradient_boosting"])),
    ])
    final_pipeline.fit(data[NUMERIC_FEATURES], data["target"])

    return best_name, final_pipeline, best_metrics

def save_model(pipeline, horizon_label, model_name, metrics):
    version = f"{horizon_label}_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    local_path = f"/tmp/{version}.joblib"
    joblib.dump(pipeline, local_path)

    storage_path = f"{CITY_NAME}/{horizon_label}/{version}.joblib"
    with open(local_path, "rb") as f:
        supabase.storage.from_("models").upload(
            storage_path, f, {"content-type": "application/octet-stream"}
        )

    supabase.table("model_registry").update({"is_active": False}) \
        .eq("city", CITY_NAME).eq("horizon", horizon_label).execute()

    supabase.table("model_registry").insert({
        "version": version,
        "storage_path": storage_path,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "is_active": True,
        "city": CITY_NAME,
        "horizon": horizon_label,
    }).execute()

    print(f"Saved and registered: {storage_path}")


def main():
    print("Fetching data from Supabase...")
    df = fetch_all_rows()
    print(f"Loaded {len(df)} rows")

    df = add_cyclical_encoding(df)

    for horizon_label, horizon_hours in HORIZONS.items():
        print(f"\n--- Training for horizon: {horizon_label} ---")
        best_name, best_pipeline, metrics = train_and_evaluate(df, horizon_hours, horizon_label)
        print(f"Best model for {horizon_label}: {best_name}")
        save_model(best_pipeline, horizon_label, best_name, metrics)


if __name__ == "__main__":
    main()