from sklearn.ensemble import RandomForestRegressor
from training_pipeline import (
    fetch_all_rows, add_cyclical_encoding, add_future_weather_features,
    NUMERIC_FEATURES, HORIZONS,
)
import numpy as np
import pandas as pd

df = fetch_all_rows()
df = add_cyclical_encoding(df)

all_features_to_test = NUMERIC_FEATURES + [
    "pm2_5_log", "pm10_log", "co_log", "no2_log", "so2_log", "o3_log"
]

importance_scores = pd.Series(0.0, index=all_features_to_test)

for horizon_label, horizon_hours in HORIZONS.items():
    d = df.sort_values("ts").reset_index(drop=True)
    d = add_future_weather_features(d, horizon_hours)
    rolling_avg = d["aqi"].rolling(window=24, min_periods=18).mean()
    d["target"] = rolling_avg.shift(-horizon_hours)
    d = d.dropna(subset=["target"] + all_features_to_test)

    X, y = d[all_features_to_test], d["target"]
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_scores += pd.Series(rf.feature_importances_, index=all_features_to_test)

importance_scores = importance_scores.sort_values(ascending=False)
print("Average feature importance across all 3 horizons:\n")
print(importance_scores.to_string())

top_features = importance_scores.head(24).index.tolist()
print(f"\nTop 24 features to keep:\n{top_features}")