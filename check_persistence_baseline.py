import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from training_pipeline import fetch_all_rows, add_cyclical_encoding, HORIZONS

df = fetch_all_rows()
df = add_cyclical_encoding(df)
df = df.sort_values("ts").reset_index(drop=True)

print("=== PERSISTENCE BASELINE CHECK ===")
print("(naive forecast: tomorrow's avg AQI = today's most recent 24h avg AQI)\n")

for horizon_label, horizon_hours in HORIZONS.items():
    rolling_avg = df["aqi"].rolling(window=24, min_periods=18).mean()
    target = rolling_avg.shift(-horizon_hours)

    # persistence prediction = current 24h rolling average, carried forward unchanged
    persistence_pred = df["aqi_roll24"] if "aqi_roll24" in df.columns else rolling_avg

    valid = target.notna() & persistence_pred.notna()
    y_true = target[valid]
    y_pred = persistence_pred[valid]

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{horizon_label}] Persistence baseline: RMSE={rmse:.2f}  R2={r2:.3f}")