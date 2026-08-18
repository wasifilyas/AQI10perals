import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone

from training_pipeline import (
    fetch_all_rows, add_cyclical_encoding, add_future_weather_features,
    NUMERIC_FEATURES, FUTURE_WEATHER_FEATURES, CITY_NAME, HORIZONS, supabase,
)


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


def rolling_backtest_clf(pipeline, data, feature_list, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = data[feature_list]
    y = data["target_category"]

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(pipeline)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        fold_metrics.append({"fold": fold, "accuracy": acc, "f1": f1})
        print(f"    fold {fold}: accuracy={acc:.1%} f1={f1:.3f} (test={len(test_idx)})")

    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
    return {"accuracy": avg_acc, "f1": avg_f1}


def train_classifier_for_horizon(df, horizon_hours, horizon_label):
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_future_weather_features(df, horizon_hours)
    rolling_avg_aqi = df["aqi"].rolling(window=24, min_periods=18).mean()
    df["target_aqi"] = rolling_avg_aqi.shift(-horizon_hours)

    data = df.dropna(
        subset=["target_aqi"] + FUTURE_WEATHER_FEATURES + ["future_wind_dir_sin", "future_wind_dir_cos"]
    ).copy()
    data["target_category"] = data["target_aqi"].apply(aqi_to_category)

    split_idx = int(len(data) * 0.85)
    train_df = data.iloc[:split_idx]
    X_train, y_train = train_df[NUMERIC_FEATURES], train_df["target_category"]

    tscv = TimeSeriesSplit(n_splits=5)

    candidates = {
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {"model__n_estimators": [200], "model__max_depth": [8, 14]},
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=42),
            {"model__max_depth": [4, 8], "model__learning_rate": [0.05, 0.1]},
        ),
    }

    best_name, best_arch, best_acc = None, None, -1
    print(f"  Selecting best classifier via grid search...")
    for name, (model, param_grid) in candidates.items():
        pipe = build_pipeline(model)
        search = GridSearchCV(pipe, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"    {name}: best_params={search.best_params_} cv_accuracy={search.best_score_:.1%}")
        if search.best_score_ > best_acc:
            best_name, best_arch, best_acc = name, search.best_estimator_, search.best_score_

    print(f"  Best: {best_name}. Running rolling backtest...")
    metrics = rolling_backtest_clf(best_arch, data, NUMERIC_FEATURES, n_splits=5)
    print(f"  [{horizon_label}] {best_name} BACKTEST AVG: "
          f"accuracy={metrics['accuracy']:.1%} f1={metrics['f1']:.3f}")

    final_pipeline = clone(best_arch)
    final_pipeline.fit(data[NUMERIC_FEATURES], data["target_category"])

    return best_name, final_pipeline, metrics


def save_classifier(pipeline, horizon_label, model_name, metrics):
    version = f"{horizon_label}_clf_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    local_path = f"/tmp/{version}.joblib"
    joblib.dump(pipeline, local_path)

    storage_path = f"{CITY_NAME}/{horizon_label}/{version}.joblib"
    with open(local_path, "rb") as f:
        supabase.storage.from_("models").upload(
            storage_path, f, {"content-type": "application/octet-stream"}
        )

    supabase.table("model_registry").update({"is_active": False}) \
        .eq("city", CITY_NAME).eq("horizon", horizon_label).eq("model_type", "classifier").execute()

    supabase.table("model_registry").insert({
        "version": version,
        "storage_path": storage_path,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1"],
        "is_active": True,
        "city": CITY_NAME,
        "horizon": horizon_label,
        "model_type": "classifier",
    }).execute()
    print(f"Saved classifier: {storage_path}")


def main():
    print("Fetching data from Supabase...")
    df = fetch_all_rows()
    df = add_cyclical_encoding(df)

    print("\n=== ACCURACY SUMMARY (AQI category prediction) ===")
    summary = {}
    for horizon_label, horizon_hours in HORIZONS.items():
        print(f"\n--- Training classifier for horizon: {horizon_label} ---")
        best_name, pipeline, metrics = train_classifier_for_horizon(df, horizon_hours, horizon_label)
        save_classifier(pipeline, horizon_label, best_name, metrics)
        summary[horizon_label] = metrics["accuracy"]

    print("\n=== FINAL ACCURACY BY DAY ===")
    print(f"Day 1 (24h): {summary['24h']:.1%}")
    print(f"Day 2 (48h): {summary['48h']:.1%}")
    print(f"Day 3 (72h): {summary['72h']:.1%}")


if __name__ == "__main__":
    main()