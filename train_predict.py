from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_features, fit_city_coordinates, prepare_categories

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET_CLIP_LOW = 0.3
TARGET_CLIP_HIGH = 3.1

def build_model(n_estimators: int, num_leaves: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=num_leaves,
        min_child_samples=20,
        colsample_bytree=1.0,
        subsample=0.9,
        reg_lambda=8.0,
        reg_alpha=0.1,
        objective="regression_l1",
        random_state=seed,
        verbosity=-1,
    )

def train() -> None:
    train = pd.read_csv(DATA / "train_test.csv")
    validation = pd.read_csv(DATA / "validation.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv")

    city_coordinates = fit_city_coordinates(train)

    train_market = build_features(train, train, city_coordinates, True)
    validation_market = build_features(validation, train, city_coordinates, True)
    train_nomarket = build_features(train, train, city_coordinates, False)
    validation_nomarket = build_features(validation, train, city_coordinates, False)
    december_nomarket = build_features(december, train, city_coordinates, False)

    cats_market = prepare_categories(train_market, [validation_market])
    cats_nomarket = prepare_categories(train_nomarket, [validation_nomarket, december_nomarket])

    y = train["posted_rate"].to_numpy()
    distance = train["distance"].to_numpy()
    target = np.log(np.clip(y / distance, TARGET_CLIP_LOW, TARGET_CLIP_HIGH))

    market_model = build_model(700, 7, 3)
    no_market_model = build_model(450, 7, 7)
    chart_model = build_model(600, 7, 9)

    market_model.fit(train_market, target, categorical_feature=cats_market)
    no_market_model.fit(train_nomarket, target, categorical_feature=cats_nomarket)

    validation_prediction = (
        0.70 * np.exp(market_model.predict(validation_market)) * validation["distance"].to_numpy()
        + 0.30 * np.exp(no_market_model.predict(validation_nomarket)) * validation["distance"].to_numpy()
    )
    validation_prediction = np.maximum(validation_prediction, 1.0)

    output = pd.DataFrame(
        {
            "load_id": validation["load_id"].astype(str),
            "predicted_rate": np.round(validation_prediction, 2),
        }
    )
    output.to_csv(ROOT / "validation_predictions.csv", index=False)

    chart_model.fit(train_nomarket, target, categorical_feature=cats_nomarket)
    december_prediction = np.exp(chart_model.predict(december_nomarket)) * december["distance"].to_numpy()
    december_prediction = np.maximum(december_prediction, 1.0)

    december_output = december.copy()
    december_output["predicted_rate"] = np.round(december_prediction, 2)
    december_output.to_csv(DATA / "december_chart_inputs.csv", index=False)

    print(f"Wrote {len(output):,} validation predictions.")
    print(f"Wrote {len(december_output)} December predictions.")

if __name__ == "__main__":
    train()
