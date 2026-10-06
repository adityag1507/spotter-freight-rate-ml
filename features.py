from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DATE = pd.Timestamp("2025-01-01")

def fit_city_coordinates(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    pickup = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].median()
    delivery = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].median()
    return {
        "pickup_lat": pickup["pickup_lat"].to_dict(),
        "pickup_lon": pickup["pickup_lon"].to_dict(),
        "delivery_lat": delivery["delivery_lat"].to_dict(),
        "delivery_lon": delivery["delivery_lon"].to_dict(),
    }

def add_coordinates(frame: pd.DataFrame, city_coordinates: dict[str, dict[str, float]]) -> pd.DataFrame:
    x = frame.copy()
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        if column not in x.columns:
            x[column] = np.nan
    x["pickup_lat"] = x["pickup_lat"].fillna(x["pickup"].map(city_coordinates["pickup_lat"]))
    x["pickup_lon"] = x["pickup_lon"].fillna(x["pickup"].map(city_coordinates["pickup_lon"]))
    x["delivery_lat"] = x["delivery_lat"].fillna(x["delivery"].map(city_coordinates["delivery_lat"]))
    x["delivery_lon"] = x["delivery_lon"].fillna(x["delivery"].map(city_coordinates["delivery_lon"]))
    return x

def add_geo_features(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    lat1 = np.radians(y["pickup_lat"].to_numpy())
    lat2 = np.radians(y["delivery_lat"].to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(y["delivery_lon"].to_numpy() - y["pickup_lon"].to_numpy())
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    haversine_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    y["haversine_km"] = haversine_km
    y["distance_hav_ratio"] = y["distance"] / (haversine_km * 0.621371 + 1.0)
    y["lat_delta"] = y["delivery_lat"] - y["pickup_lat"]
    y["lon_delta"] = y["delivery_lon"] - y["pickup_lon"]
    y["coord_distance"] = np.sqrt(y["lat_delta"] ** 2 + y["lon_delta"] ** 2)
    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    y["bearing_sin"] = np.sin(bearing)
    y["bearing_cos"] = np.cos(bearing)
    return y

def build_features(
    frame: pd.DataFrame,
    train_reference: pd.DataFrame,
    city_coordinates: dict[str, dict[str, float]],
    include_market: bool,
) -> pd.DataFrame:
    x = add_coordinates(frame.copy(), city_coordinates)
    weight_median = train_reference["weight"].abs().median()
    market_median = train_reference["market_index"].median()

    x["weight_missing"] = x["weight"].isna().astype(int)
    x["weight"] = x["weight"].abs().fillna(weight_median)

    date = pd.to_datetime(x["date"])
    x["month"] = date.dt.month
    x["dayofweek"] = date.dt.dayofweek
    x["dayofyear"] = date.dt.dayofyear
    x["weekofyear"] = date.dt.isocalendar().week.astype(int)
    x["dayofmonth"] = date.dt.day
    x["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)
    x["days_since_start"] = (date - BASE_DATE).dt.days
    x["doy_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    x["distance_log"] = np.log1p(x["distance"])
    x["weight_log"] = np.log1p(x["weight"])
    x["weight_per_mile"] = x["weight"] / (x["distance"] + 1.0)
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)
    x["route_eq"] = x["route"] + "__" + x["equipment"].astype(str)

    if include_market:
        x["market_missing"] = x["market_index"].isna().astype(int)
        x["market_index"] = x["market_index"].fillna(market_median)
        x["distance_x_market"] = x["distance"] * x["market_index"]
        x["distance_x_quote"] = x["distance"] * x["quote_signal"]
        x["market_x_quote"] = x["market_index"] * x["quote_signal"]
    else:
        x = x.drop(columns=["market_index", "quote_signal"], errors="ignore")

    x = add_geo_features(x)
    return x.drop(columns=["load_id", "date", "posted_rate", "predicted_rate"], errors="ignore")

def prepare_categories(train_features: pd.DataFrame, other_features: list[pd.DataFrame]) -> list[str]:
    categorical = [c for c in train_features.columns if train_features[c].dtype == "object"]
    for column in categorical:
        train_features[column] = train_features[column].astype("category")
        categories = train_features[column].cat.categories
        for frame in other_features:
            frame[column] = pd.Categorical(frame[column], categories=categories)
    return categorical
