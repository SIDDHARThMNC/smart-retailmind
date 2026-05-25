import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date", "product_id", "product_name", "category",
    "price", "discount", "units_sold", "revenue", "store_id", "region"
]
VALID_CATEGORIES = {"Electronics", "Fashion", "Grocery", "Home", "Beauty"}
VALID_REGIONS = {"North", "South", "East", "West"}


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if len(df) < 10:
        raise ValueError(f"Not enough data in {path}. Upload the full dataset first.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["price", "discount", "units_sold", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["price"] = df["price"].fillna(df["price"].median())
    df["discount"] = df["discount"].fillna(0)
    df["units_sold"] = df["units_sold"].fillna(df["units_sold"].median())

    df = df[(df["price"] > 0) & df["discount"].between(0, 100) & (df["units_sold"] >= 0)]

    calc = df["price"] * df["units_sold"] * (1 - df["discount"] / 100)
    df["revenue"] = df["revenue"].fillna(calc)
    mask = (df["revenue"] - calc).abs() / (calc + 1) > 0.1
    df.loc[mask, "revenue"] = calc[mask]
    df["revenue"] = df["revenue"].round(2)

    df["category"] = df["category"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["store_id"] = df["store_id"].str.strip().str.upper()
    df["product_id"] = df["product_id"].str.strip().str.upper()

    df = df[df["category"].isin(VALID_CATEGORIES) & df["region"].isin(VALID_REGIONS)]
    df = df.dropna(subset=["product_id", "store_id"])

    logger.info(f"Cleaned data: {len(df)} rows")
    return df.reset_index(drop=True)


def create_features(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    dt = df["date"].dt
    df["day"] = dt.day
    df["month"] = dt.month
    df["year"] = dt.year
    df["day_of_week"] = dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = dt.quarter
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 25).astype(int)

    df["effective_price"] = df["price"] * (1 - df["discount"] / 100)
    df["discount_squared"] = df["discount"] ** 2
    df["price_discount_interaction"] = df["price"] * df["discount"]

    med = df["units_sold"].median()
    grp = df.groupby(["product_id", "store_id"])["units_sold"]

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = grp.shift(lag).fillna(med)

    for w in [7, 14, 30]:
        df[f"rolling_mean_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        ).fillna(med)

    df["rolling_std_7"] = grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std()
    ).fillna(0)
    df["rolling_max_7"] = grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    ).fillna(med)
    df["rolling_min_7"] = grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).min()
    ).fillna(med)

    df["trend_7_vs_30"] = df["rolling_mean_7"] / (df["rolling_mean_30"] + 1)

    for col, key in [
        ("product_avg_units", "product_id"),
        ("store_avg_units", "store_id"),
        ("category_avg_units", "category"),
        ("region_avg_units", "region"),
    ]:
        df[col] = df.groupby(key)["units_sold"].transform("mean")

    df["product_std_units"] = df.groupby("product_id")["units_sold"].transform("std").fillna(0)
    df["lag1_vs_product_avg"] = df["lag_1"] / (df["product_avg_units"] + 1)
    df["price_vs_category_avg"] = df["price"] / (
        df.groupby("category")["price"].transform("mean") + 1
    )

    encoders = {}
    for col in ["product_id", "category", "store_id", "region"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "day", "month", "year", "day_of_week", "is_weekend", "quarter", "week_of_year",
        "is_month_start", "is_month_end",
        "price", "discount", "revenue", "effective_price", "discount_squared",
        "price_discount_interaction", "price_vs_category_avg",
        "product_id_enc", "category_enc", "store_id_enc", "region_enc",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_max_7", "rolling_min_7", "trend_7_vs_30",
        "product_avg_units", "product_std_units", "store_avg_units",
        "category_avg_units", "region_avg_units", "lag1_vs_product_avg",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X, y = df[feature_cols], df["units_sold"]
    valid = X.notna().all(axis=1)
    return X[valid], y[valid], feature_cols, encoders, df
