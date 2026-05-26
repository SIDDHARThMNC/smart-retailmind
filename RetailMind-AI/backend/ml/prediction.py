import os
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(_APP_ROOT, "backend", "saved_models", "demand_model.pkl")
PREPROCESSOR_PATH = os.path.join(_APP_ROOT, "backend", "saved_models", "preprocessor.pkl")


def predict_demand(product_id, date_str, price, discount, store_id, region):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run POST /train first.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Preprocessor not found. Run POST /train first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    encoders = preprocessor["encoders"]
    feature_cols = preprocessor["feature_cols"]

    date = pd.to_datetime(date_str)

    def safe_encode(enc, val):
        return int(enc.transform([val])[0]) if val in enc.classes_ else 0

    est = 20.0
    eff = price * (1 - discount / 100)

    row = {
        "day": date.day,
        "month": date.month,
        "year": date.year,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "quarter": date.quarter,
        "week_of_year": int(date.isocalendar()[1]),
        "is_month_start": int(date.day <= 5),
        "is_month_end": int(date.day >= 25),
        "price": price,
        "discount": discount,
        "revenue": eff * est,
        "effective_price": eff,
        "discount_squared": discount ** 2,
        "price_discount_interaction": price * discount,
        "price_vs_category_avg": 1.0,
        "product_id_enc": safe_encode(encoders["product_id"], product_id.upper()),
        "category_enc": 0,
        "store_id_enc": safe_encode(encoders["store_id"], store_id.upper()),
        "region_enc": safe_encode(encoders["region"], region.title()),
        "lag_1": est, "lag_7": est, "lag_14": est, "lag_30": est,
        "rolling_mean_7": est, "rolling_mean_14": est, "rolling_mean_30": est,
        "rolling_std_7": 0, "rolling_max_7": est, "rolling_min_7": est,
        "trend_7_vs_30": 1.0,
        "product_avg_units": est, "product_std_units": 0,
        "store_avg_units": est, "category_avg_units": est, "region_avg_units": est,
        "lag1_vs_product_avg": 1.0,
    }

    input_df = pd.DataFrame([row])[[c for c in feature_cols if c in row]]
    missing_features = [c for c in feature_cols if c not in row]
    if missing_features:
        logger.warning(f"Missing features (using defaults): {missing_features}")
    predicted = max(0, int(round(model.predict(input_df)[0])))

    logger.info(f"Prediction for {product_id}: {predicted} units")
    return {
        "product_id": product_id,
        "predicted_units_sold": predicted,
        "model_used": type(model).__name__,
    }
