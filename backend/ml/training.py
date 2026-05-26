import os
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.ml.preprocessing import load_data, clean_data, create_features

logger = logging.getLogger(__name__)

_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(_APP_ROOT, "backend", "saved_models", "demand_model.pkl")
PREPROCESSOR_PATH = os.path.join(_APP_ROOT, "backend", "saved_models", "preprocessor.pkl")


def train_models(
    data_path=None,
    test_size=0.2,
    n_estimators=500,
    max_depth=None,
    random_state=42,
):
    if data_path is None:
        data_path = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")
    df = clean_data(load_data(data_path))

    p99 = df["units_sold"].quantile(0.99)
    df = df[df["units_sold"] <= p99].reset_index(drop=True)
    logger.info(f"After outlier removal: {len(df)} rows")

    X, y, feature_cols, encoders, _ = create_features(df)

    test_size = max(0.1, min(float(test_size), 0.4))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth else None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    preds = np.maximum(model.predict(X_test), 0)
    mae = round(float(mean_absolute_error(y_test, preds)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
    r2 = round(float(r2_score(y_test, preds)), 4)
    logger.info(f"MAE={mae} | RMSE={rmse} | R2={r2}")

    os.makedirs(os.path.join(_APP_ROOT, "backend", "saved_models"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, PREPROCESSOR_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")

    return {
        "best_model": "RandomForestRegressor",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "all_models": [{"model": "RandomForestRegressor", "mae": mae, "rmse": rmse, "r2": r2}],
        "trained_models": ["RandomForestRegressor"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = train_models()
    print(f"Model: {r['best_model']}")
    print(f"R2: {r['r2']}  RMSE: {r['rmse']}  MAE: {r['mae']}")
    print(f"Saved to: {MODEL_PATH}")
