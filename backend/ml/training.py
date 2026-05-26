import os
import logging
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from backend.ml.preprocessing import (
    load_data,
    clean_data,
    create_features,
)

logger = logging.getLogger(__name__)

# Root path
_APP_ROOT = os.environ.get(
    "APP_ROOT",
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
)

# Save paths
MODEL_DIR = os.path.join(_APP_ROOT, "backend", "saved_models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "demand_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)


def train_models(
    data_path=None,
    test_size=0.2,
    n_estimators=30,
    max_depth=8,
    random_state=42,
):
    """
    Azure B1 optimized lightweight model training
    """

    if data_path is None:
        data_path = os.path.join(
            _APP_ROOT,
            "data",
            "cleaned_retail_sales.csv"
        )

    logger.info("Loading dataset...")

    # Load + clean
    df = clean_data(load_data(data_path))

    logger.info(f"Cleaned data: {len(df)} rows")

    # Remove extreme outliers
    p99 = df["units_sold"].quantile(0.99)

    df = df[df["units_sold"] <= p99].reset_index(drop=True)

    logger.info(f"After outlier removal: {len(df)} rows")

    # Feature engineering
    X, y, feature_cols, encoders, _ = create_features(df)

    # Safe test size
    test_size = max(0.1, min(float(test_size), 0.3))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        f"Train: {len(X_train)} | Test: {len(X_test)}"
    )

    # Lightweight Azure-safe RandomForest
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
    )

    logger.info(
        "Training optimized RandomForestRegressor..."
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = np.maximum(
        model.predict(X_test),
        0
    )

    # Metrics
    mae = round(
        float(mean_absolute_error(y_test, preds)),
        4
    )

    rmse = round(
        float(np.sqrt(mean_squared_error(y_test, preds))),
        4
    )

    r2 = round(
        float(r2_score(y_test, preds)),
        4
    )

    logger.info(
        f"MAE={mae} | RMSE={rmse} | R2={r2}"
    )

    # Create save directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save compressed lightweight model
    joblib.dump(
        model,
        MODEL_PATH,
        compress=3
    )

    # Save preprocessing metadata
    joblib.dump(
        {
            "encoders": encoders,
            "feature_cols": feature_cols,
        },
        PREPROCESSOR_PATH,
        compress=3,
    )

    logger.info(f"Model saved: {MODEL_PATH}")

    return {
        "best_model": "RandomForestRegressor",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "all_models": [
            {
                "model": "RandomForestRegressor",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        ],
        "trained_models": [
            "RandomForestRegressor"
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    result = train_models()

    print(
        f"Model: {result['best_model']}"
    )

    print(
        f"R2: {result['r2']}  "
        f"RMSE: {result['rmse']}  "
        f"MAE: {result['mae']}"
    )

    print(
        f"Saved to: {MODEL_PATH}"
    )