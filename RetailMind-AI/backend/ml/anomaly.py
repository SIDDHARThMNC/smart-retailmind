import os
import logging
from sklearn.ensemble import IsolationForest
from backend.ml.preprocessing import load_data, clean_data

logger = logging.getLogger(__name__)

_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def detect_anomalies(data_path=None, contamination=0.02,
                     high_sales_multiplier=3.0, limit=50):
    if data_path is None:
        data_path = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")
    df = clean_data(load_data(data_path))

    contamination = max(0.01, min(float(contamination), 0.5))
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    X = df[["price", "discount", "units_sold", "revenue"]].fillna(0)

    df["anomaly_flag"] = iso.fit_predict(X)
    df["anomaly_score"] = iso.decision_function(X)

    anom = df[df["anomaly_flag"] == -1].copy()
    median_units = df["units_sold"].median()

    anom["anomaly_type"] = anom["units_sold"].apply(
        lambda x: "High Sales Anomaly" if x > median_units * high_sales_multiplier else "Low Sales Anomaly"
    )

    def get_reason(row):
        if row["anomaly_type"] == "High Sales Anomaly":
            if row["discount"] >= 20:
                return "Unusually high sales, likely driven by a large discount."
            return "Sudden spike in units sold, significantly above normal range."
        if row["units_sold"] == 0:
            return "Zero units sold, possible stock-out or data issue."
        return "Unexpectedly low sales, may indicate supply or demand issue."

    anom["reason"] = anom.apply(get_reason, axis=1)
    anom["predicted_units_sold"] = round(median_units, 1)

    result = (
        anom[["date", "product_id", "product_name", "units_sold", "predicted_units_sold",
              "anomaly_type", "anomaly_score", "reason"]]
        .rename(columns={"units_sold": "actual_units_sold"})
        .copy()
    )
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date").reset_index(drop=True)

    high = int((result["anomaly_type"] == "High Sales Anomaly").sum())
    low = int((result["anomaly_type"] == "Low Sales Anomaly").sum())
    logger.info(f"Detected {len(result)} anomalies ({high} high, {low} low)")

    return {
        "total_anomalies": len(result),
        "high_sales_anomalies": high,
        "low_sales_anomalies": low,
        "anomalies": result.head(limit).to_dict(orient="records"),
    }
