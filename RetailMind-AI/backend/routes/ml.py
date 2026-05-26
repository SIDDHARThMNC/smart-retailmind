import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.ml.training import train_models
from backend.ml.prediction import predict_demand
from backend.ml.anomaly import detect_anomalies
from backend.services.azure_openai import chat
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ALLOWED_DATA_PATH = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")


def _gpt(system: str, user_msg: str, max_tokens: int = 200):
    if not settings.USE_AZURE_OPENAI:
        return None
    try:
        return chat(system, user_msg, max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"GPT insight failed: {e}")
        return None


class TrainRequest(BaseModel):
    data_path: str = Field(None)
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    n_estimators: int = Field(500, ge=10, le=2000)
    random_state: int = Field(42)


@router.post("/api/train")
async def train(req: TrainRequest):
    try:
        # Restrict data_path to the known safe location — prevents path traversal
        safe_path = _ALLOWED_DATA_PATH if req.data_path is None else None
        if req.data_path is not None:
            requested = os.path.realpath(req.data_path)
            allowed = os.path.realpath(_ALLOWED_DATA_PATH)
            if requested != allowed:
                raise HTTPException(
                    status_code=403,
                    detail="Custom data_path is not permitted. Use POST /api/ingest to upload data first.",
                )
            safe_path = req.data_path

        result = train_models(
            data_path=safe_path,
            test_size=req.test_size,
            n_estimators=req.n_estimators,
            random_state=req.random_state,
        )
        logger.info(f"Training complete: R2={result['r2']} RMSE={result['rmse']}")

        response = {
            "message": "Model trained and saved successfully.",
            "best_model": result["best_model"],
            "metrics": {
                "r2": result["r2"],
                "rmse": result["rmse"],
                "mae": result["mae"],
            },
            "trained_models": result["trained_models"],
        }

        r2_quality = (
            "excellent" if result["r2"] >= 0.85 else
            "good" if result["r2"] >= 0.70 else
            "moderate" if result["r2"] >= 0.50 else "poor"
        )
        rmse_context = (
            "low error relative to typical retail demand volumes" if result["rmse"] < 20 else
            "moderate prediction error" if result["rmse"] < 50 else
            "high prediction error — model may need more data or tuning"
        )

        gpt_insight = _gpt(
            (
                "You are a senior ML engineer presenting model results to a retail business team. "
                "Your job is to interpret the exact metric values provided — do NOT give generic definitions. "
                "Explain what these specific numbers mean for retail demand forecasting accuracy. "
                "State clearly whether the model is production-ready, needs improvement, or is excellent. "
                "Mention what the RMSE means in practical terms (units of stock). "
                "End with one concrete recommendation for the business team."
            ),
            (
                f"Model: {result['best_model']} | Trees: {req.n_estimators} | Test split: {req.test_size}\n"
                f"R² Score: {result['r2']} ({r2_quality} — explains {round(result['r2']*100, 1)}% of demand variance)\n"
                f"RMSE: {result['rmse']} units ({rmse_context})\n"
                f"MAE: {result['mae']} units (average prediction is off by {result['mae']} units per record)\n"
                f"Interpret these specific values for a retail operations manager making stocking decisions."
            ),
            max_tokens=220,
        )
        if gpt_insight:
            response["gpt_insight"] = gpt_insight

        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "P101"})
    date: str = Field(..., json_schema_extra={"example": "2025-06-15"})
    price: float = Field(..., gt=0)
    discount: float = Field(0.0, ge=0.0, le=100.0)
    store_id: str = Field(..., json_schema_extra={"example": "S01"})
    region: str = Field(..., json_schema_extra={"example": "North"})


@router.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        result = predict_demand(
            product_id=req.product_id,
            date_str=req.date,
            price=req.price,
            discount=req.discount,
            store_id=req.store_id,
            region=req.region,
        )

        response = {
            "message": "Prediction successful.",
            "product_id": result["product_id"],
            "predicted_units_sold": result["predicted_units_sold"],
            "model_used": result["model_used"],
            "input": {
                "date": req.date,
                "price": req.price,
                "discount": req.discount,
                "store_id": req.store_id,
                "region": req.region,
            },
        }

        effective_price = round(req.price * (1 - req.discount / 100), 2)
        demand_level = (
            "high" if result["predicted_units_sold"] >= 50 else
            "moderate" if result["predicted_units_sold"] >= 20 else "low"
        )
        discount_signal = (
            "significant discount applied" if req.discount >= 20 else
            "moderate discount" if req.discount >= 10 else "no meaningful discount"
        )

        gpt_insight = _gpt(
            (
                "You are a retail demand analyst with expertise in pricing and inventory strategy. "
                "You are given the output of a RandomForest ML model predicting retail demand. "
                "Do NOT restate the inputs — analyze them. "
                "Explain WHY demand is at this level based on the price point, discount, region, and timing. "
                "Identify the dominant factor driving this prediction. "
                "Give one specific inventory or pricing recommendation for this store and region. "
                "Be direct, numerical, and business-focused."
            ),
            (
                f"Product: {req.product_id} | Store: {req.store_id} | Region: {req.region}\n"
                f"Date: {req.date} | Listed Price: Rs.{req.price} | Discount: {req.discount}% | "
                f"Effective Price: Rs.{effective_price} ({discount_signal})\n"
                f"ML Predicted Units Sold: {result['predicted_units_sold']} (demand level: {demand_level})\n"
                f"Model: {result['model_used']}\n"
                f"Analyze this prediction and provide a specific business recommendation."
            ),
            max_tokens=200,
        )
        if gpt_insight:
            response["gpt_insight"] = gpt_insight

        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/anomalies")
async def anomalies(contamination: float = 0.02, limit: int = 50):
    if not (0.01 <= contamination <= 0.5):
        raise HTTPException(status_code=422, detail="contamination must be between 0.01 and 0.5")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")

    try:
        result = detect_anomalies(contamination=contamination, limit=limit)

        response = {
            "message": "Anomaly detection complete.",
            "total_anomalies": result["total_anomalies"],
            "high_sales_anomalies": result["high_sales_anomalies"],
            "low_sales_anomalies": result["low_sales_anomalies"],
            "anomalies": result["anomalies"],
        }

        top5 = result["anomalies"][:5]
        top5_text = "\n".join(
            f"  [{a['anomaly_type']}] {a['date']} | {a['product_id']} | "
            f"Actual: {a['actual_units_sold']} units | Expected: ~{a['predicted_units_sold']} units | "
            f"Score: {round(a['anomaly_score'], 4)} | {a['reason']}"
            for a in top5
        ) if top5 else "No anomalies detected."

        high_pct = round(result["high_sales_anomalies"] / max(result["total_anomalies"], 1) * 100, 1)
        low_pct = round(result["low_sales_anomalies"] / max(result["total_anomalies"], 1) * 100, 1)
        dominant = (
            "demand spikes / promotional events"
            if result["high_sales_anomalies"] > result["low_sales_anomalies"]
            else "stockouts / supply issues"
        )

        gpt_insight = _gpt(
            (
                "You are a retail operations analyst specializing in sales anomaly investigation. "
                "You are given IsolationForest anomaly detection results with actual vs expected unit figures. "
                "Do NOT explain what anomaly detection is — analyze the specific results provided. "
                "Identify the dominant pattern across the anomalies. "
                "For high-sales anomalies: explain likely causes (promotions, viral demand, data errors). "
                "For low-sales anomalies: explain likely causes (stockouts, pricing issues, regional demand drops). "
                "Prioritize which anomalies need immediate operational attention based on the deviation magnitude. "
                "End with exactly two concrete actions the operations team should take this week."
            ),
            (
                f"IsolationForest run | contamination={contamination} | Total flagged: {result['total_anomalies']}\n"
                f"High Sales Anomalies: {result['high_sales_anomalies']} ({high_pct}%) — likely {dominant}\n"
                f"Low Sales Anomalies:  {result['low_sales_anomalies']} ({low_pct}%)\n"
                f"Top anomalies (actual vs expected):\n{top5_text}\n"
                f"Provide root cause analysis and prioritized operational recommendations."
            ),
            max_tokens=250,
        )
        if gpt_insight:
            response["gpt_insight"] = gpt_insight

        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
