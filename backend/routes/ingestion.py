import io
import os
import logging
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from backend.ml.preprocessing import validate_columns, clean_data
from backend.database.db import insert_sales_data
from backend.services.azure_openai import chat
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/api/ingest")
async def ingest_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    try:
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        validate_columns(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    df_clean = clean_data(df_raw)
    os.makedirs(os.path.join(_APP_ROOT, "data"), exist_ok=True)
    df_clean.to_csv(DATA_PATH, index=False)
    logger.info(f"Saved cleaned data: {len(df_clean)} rows")

    try:
        insert_sales_data(df_clean)
    except Exception as e:
        logger.warning(f"MongoDB insert warning: {e}")

    response = {
        "message": "File uploaded and cleaned successfully.",
        "raw_rows": len(df_raw),
        "cleaned_rows": len(df_clean),
    }

    if settings.USE_AZURE_OPENAI:
        try:
            dropped = len(df_raw) - len(df_clean)
            drop_pct = round(dropped / max(len(df_raw), 1) * 100, 1)
            categories = df_clean["category"].value_counts().to_dict() if "category" in df_clean.columns else {}
            regions = df_clean["region"].value_counts().to_dict() if "region" in df_clean.columns else {}
            date_range = ""
            if "date" in df_clean.columns:
                dates = pd.to_datetime(df_clean["date"], errors="coerce").dropna()
                date_range = f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "unknown"

            gpt_summary = chat(
                (
                    "You are a retail data engineer reviewing an ingested dataset for ML readiness. "
                    "You are given cleaning statistics and dataset composition. "
                    "Do NOT just describe the numbers — assess them. "
                    "State whether the data quality is acceptable for ML training (yes/no and why). "
                    "Flag any concerns: high drop rate, missing date coverage, category/region imbalance. "
                    "End with one specific recommendation to improve data quality if needed, "
                    "or confirm the dataset is ready for model training."
                ),
                (
                    f"File: {file.filename}\n"
                    f"Raw rows: {len(df_raw)} → Cleaned rows: {len(df_clean)} "
                    f"(dropped {dropped} rows = {drop_pct}% loss)\n"
                    f"Date coverage: {date_range}\n"
                    f"Category distribution: {categories}\n"
                    f"Region distribution: {regions}\n"
                    f"Assess ML readiness and flag any data quality concerns."
                ),
                max_tokens=180,
            )
            if gpt_summary:
                response["gpt_summary"] = gpt_summary
        except Exception as e:
            logger.warning(f"GPT ingest summary failed: {e}")

    return response
