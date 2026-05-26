import os
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agents.agent_service import ask
from backend.agents.mcp import list_tools
from backend.services.azure_openai import chat
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")


class ChatRequest(BaseModel):
    message: str


@router.post("/api/agent/chat")
async def agent_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        return ask(req.message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/agent/tools")
async def get_tools():
    return {"tools": list_tools()}


@router.get("/api/dashboard")
async def dashboard():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No data available. Upload via POST /api/ingest.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required_cols = {"revenue", "units_sold", "product_name", "category", "region"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Dataset missing required columns: {sorted(missing_cols)}")

    monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().reset_index()
    monthly["date"] = monthly["date"].astype(str)

    total_revenue = round(float(df["revenue"].sum()), 2)
    total_units_sold = int(df["units_sold"].sum())

    top_products = (
        df.groupby("product_name")["revenue"].sum()
        .sort_values(ascending=False).head(5)
        .reset_index().rename(columns={"revenue": "total_revenue"})
        .to_dict(orient="records")
    )
    sales_by_category = df.groupby("category")["revenue"].sum().round(2).to_dict()
    sales_by_region = df.groupby("region")["revenue"].sum().round(2).to_dict()
    monthly_records = monthly.to_dict(orient="records")

    response = {
        "total_revenue": total_revenue,
        "total_units_sold": total_units_sold,
        "top_products": top_products,
        "sales_by_category": sales_by_category,
        "sales_by_region": sales_by_region,
        "monthly_trend": monthly_records,
    }

    if settings.USE_AZURE_OPENAI:
        try:
            top_names = ", ".join(p["product_name"] for p in top_products[:3])
            bottom_names = ", ".join(p["product_name"] for p in top_products[-2:]) if len(top_products) >= 2 else ""
            top_cat = max(sales_by_category, key=sales_by_category.get)
            bottom_cat = min(sales_by_category, key=sales_by_category.get)
            top_region = max(sales_by_region, key=sales_by_region.get)
            bottom_region = min(sales_by_region, key=sales_by_region.get)
            avg_rev_per_unit = round(total_revenue / max(total_units_sold, 1), 2)

            monthly_vals = [r["revenue"] for r in monthly_records]
            if len(monthly_vals) >= 2:
                trend_dir = "upward" if monthly_vals[-1] > monthly_vals[-2] else (
                    "downward" if monthly_vals[-1] < monthly_vals[-2] else "flat"
                )
            else:
                trend_dir = "flat"

            cat_breakdown = " | ".join(
                f"{c}: Rs.{v:,.0f}" for c, v in sorted(sales_by_category.items(), key=lambda x: -x[1])
            )
            reg_breakdown = " | ".join(
                f"{r}: Rs.{v:,.0f}" for r, v in sorted(sales_by_region.items(), key=lambda x: -x[1])
            )

            gpt_insight = chat(
                (
                    "You are a Chief Retail Analytics Officer preparing an executive briefing. "
                    "You are given live KPI data from a retail analytics system. "
                    "Do NOT list the numbers back — interpret them. "
                    "Identify: (1) the strongest performing segment and why it likely leads, "
                    "(2) the weakest segment and what risk it represents, "
                    "(3) what the monthly revenue trend signals about business momentum. "
                    "Then give two specific, actionable strategic recommendations grounded in these exact numbers. "
                    "Write in executive briefing style — confident, analytical, no filler."
                ),
                (
                    f"RETAIL PERFORMANCE SNAPSHOT\n"
                    f"Total Revenue: Rs.{total_revenue:,.2f} | Units Sold: {total_units_sold:,} | "
                    f"Avg Revenue/Unit: Rs.{avg_rev_per_unit}\n"
                    f"Monthly Trend: {trend_dir} (latest vs previous month)\n\n"
                    f"TOP PRODUCTS: {top_names}\n"
                    f"UNDERPERFORMING: {bottom_names}\n\n"
                    f"CATEGORY REVENUE: {cat_breakdown}\n"
                    f"Best: {top_cat} (Rs.{sales_by_category[top_cat]:,.2f}) | "
                    f"Weakest: {bottom_cat} (Rs.{sales_by_category[bottom_cat]:,.2f})\n\n"
                    f"REGIONAL REVENUE: {reg_breakdown}\n"
                    f"Best: {top_region} (Rs.{sales_by_region[top_region]:,.2f}) | "
                    f"Weakest: {bottom_region} (Rs.{sales_by_region[bottom_region]:,.2f})\n\n"
                    f"Provide executive-level strategic analysis and two specific recommendations."
                ),
                max_tokens=280,
            )
            if gpt_insight:
                response["gpt_insight"] = gpt_insight
        except Exception as e:
            logger.warning(f"GPT dashboard insight failed: {e}")

    return response
