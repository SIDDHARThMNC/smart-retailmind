import os
import re
import logging
import joblib
import pandas as pd

from backend.rag.rag_service import search_documents

logger = logging.getLogger(__name__)
_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(_APP_ROOT, "data", "cleaned_retail_sales.csv")
_NO_DATA = "No sales data found. Upload data first using POST /api/ingest."


def _load_df():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def _tool_sales_summary(_):
    df = _load_df()
    if df is None:
        return _NO_DATA
    return (
        f"Sales Summary:\n"
        f"Total Revenue   : Rs. {df['revenue'].sum():,.2f}\n"
        f"Total Units Sold: {int(df['units_sold'].sum())}\n"
        f"Products        : {df['product_id'].nunique()}\n"
        f"Stores          : {df['store_id'].nunique()}"
    )


def _tool_top_products(_):
    df = _load_df()
    if df is None:
        return _NO_DATA
    top = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False).head(5)
    return "Top 5 products by revenue:\n" + "\n".join(f"{n}: Rs. {r:,.2f}" for n, r in top.items())


def _tool_revenue_by_category(_):
    df = _load_df()
    if df is None:
        return _NO_DATA
    cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    return "Revenue by category:\n" + "\n".join(f"{c}: Rs. {r:,.2f}" for c, r in cat.items())


def _tool_revenue_by_region(_):
    df = _load_df()
    if df is None:
        return _NO_DATA
    reg = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    return "Revenue by region:\n" + "\n".join(f"{r}: Rs. {v:,.2f}" for r, v in reg.items())


def _tool_monthly_trend(_):
    df = _load_df()
    if df is None:
        return _NO_DATA
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().tail(6)
    return "Monthly revenue (last 6 months):\n" + "\n".join(f"{p}: Rs. {r:,.2f}" for p, r in monthly.items())


def _tool_product_detail(params):
    pid = params.get("product_id", "").upper()
    if not pid:
        return "Please provide a product_id (e.g. P101)."
    df = _load_df()
    if df is None:
        return _NO_DATA
    sub = df[df["product_id"] == pid]
    if sub.empty:
        return f"No data found for product {pid}."
    return (
        f"Analysis for {sub['product_name'].iloc[0]} ({pid}):\n"
        f"Total Revenue   : Rs. {sub['revenue'].sum():,.2f}\n"
        f"Total Units Sold: {int(sub['units_sold'].sum())}\n"
        f"Average Price   : Rs. {sub['price'].mean():.2f}\n"
        f"Average Discount: {sub['discount'].mean():.1f}%"
    )


def _tool_search_policy(params):
    result = search_documents(params.get("query", ""))
    answer = result.get("answer", "No relevant information found.")
    sources = result.get("sources", [])
    return f"{answer}\n\nSource(s): {', '.join(sources)}" if sources else answer


def _tool_model_info(_):
    model_path = os.path.join(_APP_ROOT, "backend", "saved_models", "demand_model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return (
                f"Demand forecasting model: {type(model).__name__}\n"
                f"Model is trained and saved at {model_path}.\n"
                f"Use POST /api/predict to get a demand prediction.\n"
                f"Use POST /api/train to retrain the model."
            )
        except Exception:
            pass
    return (
        "Demand forecasting uses RandomForestRegressor.\n"
        "Use POST /api/train to train and POST /api/predict to get a prediction."
    )


def _tool_anomaly_info(_):
    try:
        from backend.ml.anomaly import detect_anomalies
        result = detect_anomalies(limit=5)
        top = result["anomalies"][:3]
        lines = [
            f"Anomaly Detection (IsolationForest) — {result['total_anomalies']} anomalies found:",
            f"  High Sales: {result['high_sales_anomalies']}  |  Low Sales: {result['low_sales_anomalies']}",
        ]
        for a in top:
            lines.append(f"  {a['date']} | {a['product_id']} | {a['anomaly_type']} | {a['reason']}")
        lines.append("Use GET /api/anomalies for the full report.")
        return "\n".join(lines)
    except Exception as e:
        return (
            f"Anomaly detection uses IsolationForest on the sales dataset.\n"
            f"Use GET /api/anomalies to get the full anomaly report. ({e})"
        )


def _tool_forecast_info(_):
    return (
        "Demand forecasting predicts units sold for a given product, date, store, and price.\n"
        "Use POST /api/train to train the model, then POST /api/predict to get a prediction.\n"
        'Example: {"product_id": "P101", "date": "2025-06-15", '
        '"price": 999.0, "discount": 10.0, "store_id": "S01", "region": "North"}'
    )


TOOL_REGISTRY = [
    {
        "name": "sales_summary",
        "description": "Get overall sales summary: total revenue, units sold, product count",
        "parameters": {},
        "keywords": ["sales", "summary", "total", "overview", "revenue", "units sold", "store"],
        "handler": _tool_sales_summary,
    },
    {
        "name": "top_products",
        "description": "Get top 5 products by revenue",
        "parameters": {},
        "keywords": ["top product", "best product", "best selling", "highest revenue", "top selling"],
        "handler": _tool_top_products,
    },
    {
        "name": "revenue_by_category",
        "description": "Get revenue breakdown by product category",
        "parameters": {},
        "keywords": ["category", "categories", "electronics", "fashion", "grocery", "home", "beauty",
                     "revenue by category", "by category"],
        "handler": _tool_revenue_by_category,
    },
    {
        "name": "revenue_by_region",
        "description": "Get revenue breakdown by region",
        "parameters": {},
        "keywords": ["region", "north", "south", "east", "west", "revenue by region", "by region"],
        "handler": _tool_revenue_by_region,
    },
    {
        "name": "monthly_trend",
        "description": "Get monthly revenue trend for the last 6 months",
        "parameters": {},
        "keywords": ["trend", "monthly", "over time", "growth", "month"],
        "handler": _tool_monthly_trend,
    },
    {
        "name": "product_detail",
        "description": "Get detailed analysis for a specific product by product_id",
        "parameters": {"product_id": "string, e.g. P101"},
        "keywords": ["p0", "p1", "p2", "p3", "p4"],
        "handler": _tool_product_detail,
    },
    {
        "name": "search_policy",
        "description": "Search policy documents: return, refund, discount, inventory",
        "parameters": {"query": "string"},
        "keywords": ["policy", "refund", "return", "discount policy", "inventory",
                     "warranty", "exchange", "terms", "condition", "guideline"],
        "handler": _tool_search_policy,
    },
    {
        "name": "model_info",
        "description": "Get information about ML model performance",
        "parameters": {},
        "keywords": ["model", "accuracy", "r2", "rmse", "mae", "performance", "machine learning"],
        "handler": _tool_model_info,
    },
    {
        "name": "anomaly_info",
        "description": "Get information about sales anomaly detection",
        "parameters": {},
        "keywords": ["anomaly", "anomalies", "spike", "unusual", "outlier"],
        "handler": _tool_anomaly_info,
    },
    {
        "name": "forecast_info",
        "description": "Get information about demand forecasting",
        "parameters": {},
        "keywords": ["forecast", "predict", "prediction", "demand"],
        "handler": _tool_forecast_info,
    },
]


def _select_tool(message: str) -> dict:
    q = message.lower()
    best, best_score = TOOL_REGISTRY[0], 0
    for tool in TOOL_REGISTRY:
        score = sum(1 for kw in tool["keywords"] if kw in q)
        if score > best_score:
            best_score, best = score, tool
    return best


def _extract_params(message: str, tool: dict) -> dict:
    params = {}
    if "product_id" in tool["parameters"]:
        m = re.search(r"p\d{3}", message.lower())
        if m:
            params["product_id"] = m.group(0).upper()
    if "query" in tool["parameters"]:
        params["query"] = message
    return params


def call_tool(message: str) -> dict:
    tool = _select_tool(message)
    params = _extract_params(message, tool)
    logger.info(f"MCP tool: {tool['name']} | params: {params}")
    try:
        result = tool["handler"](params)
    except Exception as e:
        logger.error(f"Tool error: {e}")
        result = f"Tool '{tool['name']}' error: {e}"
    return {
        "tool_used": tool["name"],
        "tool_description": tool["description"],
        "parameters": params,
        "result": result,
    }


def list_tools() -> list:
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in TOOL_REGISTRY
    ]
