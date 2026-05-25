import logging
from backend.agents.mcp import call_tool
from backend.services.azure_openai import chat

logger = logging.getLogger(__name__)

_ML_KW = {
    "forecast", "prediction", "predict", "anomaly", "anomalies", "model",
    "accuracy", "r2", "rmse", "mae", "demand", "unusual", "spike", "machine learning"
}
_DOC_KW = {
    "policy", "refund", "return", "discount policy", "inventory",
    "warranty", "exchange", "terms", "condition", "guideline"
}
_DATA_KW = {
    "sales", "revenue", "trend", "top product", "best selling", "top selling",
    "category", "region", "store", "units sold", "monthly", "growth"
}

_AGENT_META = {
    "Data Analyst Agent": "Answers questions about sales, revenue, trends, and product performance.",
    "Document Assistant Agent": "Searches policy documents for return, refund, discount, and inventory info.",
    "ML Expert Agent": "Provides insights about ML models, demand forecasting, and anomaly detection.",
}

_SYSTEM_PROMPTS = {
    "Data Analyst Agent": (
        "You are a senior retail data analyst with deep expertise in sales performance and business intelligence. "
        "You are given structured sales data retrieved from a live retail analytics system. "
        "Do NOT restate the raw numbers — interpret them. "
        "Identify patterns, compare segments, explain what the data means for the business. "
        "If asked about top products, explain WHY they likely lead (price point, category demand, regional strength). "
        "If asked about trends, describe the direction and what it signals. "
        "Always ground your answer in the specific figures provided. "
        "Be analytical, direct, and business-focused. No filler."
    ),
    "Document Assistant Agent": (
        "You are a retail policy specialist who helps staff and customers understand company policies. "
        "You are given text retrieved from official policy documents. "
        "Answer the question directly and completely using ONLY the retrieved policy content. "
        "Be specific: mention exact conditions, timeframes, exceptions, and eligibility criteria from the policy. "
        "If the policy has multiple conditions, structure your answer clearly. "
        "If the retrieved content does not fully answer the question, say exactly what is and is not covered. "
        "Never invent policy details not present in the retrieved text."
    ),
    "ML Expert Agent": (
        "You are a machine learning engineer and retail analytics expert. "
        "You are given outputs from a RandomForest demand forecasting and IsolationForest anomaly detection system. "
        "Do NOT explain what these algorithms are — the user knows. "
        "Interpret the specific results: what do the prediction values mean for inventory planning? "
        "What do the anomaly patterns indicate about operational issues? "
        "Connect ML outputs to concrete business decisions: stocking levels, pricing adjustments, investigation priorities. "
        "Be precise, use the exact numbers provided, and give actionable recommendations."
    ),
}


def _route(message: str) -> str:
    q = message.lower()
    scores = {
        "ML Expert Agent": sum(1 for kw in _ML_KW if kw in q),
        "Document Assistant Agent": sum(1 for kw in _DOC_KW if kw in q),
        "Data Analyst Agent": sum(1 for kw in _DATA_KW if kw in q),
    }
    return max(scores, key=scores.get)


def route_question(question: str) -> str:
    return _route(question)


def ask(message: str) -> dict:
    agent_name = _route(message)
    mcp_result = call_tool(message)
    tool_data = mcp_result["result"]

    logger.info(f"Agent: {agent_name} | tool: {mcp_result['tool_used']}")

    try:
        system = _SYSTEM_PROMPTS[agent_name]
        user_msg = (
            f"Tool used: {mcp_result['tool_used']} — {mcp_result['tool_description']}\n"
            f"Retrieved data:\n{tool_data}\n\n"
            f"User question: {message}\n\n"
            f"Provide a complete, intelligent answer grounded in the retrieved data above."
        )
        response = chat(system, user_msg, max_tokens=350) or tool_data
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = tool_data

    return {
        "message": message,
        "agent": agent_name,
        "agent_description": _AGENT_META[agent_name],
        "tool_used": mcp_result["tool_used"],
        "tool_description": mcp_result["tool_description"],
        "response": response,
    }
