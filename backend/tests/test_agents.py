import sys
sys.path.insert(0, ".")

from starlette.testclient import TestClient
from backend.main import app
from backend.agents.agent_service import route_question

client = TestClient(app)

VALID_AGENTS = {"Data Analyst Agent", "Document Assistant Agent", "ML Expert Agent"}


# --- Routing logic tests ---

def test_route_forecast_to_ml():
    assert route_question("What is the forecast for product P101?") == "ML Expert Agent"

def test_route_anomaly_to_ml():
    assert route_question("Show me the anomalies in sales data") == "ML Expert Agent"

def test_route_model_accuracy_to_ml():
    assert route_question("Which model has the best accuracy?") == "ML Expert Agent"

def test_route_refund_to_document():
    assert route_question("What is the refund policy?") == "Document Assistant Agent"

def test_route_return_to_document():
    assert route_question("How do I return a product?") == "Document Assistant Agent"

def test_route_sales_to_data():
    assert route_question("What are the top selling products?") == "Data Analyst Agent"

def test_route_region_to_data():
    assert route_question("Which region has the highest sales?") == "Data Analyst Agent"


# --- POST /api/agent/chat ---

def test_agent_chat_returns_200():
    response = client.post("/api/agent/chat", json={"message": "What are the top selling products?"})
    assert response.status_code == 200

def test_agent_chat_response_fields():
    response = client.post("/api/agent/chat", json={"message": "What are the top selling products?"})
    data = response.json()
    assert "message" in data
    assert "agent" in data
    assert "response" in data

def test_agent_chat_valid_agent_name():
    response = client.post("/api/agent/chat", json={"message": "What are the top selling products?"})
    data = response.json()
    assert data["agent"] in VALID_AGENTS

def test_agent_chat_response_not_empty():
    response = client.post("/api/agent/chat", json={"message": "What are the top selling products?"})
    data = response.json()
    assert len(data["response"]) > 0

def test_agent_chat_empty_message_returns_400():
    response = client.post("/api/agent/chat", json={"message": ""})
    assert response.status_code == 400

def test_agent_chat_routes_to_ml():
    response = client.post("/api/agent/chat", json={"message": "Explain the demand forecast"})
    data = response.json()
    assert data["agent"] == "ML Expert Agent"

def test_agent_chat_routes_to_document():
    response = client.post("/api/agent/chat", json={"message": "What is the return policy?"})
    data = response.json()
    assert data["agent"] == "Document Assistant Agent"


# --- POST /api/search ---

def test_search_returns_200():
    response = client.post("/api/search", json={"query": "What is the return policy?"})
    assert response.status_code == 200

def test_search_response_fields():
    response = client.post("/api/search", json={"query": "What is the return policy?"})
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "sources" in data

def test_search_answer_not_empty():
    response = client.post("/api/search", json={"query": "What is the return policy?"})
    data = response.json()
    assert len(data["answer"]) > 0

def test_search_sources_is_list():
    response = client.post("/api/search", json={"query": "discount policy"})
    data = response.json()
    assert isinstance(data["sources"], list)

def test_search_empty_query_returns_400():
    response = client.post("/api/search", json={"query": ""})
    assert response.status_code == 400
