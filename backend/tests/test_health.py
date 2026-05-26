import sys
sys.path.insert(0, ".")

from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "running"


def test_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200
