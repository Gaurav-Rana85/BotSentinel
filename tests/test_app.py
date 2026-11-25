
import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    # Starts the FastAPI lifespan and loads the trained model.
    with TestClient(app) as test_client:
        yield test_client


def valid_traffic():
    return {
        "sport": 51514,
        "dport": 80,
        "pkts": 10,
        "bytes": 1200,
        "dur": 1.5,
        "mean": 0.15,
        "stddev": 0.05,
        "sum": 1.5,
        "min": 0.1,
        "max": 0.3,
        "spkts": 6,
        "dpkts": 4,
        "sbytes": 700,
        "dbytes": 500,
        "rate": 6.67,
        "srate": 4.0,
        "drate": 2.67,
        "flgs": "e",
        "proto": "tcp",
        "state": "CON"
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_dashboard_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "BotSentinel" in response.text


def test_prediction_returns_valid_result(client):
    response = client.post("/predict", json=valid_traffic())

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["label"] in ["Normal", "Attack"]
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["recommendations"], list)


def test_invalid_port_returns_422(client):
    traffic = valid_traffic()
    traffic["sport"] = 70000

    response = client.post("/predict", json=traffic)
    assert response.status_code == 422
