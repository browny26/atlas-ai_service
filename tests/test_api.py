from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client
from app.core.config import Settings
from app.llm.base import LLMError
from tests.conftest import FakeLLM

PAYLOAD = {"destination": "Rome", "days": 2, "budget_eur": 500, "interests": ["food"]}


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_itinerary(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["destination"] == "Rome"
    assert len(body["days"]) == 2
    assert body["budget_status"] == "within_budget"
    assert body["metadata"]["source"] == "llm"


def test_invalid_request_returns_422(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json={**PAYLOAD, "days": 99})

    assert response.status_code == 422


def test_missing_api_key_returns_401(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json=PAYLOAD, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_non_ascii_api_key_returns_401(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json=PAYLOAD, headers={"X-API-Key": "clé".encode("latin-1")})

    assert response.status_code == 401


def test_provider_down_without_fallback_returns_502(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    settings.enable_fallback = False
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(LLMError("down"))

    response = client.post("/v1/itineraries", json=PAYLOAD)

    assert response.status_code == 502
    assert "detail" in response.json()
