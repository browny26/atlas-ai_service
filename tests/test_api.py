import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client
from app.core.config import Settings
from app.llm.base import LLMError
from tests.conftest import FakeLLM, make_plan

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


def test_stream_endpoint_sends_server_sent_events(client: TestClient) -> None:
    with client.stream("POST", "/v1/itineraries/stream", json=PAYLOAD) as response:
        body = "".join(response.iter_text())

    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event:")
    ]
    assert events == ["day", "day", "result"]


def test_regenerate_day_endpoint(app: FastAPI, client: TestClient) -> None:
    plan = make_plan(days=2)
    new_day = {
        "day": 2,
        "morning": {"activity": "Vatican Museums", "time": "09:00", "cost_eur": 20},
        "afternoon": {"activity": "Castel Sant'Angelo", "time": "14:00", "cost_eur": 15},
        "evening": {"activity": "Dinner in Prati", "time": "20:00", "cost_eur": 30},
    }
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(json.dumps(new_day))

    response = client.post(
        "/v1/itineraries/regenerate-day",
        json={"trip": PAYLOAD, "plan": plan, "day": 2, "feedback": "less walking"},
    )

    assert response.status_code == 200
    assert response.json()["days"][1]["morning"]["activity"] == "Vatican Museums"


def test_invalid_request_returns_422(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json={**PAYLOAD, "days": 99})

    assert response.status_code == 422


def test_missing_api_key_returns_401(client: TestClient) -> None:
    response = client.post("/v1/itineraries", json=PAYLOAD, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_non_ascii_api_key_returns_401(client: TestClient) -> None:
    response = client.post(
        "/v1/itineraries", json=PAYLOAD, headers={b"X-API-Key": "clé".encode("latin-1")}
    )

    assert response.status_code == 401


def test_provider_down_without_fallback_returns_502(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    settings.enable_fallback = False
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(LLMError("down"))

    response = client.post("/v1/itineraries", json=PAYLOAD)

    assert response.status_code == 502
    assert "detail" in response.json()
