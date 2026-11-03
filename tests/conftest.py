import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_llm_client
from app.core.config import Settings, get_settings
from app.llm.base import ChatMessage, JsonSchema, LLMResult, Usage
from app.main import create_app


class FakeLLM:
    """Scripted LLMClient: returns (or raises) the given replies in order, records calls.

    stream_json() sends the same replies split into small chunks, like a real stream.
    """

    def __init__(self, *replies: str | Exception, chunk_size: int = 7) -> None:
        self._replies = list(replies)
        self._chunk_size = chunk_size
        self.calls: list[list[ChatMessage]] = []
        self.schemas: list[JsonSchema | None] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def _next(self, messages: list[ChatMessage], schema: JsonSchema | None) -> str:
        self.calls.append(list(messages))
        self.schemas.append(schema)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def complete_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> LLMResult:
        content = self._next(messages, schema)
        return LLMResult(content=content, model=self.model, usage=Usage(100, 50))

    async def stream_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> AsyncIterator[str]:
        content = self._next(messages, schema)
        for start in range(0, len(content), self._chunk_size):
            yield content[start : start + self._chunk_size]


def make_plan(days: int = 2, activity_cost: float = 20, night_cost: float = 80) -> dict[str, Any]:
    def activity(name: str, time: str) -> dict[str, Any]:
        return {"activity": name, "time": time, "cost_eur": activity_cost}

    return {
        "days": [
            {
                "day": n,
                "morning": activity(f"Museum {n}", "09:00"),
                "afternoon": activity(f"Park {n}", "14:00"),
                "evening": activity(f"Dinner {n}", "20:00"),
            }
            for n in range(1, days + 1)
        ],
        "accommodation": {
            "name": "Hotel Test",
            "type": "hotel",
            "details": "City centre",
            "cost_per_night_eur": night_cost,
        },
        "tips": ["Book in advance"],
    }


def plan_json(**kwargs: Any) -> str:
    return json.dumps(make_plan(**kwargs))


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, service_api_key="test-key")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(plan_json(days=2))


@pytest.fixture
def app(settings: Settings, fake_llm: FakeLLM) -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    # Not used as a context manager: the lifespan (real LLM client) is not started.
    return TestClient(app, headers={"X-API-Key": "test-key"})
