"""Tests the real client against a mocked HTTP transport: no network, no tokens spent."""

import json

import httpx2  # HTTP library used by the openai SDK v3
import pytest
from openai import AsyncOpenAI

from app.llm.base import LLMError
from app.llm.openai_compatible import OpenAICompatibleClient


def make_client(handler: httpx2.MockTransport) -> OpenAICompatibleClient:
    sdk = AsyncOpenAI(
        base_url="https://llm.test/v1",
        api_key="key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=handler),
    )
    return OpenAICompatibleClient(
        base_url="unused",
        api_key="unused",
        model="test-model",
        temperature=0.3,
        max_tokens=100,
        timeout_seconds=5,
        max_retries=0,
        client=sdk,
    )


async def test_sends_json_mode_request_and_parses_answer() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = make_client(httpx2.MockTransport(handler))
    result = await client.complete_json([{"role": "user", "content": "hi"}])

    assert sent["response_format"] == {"type": "json_object"}
    assert sent["model"] == "test-model"
    assert result.content == '{"ok": true}'
    assert result.usage.total_tokens == 15


async def test_provider_error_becomes_llm_error() -> None:
    client = make_client(httpx2.MockTransport(lambda _: httpx2.Response(500, json={})))

    with pytest.raises(LLMError):
        await client.complete_json([{"role": "user", "content": "hi"}])
