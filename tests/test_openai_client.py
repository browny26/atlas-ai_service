"""Tests the real client against a mocked HTTP transport: no network, no tokens spent."""

import json

import httpx2  # HTTP library used by the openai SDK v3
import pytest
from openai import AsyncOpenAI

from app.llm.base import JsonSchema, LLMError
from app.llm.openai_compatible import OpenAICompatibleClient, ResponseFormat


def make_client(
    handler: httpx2.MockTransport, response_format: ResponseFormat = "json_object"
) -> OpenAICompatibleClient:
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
        response_format=response_format,
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


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


async def test_json_schema_mode_sends_the_schema() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        return httpx2.Response(200, json=_completion("{}"))

    client = make_client(httpx2.MockTransport(handler), response_format="json_schema")
    schema = JsonSchema(name="thing", schema={"type": "object"})
    await client.complete_json([{"role": "user", "content": "hi"}], schema=schema)

    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "thing", "schema": {"type": "object"}, "strict": False},
    }


async def test_stream_yields_content_and_usage() -> None:
    sent: dict[str, object] = {}

    def chunk(text: str | None, usage: dict[str, int] | None = None) -> str:
        choices = [] if text is None else [{"index": 0, "delta": {"content": text}}]
        body = {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": choices,
            "usage": usage,
        }
        return f"data: {json.dumps(body)}\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        body = chunk('{"a"') + chunk(": 1}") + chunk(None, usage) + "data: [DONE]\n\n"
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = make_client(httpx2.MockTransport(handler))
    chunks = [c async for c in client.stream_json([{"role": "user", "content": "hi"}])]

    assert sent["stream_options"] == {"include_usage": True}
    assert "".join(c.content for c in chunks) == '{"a": 1}'
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 15
