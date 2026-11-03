"""LLMClient for any OpenAI-compatible Chat Completions API (Groq, Ollama, OpenAI, ...)."""

import logging
from collections.abc import AsyncIterator
from typing import Literal, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema

from app.llm.base import (
    ChatMessage,
    JsonSchema,
    LLMError,
    LLMOutputRejectedError,
    LLMResult,
    StreamChunk,
    Usage,
)

logger = logging.getLogger(__name__)

ResponseFormat = Literal["json_object", "json_schema"]

# Error codes meaning "the model wrote an invalid answer", not "the provider is down".
# Groq validates JSON mode server-side and returns the rejected text as failed_generation.
_REJECTED_OUTPUT_CODES = {"json_validate_failed"}


def _to_llm_error(exc: openai.APIError) -> LLMError:
    if exc.code not in _REJECTED_OUTPUT_CODES:
        return LLMError(str(exc))
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    failed = error.get("failed_generation") if isinstance(error, dict) else None
    failed_output = failed if isinstance(failed, str) else ""
    logger.warning("Provider rejected the model output (%s): %.300s", exc.code, failed_output)
    return LLMOutputRejectedError(str(exc), failed_output=failed_output)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        response_format: ResponseFormat = "json_object",
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._response_format = response_format
        # The SDK retries timeouts, 429 and 5xx with exponential backoff.
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",  # Ollama ignores the key, the SDK requires one
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def model(self) -> str:
        return self._model

    def _format(
        self, schema: JsonSchema | None
    ) -> ResponseFormatJSONObject | ResponseFormatJSONSchema:
        # json_schema: the provider constrains generation to the schema (structured outputs),
        # only on models that support it. json_object (default): valid JSON is guaranteed and
        # the structure is checked by Pydantic in the service layer either way.
        if self._response_format == "json_schema" and schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.name, "schema": schema.schema, "strict": False},
            }
        return {"type": "json_object"}

    async def complete_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> LLMResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=self._format(schema),
            )
        except openai.APIError as exc:
            logger.warning("LLM provider error: %s", exc)
            raise _to_llm_error(exc) from exc

        if not response.choices:
            raise LLMError("LLM returned no choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("LLM output truncated by max_tokens=%d", self._max_tokens)

        usage = (
            Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            if response.usage
            else Usage()
        )
        return LLMResult(
            content=choice.message.content or "",
            model=response.model or self._model,
            usage=usage,
        )

    async def stream_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> AsyncIterator[StreamChunk]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=self._format(schema),
                stream=True,
                # Without this, streamed responses report no token usage.
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                usage = (
                    Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                    if chunk.usage
                    else None
                )
                if content or usage:
                    yield StreamChunk(content=content or "", usage=usage)
        except openai.APIError as exc:
            logger.warning("LLM provider error while streaming: %s", exc)
            raise _to_llm_error(exc) from exc

    async def aclose(self) -> None:
        await self._client.close()
