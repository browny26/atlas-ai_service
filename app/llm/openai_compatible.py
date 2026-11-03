"""LLMClient for any OpenAI-compatible Chat Completions API (Groq, Ollama, OpenAI, ...)."""

import logging
from typing import cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.llm.base import ChatMessage, LLMError, LLMResult, Usage

logger = logging.getLogger(__name__)


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
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
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

    async def complete_json(self, messages: list[ChatMessage]) -> LLMResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                # JSON mode: the provider guarantees syntactically valid JSON.
                # The structure is still checked by Pydantic in the service layer.
                response_format={"type": "json_object"},
            )
        except openai.APIError as exc:
            logger.warning("LLM provider error: %s", exc)
            raise LLMError(str(exc)) from exc

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

    async def aclose(self) -> None:
        await self._client.close()
