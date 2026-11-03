"""Provider-agnostic LLM interface.

The service layer depends on this Protocol, not on a concrete SDK. That keeps the business
logic testable (tests inject a fake) and lets us swap providers through configuration.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class JsonSchema:
    """Expected output structure. Providers that support it enforce it natively."""

    name: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)


class LLMError(Exception):
    """The provider could not be reached or returned an error (after transport retries)."""


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> LLMResult:
        """Return the model's whole answer as a JSON object."""
        ...

    def stream_json(
        self, messages: list[ChatMessage], *, schema: JsonSchema | None = None
    ) -> AsyncIterator[str]:
        """Yield the model's JSON answer chunk by chunk, as it is generated."""
        ...
