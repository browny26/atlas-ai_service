"""Provider-agnostic LLM interface.

The service layer depends on this Protocol, not on a concrete SDK. That keeps the business
logic testable (tests inject a fake) and lets us swap providers through configuration.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


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

    async def complete_json(self, messages: list[ChatMessage]) -> LLMResult:
        """Return the model's answer, asking the provider for a JSON object."""
        ...
