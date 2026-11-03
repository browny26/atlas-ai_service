"""Pydantic models: the single source of truth for the service's data contracts.

There are three groups:
- ItineraryRequest: what the backend sends us (validated input).
- LLMItinerary: what the model must return. Its JSON Schema is embedded in the prompt,
  and the same model validates the answer, so prompt and parser can never drift apart.
- ItineraryResponse: what we return. It adds values computed in Python (total cost,
  budget status) and metadata about how the answer was produced.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_DAYS = 14

Time = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Interest = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Euros = Annotated[float, Field(ge=0, le=100_000)]


# ---------- Input ----------


class ItineraryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "destination": "Rome, Italy",
                    "days": 3,
                    "budget_eur": 600,
                    "interests": ["history", "food"],
                }
            ]
        }
    )

    destination: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=100)
    ]
    days: int = Field(ge=1, le=MAX_DAYS)
    budget_eur: float = Field(gt=0, le=100_000, description="Total trip budget in euros.")
    interests: list[Interest] = Field(default_factory=list, max_length=10)


# ---------- LLM output (structured output schema) ----------


class Activity(BaseModel):
    activity: Text
    time: Time = Field(description="Start time, 24h format HH:MM.")
    cost_eur: Euros = Field(description="Cost per person in euros, as a number.")


class DayPlan(BaseModel):
    day: int = Field(ge=1, le=MAX_DAYS)
    morning: Activity
    afternoon: Activity
    evening: Activity

    @property
    def activities(self) -> tuple[Activity, Activity, Activity]:
        return (self.morning, self.afternoon, self.evening)


class Accommodation(BaseModel):
    name: Text
    type: Text = Field(description="E.g. hotel, B&B, hostel, apartment.")
    details: Text = Field(description="Neighbourhood and why it fits the trip.")
    cost_per_night_eur: Euros


class LLMItinerary(BaseModel):
    """The JSON object the model is asked to produce."""

    days: list[DayPlan] = Field(min_length=1, max_length=MAX_DAYS)
    accommodation: Accommodation
    tips: list[Text] = Field(min_length=1, max_length=6)

    @field_validator("days")
    @classmethod
    def days_are_consecutive(cls, days: list[DayPlan]) -> list[DayPlan]:
        numbers = [d.day for d in days]
        if numbers != list(range(1, len(days) + 1)):
            raise ValueError(f"days must be numbered 1..{len(days)} in order, got {numbers}")
        return days


# ---------- Output ----------


class BudgetStatus(StrEnum):
    WITHIN_BUDGET = "within_budget"
    OVER_BUDGET = "over_budget"


class Source(StrEnum):
    LLM = "llm"
    FALLBACK = "fallback"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerationMetadata(BaseModel):
    source: Source
    model: str | None = None
    prompt_version: str
    attempts: int
    latency_ms: int
    usage: TokenUsage = TokenUsage()


class ItineraryResponse(BaseModel):
    destination: str
    total_days: int
    currency: str = "EUR"
    budget_eur: float
    nights: int
    total_estimated_cost_eur: float
    budget_status: BudgetStatus
    days: list[DayPlan]
    accommodation: Accommodation
    tips: list[str]
    metadata: GenerationMetadata
