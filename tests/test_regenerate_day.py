import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.llm.base import LLMError
from app.prompts.itinerary import DAY_SCHEMA
from app.schemas.itinerary import ItineraryRequest, LLMItinerary, RegenerateDayRequest
from app.services.itinerary_service import ItineraryGenerationError, ItineraryService
from tests.conftest import FakeLLM, make_plan

TRIP = ItineraryRequest(destination="Rome", days=2, budget_eur=500)


def new_day(cost: float = 50) -> str:
    activity: dict[str, Any] = {"activity": "Vatican Museums", "time": "09:00", "cost_eur": cost}
    return json.dumps({"day": 7, "morning": activity, "afternoon": activity, "evening": activity})


def make_request(**overrides: Any) -> RegenerateDayRequest:
    data: dict[str, Any] = {
        "trip": TRIP,
        "plan": LLMItinerary.model_validate(make_plan(days=2, activity_cost=20, night_cost=80)),
        "day": 2,
        "feedback": "more museums",
        **overrides,
    }
    return RegenerateDayRequest(**data)


async def test_replaces_only_the_requested_day_and_recomputes_totals() -> None:
    llm = FakeLLM(new_day(cost=50))

    result = await ItineraryService(llm, max_attempts=2, enable_fallback=True).regenerate_day(
        make_request()
    )

    assert result.days[0].morning.activity == "Museum 1"  # day 1 untouched
    assert result.days[1].day == 2  # the model wrote 7: the number is forced
    assert result.days[1].morning.activity == "Vatican Museums"
    # day 1: 3 x 20, day 2: 3 x 50, 1 night x 80
    assert result.total_estimated_cost_eur == 60 + 150 + 80
    assert llm.schemas == [DAY_SCHEMA]


async def test_prompt_contains_feedback_target_cost_and_other_days() -> None:
    llm = FakeLLM(new_day())

    await ItineraryService(llm, max_attempts=1, enable_fallback=True).regenerate_day(make_request())

    payload = json.loads(llm.calls[0][1]["content"].split("\n", 1)[1])
    assert payload["feedback"] == "more museums"
    assert payload["day_to_rewrite"] == 2
    assert payload["target_cost_eur"] == 500 - 60 - 80  # budget minus everything else
    assert payload["activities_on_other_days"] == {"day_1": ["Museum 1", "Park 1", "Dinner 1"]}


async def test_failure_raises_instead_of_falling_back() -> None:
    llm = FakeLLM(LLMError("down"))

    with pytest.raises(ItineraryGenerationError):
        await ItineraryService(llm, max_attempts=1, enable_fallback=True).regenerate_day(
            make_request()
        )


def test_day_must_exist_in_plan() -> None:
    with pytest.raises(ValidationError, match="day must be between 1 and 2"):
        make_request(day=3)
