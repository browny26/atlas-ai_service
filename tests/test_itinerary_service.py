import pytest

from app.llm.base import LLMError
from app.prompts.itinerary import PROMPT_VERSION
from app.schemas.itinerary import BudgetStatus, ItineraryRequest, Source
from app.services.itinerary_service import ItineraryGenerationError, ItineraryService
from tests.conftest import FakeLLM, plan_json

REQUEST = ItineraryRequest(destination="Rome", days=2, budget_eur=500, interests=["food"])


def make_service(llm: FakeLLM, *, max_attempts: int = 3, fallback: bool = True) -> ItineraryService:
    return ItineraryService(llm, max_attempts=max_attempts, enable_fallback=fallback)


async def test_valid_answer_on_first_attempt() -> None:
    llm = FakeLLM(plan_json(days=2, activity_cost=20, night_cost=80))

    result = await make_service(llm).generate(REQUEST)

    # 6 activities x 20 + 1 night x 80 = 200, computed in Python
    assert result.total_estimated_cost_eur == 200
    assert result.budget_status is BudgetStatus.WITHIN_BUDGET
    assert result.nights == 1
    assert result.metadata.source is Source.LLM
    assert result.metadata.attempts == 1
    assert result.metadata.prompt_version == PROMPT_VERSION
    assert result.metadata.usage.total_tokens == 150


async def test_invalid_answer_is_repaired_with_validation_errors() -> None:
    llm = FakeLLM('{"days": "not a list"}', plan_json(days=2))

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.source is Source.LLM
    assert result.metadata.attempts == 2
    assert result.metadata.usage.total_tokens == 300  # usage is summed across attempts
    # The second call contains the bad answer and the errors to fix.
    repair_conversation = llm.calls[1]
    assert repair_conversation[-2] == {"role": "assistant", "content": '{"days": "not a list"}'}
    assert "days" in repair_conversation[-1]["content"]


async def test_wrong_number_of_days_triggers_repair() -> None:
    llm = FakeLLM(plan_json(days=1), plan_json(days=2))

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.attempts == 2
    assert "expected exactly 2 days" in llm.calls[1][-1]["content"]


async def test_fallback_after_all_attempts_fail() -> None:
    llm = FakeLLM("not json", "still not json")

    result = await make_service(llm, max_attempts=2).generate(REQUEST)

    assert result.metadata.source is Source.FALLBACK
    assert result.metadata.attempts == 2
    assert len(result.days) == REQUEST.days
    assert result.budget_status is BudgetStatus.WITHIN_BUDGET


async def test_fallback_when_provider_is_down() -> None:
    llm = FakeLLM(LLMError("connection refused"))

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.source is Source.FALLBACK


async def test_error_when_fallback_disabled() -> None:
    llm = FakeLLM(LLMError("connection refused"))

    with pytest.raises(ItineraryGenerationError):
        await make_service(llm, fallback=False).generate(REQUEST)


async def test_over_budget_is_detected() -> None:
    llm = FakeLLM(plan_json(days=2, activity_cost=100, night_cost=100))

    result = await make_service(llm).generate(REQUEST)  # 600 + 100 > 500

    assert result.total_estimated_cost_eur == 700
    assert result.budget_status is BudgetStatus.OVER_BUDGET
