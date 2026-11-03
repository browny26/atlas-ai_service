import json

import pytest

from app.llm.base import LLMError, LLMOutputRejectedError
from app.prompts.itinerary import PROMPT_VERSION
from app.schemas.itinerary import BudgetStatus, ItineraryRequest, Source
from app.services.itinerary_service import ItineraryGenerationError, ItineraryService
from tests.conftest import FakeLLM, make_plan, plan_json

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


async def test_over_budget_plan_is_sent_back_with_the_amount_to_cut() -> None:
    over = plan_json(days=2, activity_cost=100, night_cost=100)  # 700 > max 475
    within = plan_json(days=2, activity_cost=40, night_cost=150)  # 390 <= 475
    llm = FakeLLM(over, within)

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.attempts == 2
    assert result.total_estimated_cost_eur == 390
    assert result.budget_status is BudgetStatus.WITHIN_BUDGET
    feedback = llm.calls[1][-1]["content"]
    assert "€700" in feedback and "€475" in feedback  # total and maximum
    assert "Cut at least €275" in feedback  # down to the 85% target (425)


async def test_best_attempt_is_kept_when_none_fits_the_budget() -> None:
    llm = FakeLLM(
        plan_json(days=2, activity_cost=100, night_cost=100),  # 700
        plan_json(days=2, activity_cost=80, night_cost=100),  # 580: closest
        plan_json(days=2, activity_cost=120, night_cost=100),  # 820
    )

    result = await make_service(llm).generate(REQUEST)

    # A real, slightly-over plan is better than the generic fallback.
    assert result.metadata.source is Source.LLM
    assert result.total_estimated_cost_eur == 580
    assert result.budget_status is BudgetStatus.OVER_BUDGET


async def test_incoherent_times_and_repeated_places_are_sent_back() -> None:
    bad = make_plan(days=2)
    bad["days"][0]["evening"]["time"] = "10:00"  # dinner in the morning
    bad["days"][1]["morning"]["activity"] = "Museum 1"  # already on day 1
    llm = FakeLLM(json.dumps(bad), plan_json(days=2))

    result = await make_service(llm).generate(REQUEST)

    feedback = llm.calls[1][-1]["content"]
    assert "day 1 evening: time 10:00 is outside 18:00-23:59" in feedback
    assert "'Museum 1' appears more than once" in feedback
    assert result.metadata.attempts == 2


async def test_fallback_template_stays_under_budget() -> None:
    result = await make_service(FakeLLM(LLMError("down"))).generate(REQUEST)

    assert result.metadata.source is Source.FALLBACK
    assert result.total_estimated_cost_eur <= REQUEST.budget_eur * 0.85 + 1


async def test_fallback_template_stays_under_budget_on_long_cheap_trips() -> None:
    request = ItineraryRequest(destination="Rome", days=14, budget_eur=20)

    result = await make_service(FakeLLM(LLMError("down"))).generate(request)

    assert result.total_estimated_cost_eur <= 20 * 0.85


async def test_provider_failure_after_a_valid_answer_keeps_that_answer() -> None:
    over = plan_json(days=2, activity_cost=100, night_cost=100)  # valid but over budget
    llm = FakeLLM(over, LLMError("rate limited"))

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.source is Source.LLM  # not the generic fallback
    assert result.total_estimated_cost_eur == 700


async def test_same_place_with_different_details_counts_as_a_repetition() -> None:
    plan = make_plan(days=2)
    plan["days"][0]["morning"]["activity"] = "Colosseum - book online"
    plan["days"][1]["morning"]["activity"] = "Colosseum - guided tour"
    llm = FakeLLM(json.dumps(plan), plan_json(days=2))

    await make_service(llm).generate(REQUEST)

    assert "appears more than once" in llm.calls[1][-1]["content"]


async def test_json_rejected_by_the_provider_is_repaired() -> None:
    llm = FakeLLM(LLMOutputRejectedError("json_validate_failed", '{"days": ['), plan_json(days=2))

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.source is Source.LLM
    assert result.metadata.attempts == 2
    # The model sees its rejected answer and is asked to fix it.
    assert llm.calls[1][-2] == {"role": "assistant", "content": '{"days": ['}
    assert "not a single valid JSON object" in llm.calls[1][-1]["content"]


async def test_fallback_when_the_provider_rejects_every_answer() -> None:
    llm = FakeLLM(*[LLMOutputRejectedError("json_validate_failed")] * 3)

    result = await make_service(llm).generate(REQUEST)

    assert result.metadata.source is Source.FALLBACK
    assert result.metadata.attempts == 3
