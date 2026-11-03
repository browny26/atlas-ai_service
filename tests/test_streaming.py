import json

from app.llm.base import LLMError
from app.schemas.itinerary import ItineraryRequest
from app.services.itinerary_service import ItineraryService, StreamEvent
from app.services.streaming import DayStreamParser, extract_completed_days
from tests.conftest import FakeLLM, make_plan, plan_json

REQUEST = ItineraryRequest(destination="Rome", days=2, budget_eur=500)


def test_extracts_only_closed_day_objects() -> None:
    full = plan_json(days=2)
    cut = full.index('"day": 2') + 5  # in the middle of day 2

    assert len(extract_completed_days(full[:cut])) == 1
    assert len(extract_completed_days(full)) == 2
    assert json.loads(extract_completed_days(full)[1])["day"] == 2


def test_braces_and_keys_inside_strings_are_ignored() -> None:
    plan = make_plan(days=1)
    plan["days"][0]["morning"]["activity"] = 'Visit "the {days}" exhibition: "days": [ {'
    plan = {"tips": ['say "days": [{}]'], **plan}  # "days" also appears in an earlier string

    days = extract_completed_days(json.dumps(plan))

    assert len(days) == 1
    assert json.loads(days[0])["morning"]["activity"].startswith("Visit")


def test_parser_keeps_state_across_chunks() -> None:
    full = plan_json(days=3)
    parser = DayStreamParser()

    days = [day for i in range(0, len(full), 3) for day in parser.feed(full[i : i + 3])]

    assert days == extract_completed_days(full)
    assert [json.loads(d)["day"] for d in days] == [1, 2, 3]


def test_each_day_is_returned_once() -> None:
    full = plan_json(days=2)
    parser = DayStreamParser()
    cut = full.index('"day": 2')

    first = parser.feed(full[:cut])
    rest = parser.feed(full[cut:])

    assert len(first) == 1
    assert len(rest) == 1


def test_no_days_key_yet() -> None:
    assert extract_completed_days('{"accommodation": {"name": "Hotel') == []


async def _collect(service: ItineraryService) -> list[StreamEvent]:
    return [event async for event in service.generate_stream(REQUEST)]


async def test_days_are_streamed_before_the_result() -> None:
    llm = FakeLLM(plan_json(days=2))

    events = await _collect(ItineraryService(llm, max_attempts=3, enable_fallback=True))

    assert [e.event for e in events] == ["day", "day", "result"]
    assert events[0].data["day"] == 1
    assert events[-1].data["metadata"]["source"] == "llm"
    assert events[-1].data["metadata"]["usage"]["total_tokens"] == 150  # streamed usage counted


async def test_invalid_stream_is_repaired_and_result_is_authoritative() -> None:
    llm = FakeLLM(plan_json(days=1), plan_json(days=2))  # first answer has too few days

    events = await _collect(ItineraryService(llm, max_attempts=3, enable_fallback=True))

    assert [e.event for e in events] == ["day", "status", "result"]
    assert len(events[-1].data["days"]) == 2
    assert events[-1].data["metadata"]["attempts"] == 2
    assert events[-1].data["metadata"]["usage"]["total_tokens"] == 300  # stream + repair


async def test_provider_down_streams_fallback_result() -> None:
    llm = FakeLLM(LLMError("down"))

    events = await _collect(ItineraryService(llm, max_attempts=3, enable_fallback=True))

    assert [e.event for e in events] == ["result"]
    assert events[0].data["metadata"]["source"] == "fallback"


async def test_provider_down_without_fallback_streams_error() -> None:
    llm = FakeLLM(LLMError("down"))

    events = await _collect(ItineraryService(llm, max_attempts=3, enable_fallback=False))

    assert [e.event for e in events] == ["error"]


async def test_over_budget_stream_is_adjusted_before_the_result() -> None:
    over = plan_json(days=2, activity_cost=100, night_cost=100)  # 700 > max 475
    llm = FakeLLM(over, plan_json(days=2))  # then 200

    events = await _collect(ItineraryService(llm, max_attempts=3, enable_fallback=True))

    assert [e.event for e in events] == ["day", "day", "status", "result"]
    assert "budget" in events[2].data["message"]
    assert events[-1].data["total_estimated_cost_eur"] == 200
    assert events[-1].data["budget_status"] == "within_budget"
