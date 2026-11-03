from app.schemas.itinerary import ItineraryRequest, Source
from app.services.cache import TTLCache, cache_key
from app.services.itinerary_service import ItineraryService
from tests.conftest import FakeLLM, plan_json

REQUEST = ItineraryRequest(destination="Rome", days=2, budget_eur=500)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache = TTLCache(ttl_seconds=60, max_entries=10, clock=clock)
    response = await ItineraryService(
        FakeLLM(plan_json(days=2)), max_attempts=1, enable_fallback=True
    ).generate(REQUEST)

    cache.set("k", response)
    clock.now = 59
    assert cache.get("k") is response
    clock.now = 60
    assert cache.get("k") is None


async def test_least_recently_used_entry_is_evicted() -> None:
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    response = await ItineraryService(
        FakeLLM(plan_json(days=2)), max_attempts=1, enable_fallback=True
    ).generate(REQUEST)

    cache.set("a", response)
    cache.set("b", response)
    cache.get("a")  # "a" becomes the most recently used
    cache.set("c", response)

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert len(cache) == 2


def test_key_depends_on_request_model_and_prompt_version() -> None:
    base = cache_key(REQUEST, "model-a", "v1")

    assert base == cache_key(REQUEST, "model-a", "v1")
    assert base != cache_key(REQUEST, "model-b", "v1")
    assert base != cache_key(REQUEST, "model-a", "v2")
    assert base != cache_key(REQUEST.model_copy(update={"days": 3}), "model-a", "v1")


async def test_second_identical_request_does_not_call_the_llm() -> None:
    llm = FakeLLM(plan_json(days=2))  # a single scripted reply
    service = ItineraryService(
        llm, max_attempts=1, enable_fallback=True, cache=TTLCache(ttl_seconds=60, max_entries=10)
    )

    first = await service.generate(REQUEST)
    second = await service.generate(REQUEST)

    assert len(llm.calls) == 1
    assert first.metadata.cached is False
    assert second.metadata.cached is True
    assert second.days == first.days


async def test_fallback_answers_are_not_cached() -> None:
    llm = FakeLLM("not json", plan_json(days=2))
    service = ItineraryService(
        llm, max_attempts=1, enable_fallback=True, cache=TTLCache(ttl_seconds=60, max_entries=10)
    )

    first = await service.generate(REQUEST)
    second = await service.generate(REQUEST)

    assert first.metadata.source is Source.FALLBACK
    assert second.metadata.source is Source.LLM  # the provider was asked again
