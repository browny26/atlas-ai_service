"""Itinerary use cases: generate (whole or streamed) and regenerate a single day.

Flow: prompt -> LLM -> validation -> (repair) -> deterministic calculations -> response.

Two kinds of failure are handled at two levels:
- transport errors (timeouts, 429, 5xx) are retried by the LLM client;
- invalid output (wrong structure, wrong number of days) is retried here, by sending the
  validation errors back to the model and asking it to fix its answer ("self-repair").
If both fail, a clearly-flagged fallback is returned (or an error, if fallback is disabled).
"""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import ChatMessage, JsonSchema, LLMClient, LLMError, Usage
from app.prompts.itinerary import (
    DAY_SCHEMA,
    ITINERARY_SCHEMA,
    PROMPT_VERSION,
    build_messages,
    build_regenerate_day_messages,
    build_repair_message,
    nights_for,
)
from app.schemas.itinerary import (
    DayPlan,
    GenerationMetadata,
    ItineraryRequest,
    ItineraryResponse,
    LLMItinerary,
    RegenerateDayRequest,
    Source,
    TokenUsage,
)
from app.services.budget import budget_status, estimate_total_cost
from app.services.cache import ItineraryCache, cache_key
from app.services.fallback import build_fallback_plan
from app.services.streaming import DayStreamParser

logger = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 10


class InvalidLLMOutputError(Exception):
    """The model's answer does not match the expected schema."""


class ItineraryGenerationError(Exception):
    """No valid itinerary could be produced."""


def _validate[M: BaseModel](model: type[M], content: str) -> M:
    """Validate raw JSON, raising InvalidLLMOutputError with errors the model can act on."""
    try:
        return model.model_validate_json(content)
    except ValidationError as exc:
        errors = [
            f"- {'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()[:_MAX_REPORTED_ERRORS]
        ]
        raise InvalidLLMOutputError("\n".join(errors)) from exc


def parse_plan(content: str, expected_days: int) -> LLMItinerary:
    plan = _validate(LLMItinerary, content)
    if len(plan.days) != expected_days:
        raise InvalidLLMOutputError(
            f"- days: expected exactly {expected_days} days, got {len(plan.days)}"
        )
    return plan


def parse_day(content: str, day_number: int) -> DayPlan:
    # The day number is ours to decide, whatever the model wrote.
    return _validate(DayPlan, content).model_copy(update={"day": day_number})


@dataclass
class _RunStats:
    started: float = field(default_factory=time.perf_counter)
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    model: str | None = None

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


@dataclass(frozen=True)
class StreamEvent:
    """One Server-Sent Event: "day" (a validated day), "status", "result" or "error"."""

    event: str
    data: dict[str, Any]


class ItineraryService:
    def __init__(
        self,
        llm: LLMClient,
        *,
        max_attempts: int,
        enable_fallback: bool,
        cache: ItineraryCache | None = None,
    ) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._enable_fallback = enable_fallback
        self._cache = cache

    # ---------- Use cases ----------

    async def generate(self, request: ItineraryRequest) -> ItineraryResponse:
        if cached := self._from_cache(request):
            return cached

        stats = _RunStats()
        source = Source.LLM
        try:
            plan = await self._complete_validated(
                build_messages(request),
                ITINERARY_SCHEMA,
                lambda content: parse_plan(content, request.days),
                stats,
            )
        except (LLMError, InvalidLLMOutputError) as exc:
            plan, source = self._fallback_or_raise(request, exc)

        return self._finish(request, plan, source, stats)

    async def generate_stream(self, request: ItineraryRequest) -> AsyncIterator[StreamEvent]:
        """Like generate(), but sends each day as soon as the model has written it."""
        if cached := self._from_cache(request):
            for day in cached.days:
                yield StreamEvent("day", day.model_dump())
            yield StreamEvent("result", cached.model_dump(mode="json"))
            return

        stats = _RunStats(attempts=1, model=self._llm.model)
        messages = build_messages(request)
        source = Source.LLM
        try:
            parts: list[str] = []
            parser = DayStreamParser()
            async for chunk in self._llm.stream_json(messages, schema=ITINERARY_SCHEMA):
                if chunk.usage:
                    stats.usage += chunk.usage
                parts.append(chunk.content)
                for raw_day in parser.feed(chunk.content):
                    # An invalid day is skipped: the final validation reports and repairs it.
                    with suppress(InvalidLLMOutputError):
                        yield StreamEvent("day", _validate(DayPlan, raw_day).model_dump())
            buffer = "".join(parts)

            try:
                plan = parse_plan(buffer, request.days)
            except InvalidLLMOutputError as exc:
                # The streamed days may be wrong: the final "result" event is authoritative.
                yield StreamEvent("status", {"message": "Fixing the itinerary format..."})
                plan = await self._complete_validated(
                    [
                        *messages,
                        {"role": "assistant", "content": buffer},
                        build_repair_message(str(exc)),
                    ],
                    ITINERARY_SCHEMA,
                    lambda content: parse_plan(content, request.days),
                    stats,
                    first_attempt=2,
                    last_error=exc,
                )
        except (LLMError, InvalidLLMOutputError) as exc:
            try:
                plan, source = self._fallback_or_raise(request, exc)
            except ItineraryGenerationError:
                yield StreamEvent(
                    "error", {"detail": "The AI provider could not generate a valid itinerary."}
                )
                return

        response = self._finish(request, plan, source, stats)
        yield StreamEvent("result", response.model_dump(mode="json"))

    async def regenerate_day(self, request: RegenerateDayRequest) -> ItineraryResponse:
        stats = _RunStats()
        target = self._day_target_cost(request)
        try:
            new_day = await self._complete_validated(
                build_regenerate_day_messages(request, target),
                DAY_SCHEMA,
                lambda content: parse_day(content, request.day),
                stats,
            )
        except (LLMError, InvalidLLMOutputError) as exc:
            # No fallback here: silently returning the unchanged day would mislead the user.
            raise ItineraryGenerationError(str(exc)) from exc

        days = [new_day if d.day == request.day else d for d in request.plan.days]
        plan = request.plan.model_copy(update={"days": days})
        return self._build_response(request.trip, plan, Source.LLM, stats)

    # ---------- Building blocks ----------

    async def _complete_validated[T](
        self,
        messages: list[ChatMessage],
        schema: JsonSchema,
        parse: Callable[[str], T],
        stats: _RunStats,
        *,
        first_attempt: int = 1,
        last_error: InvalidLLMOutputError | None = None,
    ) -> T:
        """Ask the model until `parse` accepts its answer, feeding back validation errors."""
        for attempt in range(first_attempt, self._max_attempts + 1):
            stats.attempts = attempt
            result = await self._llm.complete_json(messages, schema=schema)
            stats.usage += result.usage
            stats.model = result.model
            try:
                return parse(result.content)
            except InvalidLLMOutputError as exc:
                logger.info("Attempt %d/%d invalid output:\n%s", attempt, self._max_attempts, exc)
                last_error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.content},
                    build_repair_message(str(exc)),
                ]

        assert last_error is not None  # the loop ran at least once, or the caller passed one
        raise last_error

    def _fallback_or_raise(
        self, request: ItineraryRequest, exc: Exception
    ) -> tuple[LLMItinerary, Source]:
        if not self._enable_fallback:
            raise ItineraryGenerationError(str(exc)) from exc
        logger.warning("Using fallback itinerary for %r: %s", request.destination, exc)
        return build_fallback_plan(request), Source.FALLBACK

    def _cache_key(self, request: ItineraryRequest) -> str:
        return cache_key(request, self._llm.model, PROMPT_VERSION)

    def _from_cache(self, request: ItineraryRequest) -> ItineraryResponse | None:
        if self._cache is None:
            return None
        hit = self._cache.get(self._cache_key(request))
        if hit is None:
            return None
        logger.info("Cache hit for %r", request.destination)
        metadata = hit.metadata.model_copy(update={"cached": True, "latency_ms": 0})
        return hit.model_copy(update={"metadata": metadata})

    def _finish(
        self,
        request: ItineraryRequest,
        plan: LLMItinerary,
        source: Source,
        stats: _RunStats,
    ) -> ItineraryResponse:
        response = self._build_response(request, plan, source, stats)
        # Only real LLM answers are cached: a fallback must not hide a recovered provider.
        if self._cache is not None and source is Source.LLM:
            self._cache.set(self._cache_key(request), response)
        logger.info(
            "Itinerary generated destination=%r source=%s attempts=%d tokens=%d latency_ms=%d",
            request.destination,
            source,
            stats.attempts,
            stats.usage.total_tokens,
            response.metadata.latency_ms,
        )
        return response

    @staticmethod
    def _day_target_cost(request: RegenerateDayRequest) -> float:
        """Budget left for the day being rewritten, after everything else is paid for."""
        nights = nights_for(request.trip.days)
        others = sum(
            a.cost_eur for d in request.plan.days if d.day != request.day for a in d.activities
        )
        accommodation = request.plan.accommodation.cost_per_night_eur * nights
        return round(max(request.trip.budget_eur - others - accommodation, 0), 2)

    @staticmethod
    def _build_response(
        request: ItineraryRequest,
        plan: LLMItinerary,
        source: Source,
        stats: _RunStats,
    ) -> ItineraryResponse:
        nights = nights_for(request.days)
        total = estimate_total_cost(plan, nights)
        return ItineraryResponse(
            destination=request.destination,
            total_days=request.days,
            budget_eur=request.budget_eur,
            nights=nights,
            total_estimated_cost_eur=total,
            budget_status=budget_status(total, request.budget_eur),
            days=plan.days,
            accommodation=plan.accommodation,
            tips=plan.tips,
            metadata=GenerationMetadata(
                source=source,
                model=stats.model,
                prompt_version=PROMPT_VERSION,
                attempts=stats.attempts,
                latency_ms=stats.latency_ms,
                usage=TokenUsage(
                    prompt_tokens=stats.usage.prompt_tokens,
                    completion_tokens=stats.usage.completion_tokens,
                    total_tokens=stats.usage.total_tokens,
                ),
            ),
        )
