"""Itinerary generation use case: prompt -> LLM -> validation -> (repair) -> response.

Two kinds of failure are handled at two levels:
- transport errors (timeouts, 429, 5xx) are retried by the LLM client;
- invalid output (wrong structure, wrong number of days) is retried here, by sending the
  validation errors back to the model and asking it to fix its answer ("self-repair").
If both fail, a clearly-flagged fallback is returned (or an error, if fallback is disabled).
"""

import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.llm.base import ChatMessage, LLMClient, LLMError, Usage
from app.prompts.itinerary import (
    PROMPT_VERSION,
    build_messages,
    build_repair_message,
    nights_for,
)
from app.schemas.itinerary import (
    GenerationMetadata,
    ItineraryRequest,
    ItineraryResponse,
    LLMItinerary,
    Source,
    TokenUsage,
)
from app.services.budget import budget_status, estimate_total_cost
from app.services.fallback import build_fallback_plan

logger = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 10


class InvalidLLMOutputError(Exception):
    """The model's answer does not match the expected schema."""


class ItineraryGenerationError(Exception):
    """No valid itinerary could be produced."""


def parse_plan(content: str, expected_days: int) -> LLMItinerary:
    """Validate the model's raw answer, raising InvalidLLMOutputError with readable errors."""
    try:
        plan = LLMItinerary.model_validate_json(content)
    except ValidationError as exc:
        errors = [
            f"- {'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()[:_MAX_REPORTED_ERRORS]
        ]
        raise InvalidLLMOutputError("\n".join(errors)) from exc

    if len(plan.days) != expected_days:
        raise InvalidLLMOutputError(
            f"- days: expected exactly {expected_days} days, got {len(plan.days)}"
        )
    return plan


@dataclass
class _RunStats:
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    model: str | None = None


class ItineraryService:
    def __init__(self, llm: LLMClient, *, max_attempts: int, enable_fallback: bool) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._enable_fallback = enable_fallback

    async def generate(self, request: ItineraryRequest) -> ItineraryResponse:
        started = time.perf_counter()
        stats = _RunStats()
        source = Source.LLM
        try:
            plan = await self._generate_with_llm(request, stats)
        except (LLMError, InvalidLLMOutputError) as exc:
            if not self._enable_fallback:
                raise ItineraryGenerationError(str(exc)) from exc
            logger.warning("Using fallback itinerary for %r: %s", request.destination, exc)
            plan = build_fallback_plan(request)
            source = Source.FALLBACK

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Itinerary generated destination=%r source=%s attempts=%d tokens=%d latency_ms=%d",
            request.destination,
            source,
            stats.attempts,
            stats.usage.total_tokens,
            latency_ms,
        )
        return self._build_response(request, plan, source, stats, latency_ms)

    async def _generate_with_llm(self, request: ItineraryRequest, stats: _RunStats) -> LLMItinerary:
        messages: list[ChatMessage] = build_messages(request)
        last_error: InvalidLLMOutputError | None = None

        for attempt in range(1, self._max_attempts + 1):
            stats.attempts = attempt
            result = await self._llm.complete_json(messages)
            stats.usage += result.usage
            stats.model = result.model
            try:
                return parse_plan(result.content, expected_days=request.days)
            except InvalidLLMOutputError as exc:
                logger.info("Attempt %d/%d invalid output:\n%s", attempt, self._max_attempts, exc)
                last_error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.content},
                    build_repair_message(str(exc)),
                ]

        assert last_error is not None  # the loop runs at least once (max_attempts >= 1)
        raise last_error

    @staticmethod
    def _build_response(
        request: ItineraryRequest,
        plan: LLMItinerary,
        source: Source,
        stats: _RunStats,
        latency_ms: int,
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
                latency_ms=latency_ms,
                usage=TokenUsage(
                    prompt_tokens=stats.usage.prompt_tokens,
                    completion_tokens=stats.usage.completion_tokens,
                    total_tokens=stats.usage.total_tokens,
                ),
            ),
        )
