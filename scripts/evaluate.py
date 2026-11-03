"""Offline evaluation of the itinerary prompt against the configured (real) LLM.

Answers "how do we know the prompt works?" with numbers, so prompt or model changes
can be compared before deploying them.

    uv run python -m scripts.evaluate
"""

import asyncio
import statistics

from app.core.config import get_settings
from app.main import build_llm_client
from app.prompts.itinerary import PROMPT_VERSION
from app.schemas.itinerary import BudgetStatus, ItineraryRequest, ItineraryResponse, Source
from app.services.itinerary_service import ItineraryService

CASES = [
    ItineraryRequest(destination="Rome, Italy", days=3, budget_eur=600, interests=["history"]),
    ItineraryRequest(destination="Tokyo, Japan", days=5, budget_eur=2000, interests=["food"]),
    ItineraryRequest(destination="Lisbon", days=2, budget_eur=300, interests=["views", "music"]),
    ItineraryRequest(destination="New York", days=4, budget_eur=1500, interests=["art"]),
    ItineraryRequest(destination="Reykjavik", days=1, budget_eur=150, interests=["nature"]),
]


async def main() -> None:
    settings = get_settings()
    llm = build_llm_client(settings)
    # No fallback: we want to measure the model, not hide its failures.
    service = ItineraryService(
        llm, max_attempts=settings.generation_max_attempts, enable_fallback=False
    )

    results: list[ItineraryResponse] = []
    for case in CASES:
        try:
            result = await service.generate(case)
        except Exception as exc:
            print(f"FAIL  {case.destination:<15} {exc}")
            continue
        results.append(result)
        print(
            f"OK    {case.destination:<15} attempts={result.metadata.attempts} "
            f"cost={result.total_estimated_cost_eur:>7.0f}/{case.budget_eur:<5.0f} "
            f"{result.budget_status} {result.metadata.latency_ms}ms"
        )
    await llm.aclose()

    valid = [r for r in results if r.metadata.source is Source.LLM]
    print(f"\nmodel={settings.llm_model} prompt={PROMPT_VERSION}")
    print(f"valid itineraries:   {len(valid)}/{len(CASES)}")
    if valid:
        first_try = sum(r.metadata.attempts == 1 for r in valid)
        in_budget = sum(r.budget_status is BudgetStatus.WITHIN_BUDGET for r in valid)
        print(f"valid at 1st try:    {first_try}/{len(valid)}")
        print(f"within budget:       {in_budget}/{len(valid)}")
        latency = statistics.median(r.metadata.latency_ms for r in valid)
        tokens = statistics.mean(r.metadata.usage.total_tokens for r in valid)
        print(f"median latency (ms): {latency:.0f}")
        print(f"avg tokens:          {tokens:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
