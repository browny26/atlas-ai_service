"""Offline evaluation of the itinerary prompt against the configured (real) LLM.

Answers "how do we know the prompt works?" with numbers, so prompt or model changes
can be compared before deploying them. In CI it fails the build below a quality threshold.

    uv run python -m scripts.evaluate
    uv run python -m scripts.evaluate --min-valid-rate 0.8 --report evaluation.md
"""

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.main import build_llm_client
from app.prompts.itinerary import PROMPT_VERSION
from app.schemas.itinerary import BudgetStatus, ItineraryRequest, ItineraryResponse
from app.services.itinerary_service import ItineraryService

CASES = [
    ItineraryRequest(destination="Rome, Italy", days=3, budget_eur=600, interests=["history"]),
    ItineraryRequest(destination="Tokyo, Japan", days=5, budget_eur=2000, interests=["food"]),
    ItineraryRequest(destination="Lisbon", days=2, budget_eur=300, interests=["views", "music"]),
    ItineraryRequest(destination="New York", days=4, budget_eur=1500, interests=["art"]),
    ItineraryRequest(destination="Reykjavik", days=1, budget_eur=150, interests=["nature"]),
    ItineraryRequest(destination="Marrakech", days=3, budget_eur=450, interests=["markets"]),
]


@dataclass
class CaseResult:
    request: ItineraryRequest
    response: ItineraryResponse | None = None
    error: str | None = None


@dataclass
class Summary:
    total: int
    valid: int
    first_try: int
    within_budget: int
    median_latency_ms: float
    avg_tokens: float

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total if self.total else 0.0


def summarize(results: list[CaseResult]) -> Summary:
    ok = [r.response for r in results if r.response is not None]
    return Summary(
        total=len(results),
        valid=len(ok),
        first_try=sum(r.metadata.attempts == 1 for r in ok),
        within_budget=sum(r.budget_status is BudgetStatus.WITHIN_BUDGET for r in ok),
        median_latency_ms=statistics.median(r.metadata.latency_ms for r in ok) if ok else 0,
        avg_tokens=statistics.mean(r.metadata.usage.total_tokens for r in ok) if ok else 0,
    )


def to_markdown(results: list[CaseResult], summary: Summary, model: str) -> str:
    lines = [
        f"## Prompt evaluation — `{PROMPT_VERSION}` on `{model}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Valid itineraries | {summary.valid}/{summary.total} ({summary.valid_rate:.0%}) |",
        f"| Valid at first attempt | {summary.first_try}/{summary.valid} |",
        f"| Within budget | {summary.within_budget}/{summary.valid} |",
        f"| Median latency | {summary.median_latency_ms:.0f} ms |",
        f"| Average tokens | {summary.avg_tokens:.0f} |",
        "",
        "| Destination | Days | Result | Attempts | Cost / budget (€) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.response is None:
            lines.append(f"| {r.request.destination} | {r.request.days} | ❌ {r.error} | - | - |")
        else:
            m = r.response.metadata
            lines.append(
                f"| {r.request.destination} | {r.request.days} | ✅ {r.response.budget_status} "
                f"| {m.attempts} | {r.response.total_estimated_cost_eur:.0f} / "
                f"{r.request.budget_eur:.0f} |"
            )
    return "\n".join(lines) + "\n"


async def run() -> tuple[list[CaseResult], str]:
    settings = get_settings()
    llm = build_llm_client(settings)
    # No fallback and no cache: we want to measure the model, not hide its failures.
    service = ItineraryService(
        llm, max_attempts=settings.generation_max_attempts, enable_fallback=False
    )
    results: list[CaseResult] = []
    for case in CASES:
        try:
            results.append(CaseResult(case, response=await service.generate(case)))
        except Exception as exc:
            results.append(CaseResult(case, error=str(exc).splitlines()[0].lstrip("- ")[:80]))
    await llm.aclose()
    return results, settings.llm_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-valid-rate", type=float, default=0.0, help="fail below this rate")
    parser.add_argument("--report", type=Path, help="write a Markdown report to this file")
    args = parser.parse_args()

    results, model = asyncio.run(run())
    summary = summarize(results)
    report = to_markdown(results, summary, model)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")

    if summary.valid_rate < args.min_valid_rate:
        print(f"FAILED: valid rate {summary.valid_rate:.0%} < {args.min_valid_rate:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
