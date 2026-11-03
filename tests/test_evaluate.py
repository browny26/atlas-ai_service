from app.schemas.itinerary import ItineraryRequest
from app.services.itinerary_service import ItineraryService
from scripts.evaluate import CaseResult, summarize, to_markdown
from tests.conftest import FakeLLM, plan_json


async def test_summary_and_report() -> None:
    request = ItineraryRequest(destination="Rome", days=2, budget_eur=500)
    response = await ItineraryService(
        FakeLLM(plan_json(days=2)), max_attempts=1, enable_fallback=False
    ).generate(request)
    results = [
        CaseResult(request, response=response),
        CaseResult(request.model_copy(update={"destination": "Oslo"}), error="invalid output"),
    ]

    summary = summarize(results)
    report = to_markdown(results, summary, "fake-model")

    assert summary.valid_rate == 0.5
    assert summary.first_try == 1
    assert "| Valid itineraries | 1/2 (50%) |" in report
    assert "❌ invalid output" in report
    assert "| Average budget used | 40% (target 85%) |" in report  # 200 of 500
