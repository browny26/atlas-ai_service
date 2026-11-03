"""Template itinerary used when the LLM is unavailable or keeps producing invalid output.

Responses built from it carry metadata.source = "fallback", so callers can tell the
user the plan is generic, instead of passing it off as an AI result.
"""

from app.prompts.itinerary import nights_for
from app.schemas.itinerary import Accommodation, Activity, DayPlan, ItineraryRequest, LLMItinerary

_ACCOMMODATION_SHARE = 0.4  # share of the budget reserved for accommodation
_SPEND_RATIO = 0.9  # plan to spend 90% of the budget, leaving a margin


def build_fallback_plan(request: ItineraryRequest) -> LLMItinerary:
    budget = request.budget_eur * _SPEND_RATIO
    nights = nights_for(request.days)
    per_night = round(budget * _ACCOMMODATION_SHARE / nights, 2)
    per_activity = round(budget * (1 - _ACCOMMODATION_SHARE) / (request.days * 3), 2)
    place = request.destination

    days = [
        DayPlan(
            day=day,
            morning=Activity(
                activity=f"Visit a main landmark of {place}", time="09:30", cost_eur=per_activity
            ),
            afternoon=Activity(
                activity=f"Walk around the city centre of {place}",
                time="14:00",
                cost_eur=per_activity,
            ),
            evening=Activity(
                activity="Dinner at a local restaurant", time="20:00", cost_eur=per_activity
            ),
        )
        for day in range(1, request.days + 1)
    ]
    return LLMItinerary(
        days=days,
        accommodation=Accommodation(
            name=f"Central hotel in {place}",
            type="hotel",
            details="Choose a well-connected area close to public transport.",
            cost_per_night_eur=per_night,
        ),
        tips=[
            "Book the main attractions online in advance",
            "Download offline maps before you leave",
            "Carry some cash for small purchases",
        ],
    )
