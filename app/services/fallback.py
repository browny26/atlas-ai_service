"""Template itinerary used when the LLM is unavailable or keeps producing invalid output.

Responses built from it carry metadata.source = "fallback", so callers can tell the
user the plan is generic, instead of passing it off as an AI result.
"""

import math

from app.schemas.itinerary import Accommodation, Activity, DayPlan, ItineraryRequest, LLMItinerary
from app.services.budget import plan_budget


def build_fallback_plan(request: ItineraryRequest) -> LLMItinerary:
    # Same targets the LLM gets: the template stays under budget too.
    budget = plan_budget(request)
    per_night = budget.accommodation_per_night_target_eur
    # Rounded down: a template that must stay under budget never rounds up.
    per_activity = math.floor(budget.daily_activities_and_meals_target_eur / 3 * 100) / 100
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
