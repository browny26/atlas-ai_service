"""Deterministic budget calculations.

The model proposes costs; Python does the arithmetic. LLMs are unreliable at sums,
so totals and budget status are never taken from the model's answer.
"""

from app.schemas.itinerary import BudgetStatus, LLMItinerary


def estimate_total_cost(plan: LLMItinerary, nights: int) -> float:
    activities = sum(activity.cost_eur for day in plan.days for activity in day.activities)
    accommodation = plan.accommodation.cost_per_night_eur * nights
    return round(activities + accommodation, 2)


def budget_status(total_cost: float, budget: float) -> BudgetStatus:
    return BudgetStatus.WITHIN_BUDGET if total_cost <= budget else BudgetStatus.OVER_BUDGET
