"""Deterministic budget calculations.

The model proposes costs; Python does the arithmetic. LLMs are unreliable at sums, so
totals and budget status are never taken from the model's answer, and the budget the
model must respect is given to it as concrete euro amounts computed here.
"""

import math
from dataclasses import dataclass

from app.schemas.itinerary import BudgetStatus, ItineraryRequest, LLMItinerary

# Staying under budget is better than going over: aim at 85%, never accept more than 95%.
TARGET_SHARE = 0.85
MAX_SHARE = 0.95
# Typical split of the target between accommodation and the rest (activities, meals).
ACCOMMODATION_SHARE = 0.40


def floor_cents(amount: float) -> float:
    """Budget figures are rounded down: rounding must never let a plan spend more."""
    # round() first removes floating-point noise (600 * 0.85 = 509.99999999999994).
    return math.floor(round(amount * 100, 6)) / 100


def nights_for(days: int) -> int:
    """A 3-day trip has 2 nights; a 1-day trip still counts 1 night."""
    return max(days - 1, 1)


@dataclass(frozen=True)
class BudgetPlan:
    """Budget targets for one traveler, in euros."""

    total_budget_eur: float
    target_total_eur: float
    max_total_eur: float
    nights: int
    accommodation_per_night_target_eur: float
    daily_activities_and_meals_target_eur: float

    def as_prompt_data(self) -> dict[str, float | int]:
        return {
            "target_total_eur": self.target_total_eur,
            "max_total_eur": self.max_total_eur,
            "nights": self.nights,
            "accommodation_per_night_target_eur": self.accommodation_per_night_target_eur,
            "daily_activities_and_meals_target_eur": self.daily_activities_and_meals_target_eur,
        }


def plan_budget(request: ItineraryRequest) -> BudgetPlan:
    nights = nights_for(request.days)
    target = request.budget_eur * TARGET_SHARE
    return BudgetPlan(
        total_budget_eur=request.budget_eur,
        target_total_eur=floor_cents(target),
        max_total_eur=floor_cents(request.budget_eur * MAX_SHARE),
        nights=nights,
        accommodation_per_night_target_eur=floor_cents(target * ACCOMMODATION_SHARE / nights),
        daily_activities_and_meals_target_eur=floor_cents(
            target * (1 - ACCOMMODATION_SHARE) / request.days
        ),
    )


def estimate_total_cost(plan: LLMItinerary, nights: int) -> float:
    activities = sum(activity.cost_eur for day in plan.days for activity in day.activities)
    accommodation = plan.accommodation.cost_per_night_eur * nights
    return round(activities + accommodation, 2)


def budget_status(total_cost: float, budget: float) -> BudgetStatus:
    return BudgetStatus.WITHIN_BUDGET if total_cost <= budget else BudgetStatus.OVER_BUDGET
