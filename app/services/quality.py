"""Deterministic quality checks on a structurally valid plan.

Pydantic guarantees the shape of the answer; these checks judge its content: does it
respect the budget, are the times of day coherent, are places repeated? Each problem is
described precisely enough for the model to fix it (it is sent back as feedback), and
plans are ranked so that, if no attempt is perfect, the best one is kept.
"""

from dataclasses import dataclass, field

from app.schemas.itinerary import Activity, DayPlan, LLMItinerary
from app.services.budget import BudgetPlan, estimate_total_cost

# Start-time windows, HH:MM strings compare correctly.
_SLOTS = {
    "morning": ("06:00", "11:59"),
    "afternoon": ("12:00", "17:59"),
    "evening": ("18:00", "23:59"),
}


@dataclass(frozen=True)
class Assessment:
    issues: list[str] = field(default_factory=list)
    over_budget_eur: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.issues

    def rank(self) -> tuple[bool, float, int]:
        """Lower is better: within budget first, then the smallest overspend, then issues."""
        return (self.over_budget_eur > 0, self.over_budget_eur, len(self.issues))

    def is_better_than(self, other: "Assessment") -> bool:
        return self.rank() < other.rank()


# hyphen, en dash, em dash, opening parenthesis, colon
_DETAIL_SEPARATORS = (" - ", " \u2013 ", " \u2014 ", " (", ": ")


def normalize_name(description: str) -> str:
    """The place itself, without the practical details that follow it.

    "Colosseum - book online" and "Colosseum - guided tour" are the same place.
    """
    name = description
    for separator in _DETAIL_SEPARATORS:
        name = name.split(separator, 1)[0]
    return " ".join(name.lower().split())


def _slot_issues(day: DayPlan) -> list[str]:
    issues = []
    for slot, (start, end) in _SLOTS.items():
        activity: Activity = getattr(day, slot)
        if not start <= activity.time <= end:
            issues.append(f"- day {day.day} {slot}: time {activity.time} is outside {start}-{end}")
    if not day.morning.time < day.afternoon.time < day.evening.time:
        issues.append(f"- day {day.day}: times must be in order (morning < afternoon < evening)")
    return issues


def _overspend_issue(
    total: float, maximum: float, target: float, items: list[tuple[str, float]]
) -> str:
    expensive = sorted(items, key=lambda item: item[1], reverse=True)[:4]
    listed = ", ".join(f"{name} (€{cost:.0f})" for name, cost in expensive)
    return (
        f"- budget: the total is €{total:.0f}, above the maximum €{maximum:.0f}. "
        f"Cut at least €{total - target:.0f} to reach the target €{target:.0f}: "
        f"replace expensive items with cheaper or free alternatives. Most expensive: {listed}"
    )


def assess_plan(plan: LLMItinerary, budget: BudgetPlan) -> Assessment:
    issues: list[str] = []

    total = estimate_total_cost(plan, budget.nights)
    over = max(total - budget.max_total_eur, 0.0)
    if over > 0:
        items = [(a.activity, a.cost_eur) for d in plan.days for a in d.activities]
        accommodation = plan.accommodation.cost_per_night_eur * budget.nights
        items.append((f"accommodation, {budget.nights} night(s)", accommodation))
        issues.append(_overspend_issue(total, budget.max_total_eur, budget.target_total_eur, items))

    for day in plan.days:
        issues.extend(_slot_issues(day))

    seen: dict[str, int] = {}
    for day in plan.days:
        for activity in day.activities:
            key = normalize_name(activity.activity)
            if key in seen:
                issues.append(
                    f"- '{activity.activity}' appears more than once (day {seen[key]} and "
                    f"day {day.day}): use a different place"
                )
            else:
                seen[key] = day.day

    return Assessment(issues=issues, over_budget_eur=round(over, 2))


def assess_day(day: DayPlan, max_cost_eur: float, other_activities: set[str]) -> Assessment:
    issues = _slot_issues(day)

    cost = sum(a.cost_eur for a in day.activities)
    over = max(cost - max_cost_eur, 0.0)
    if over > 0:
        issues.append(
            f"- budget: this day costs €{cost:.0f}, above the maximum €{max_cost_eur:.0f} for it: "
            "choose cheaper or free activities"
        )

    names = [normalize_name(a.activity) for a in day.activities]
    if len(set(names)) < len(names):
        issues.append("- the same place appears twice in this day: use different places")
    for activity in day.activities:
        if normalize_name(activity.activity) in other_activities:
            issues.append(f"- '{activity.activity}' is already planned on another day")

    return Assessment(issues=issues, over_budget_eur=round(over, 2))
