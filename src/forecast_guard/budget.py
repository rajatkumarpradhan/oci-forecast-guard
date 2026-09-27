"""Budget risk assessment: will this month land over budget?

Uses the forecast trajectory plus residual spread to estimate how much of
the remaining-month spend distribution sits above the budget line.
Offline Monte-Carlo-free: Gaussian approximation from residual sd.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class BudgetRisk:
    spent_to_date: float
    remaining_forecast: float
    projected_total: float
    budget: float
    p_over_budget: float          # probability month total exceeds budget
    alert_level: str              # ok | watch | warning | critical
    days_remaining: int


def assess_budget(spent_to_date: float, forecast: list[float],
                  residual_sd: float, budget: float,
                  alert_thresholds: tuple[float, float, float] = (0.2, 0.5, 0.8)) -> BudgetRisk:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if spent_to_date < 0:
        raise ValueError("spent_to_date cannot be negative")
    remaining = sum(forecast)
    projected = spent_to_date + remaining
    # sd of the remaining-spend total grows with sqrt(days)
    total_sd = residual_sd * math.sqrt(max(len(forecast), 1))
    if total_sd == 0:
        p_over = 1.0 if projected > budget else 0.0
    else:
        p_over = 1 - _norm_cdf((budget - projected) / total_sd)
    t_watch, t_warn, t_crit = alert_thresholds
    if p_over >= t_crit:
        level = "critical"
    elif p_over >= t_warn:
        level = "warning"
    elif p_over >= t_watch:
        level = "watch"
    else:
        level = "ok"
    return BudgetRisk(
        spent_to_date=round(spent_to_date, 2),
        remaining_forecast=round(remaining, 2),
        projected_total=round(projected, 2),
        budget=budget,
        p_over_budget=round(p_over, 3),
        alert_level=level,
        days_remaining=len(forecast),
    )
