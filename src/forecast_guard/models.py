"""Forecasting models implemented from first principles.

- seasonal_naive: repeat last week's pattern (baseline every model must beat)
- Holt-Winters additive: level + trend + weekly seasonality

Both are fit on plain lists of floats; the caller owns day alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def seasonal_naive(series: list[float], season: int, horizon: int) -> list[float]:
    if season <= 0 or horizon <= 0:
        raise ValueError("season and horizon must be positive")
    if len(series) < season:
        raise ValueError("series shorter than one season")
    tail = series[-season:]
    return [tail[i % season] for i in range(horizon)]


@dataclass
class HoltWintersFit:
    level: list[float]
    trend: list[float]
    seasonal: list[float]
    alpha: float
    beta: float
    gamma: float
    season: int

    def forecast(self, horizon: int) -> list[float]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        n = len(self.level)
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        out = []
        for h in range(1, horizon + 1):
            seasonal_idx = (n - 1 + h) % self.season
            out.append(last_level + h * last_trend + self.seasonal[seasonal_idx])
        return out

    def residual_sd(self, series: list[float]) -> float:
        residuals = []
        for t in range(self.season, len(series)):
            fitted = self.level[t - 1] + self.trend[t - 1] + self.seasonal[t % self.season]
            residuals.append(series[t] - fitted)
        if len(residuals) < 2:
            return 0.0
        mean = sum(residuals) / len(residuals)
        return math.sqrt(sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1))


def holt_winters(series: list[float], season: int = 7, alpha: float = 0.4,
                 beta: float = 0.05, gamma: float = 0.3) -> HoltWintersFit:
    """Additive Holt-Winters with recursions from the original 1960 papers."""
    if len(series) < 2 * season:
        raise ValueError(f"need at least {2 * season} points for season={season}")
    for name, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not 0 < v < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    # initialise: level = mean of first season, trend = avg per-period slope,
    # seasonal = deviation from level per season position
    level = [sum(series[:season]) / season]
    trend = [(sum(series[season:2 * season]) - sum(series[:season])) / (season * season)]
    seasonal = [series[t] - level[0] for t in range(season)]
    for t in range(1, len(series)):
        s_idx = t % season
        prev_level = level[-1]
        prev_trend = trend[-1]
        level.append(alpha * (series[t] - seasonal[s_idx]) + (1 - alpha) * (prev_level + prev_trend))
        trend.append(beta * (level[-1] - prev_level) + (1 - beta) * prev_trend)
        seasonal[s_idx] = gamma * (series[t] - level[-1]) + (1 - gamma) * seasonal[s_idx]
    return HoltWintersFit(level=level, trend=trend, seasonal=seasonal,
                          alpha=alpha, beta=beta, gamma=gamma, season=season)


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("equal non-empty series required")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def mape(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("equal non-empty series required")
    terms = [abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0]
    if not terms:
        raise ValueError("all actuals are zero")
    return 100.0 * sum(terms) / len(terms)
