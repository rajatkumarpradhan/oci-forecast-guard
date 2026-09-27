"""Synthetic OCI usage-cost data generator.

Produces daily cost series shaped like OCI Usage Reports (service,
compartment, day, cost units) with trend, weekly seasonality, noise and
injectable anomalies. Deterministic per seed so tests and evals
reproduce exactly.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass
class UsagePoint:
    day: int           # days since series start
    service: str
    compartment: str
    cost: float


def _noise(seed: str, day: int) -> float:
    """Deterministic pseudo-noise in [-1, 1) from (seed, day)."""
    digest = hashlib.sha256(f"{seed}:{day}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") / 2**64) * 2 - 1


def daily_cost(day: int, base: float, trend: float, weekly_amp: float,
               noise_amp: float, seed: str, anomaly_day: int | None = None,
               anomaly_mult: float = 1.0) -> float:
    weekly = weekly_amp * math.sin(2 * math.pi * (day % 7) / 7)
    value = base + trend * day + weekly + noise_amp * _noise(seed, day)
    if anomaly_day is not None and day == anomaly_day:
        value *= anomaly_mult
    return max(0.0, value)


def generate_series(days: int, base: float = 40.0, trend: float = 0.15,
                    weekly_amp: float = 6.0, noise_amp: float = 2.0,
                    seed: str = "compute", service: str = "compute",
                    compartment: str = "prod", anomaly_day: int | None = None,
                    anomaly_mult: float = 2.5) -> list[UsagePoint]:
    if days <= 0:
        raise ValueError("days must be positive")
    return [
        UsagePoint(
            day=d,
            service=service,
            compartment=compartment,
            cost=round(daily_cost(d, base, trend, weekly_amp, noise_amp, seed,
                                  anomaly_day, anomaly_mult), 4),
        )
        for d in range(days)
    ]


def generate_fleet() -> dict[str, list[UsagePoint]]:
    """A small realistic fleet of services for demos and fixtures."""
    return {
        "compute": generate_series(90, base=40.0, trend=0.15, seed="compute", service="compute"),
        "object-storage": generate_series(90, base=12.0, trend=0.05, weekly_amp=1.5,
                                          noise_amp=0.8, seed="storage", service="object-storage"),
        "autonomous-db": generate_series(90, base=25.0, trend=0.02, weekly_amp=3.0,
                                         noise_amp=1.2, seed="adb", service="autonomous-db",
                                         anomaly_day=72, anomaly_mult=2.5),
        "genai-inference": generate_series(90, base=8.0, trend=0.4, weekly_amp=4.0,
                                           noise_amp=1.5, seed="genai", service="genai-inference"),
    }
