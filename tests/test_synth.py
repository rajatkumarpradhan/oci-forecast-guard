import pytest

from forecast_guard.synth import generate_fleet, generate_series


def test_deterministic_per_seed():
    a = generate_series(30, seed="x")
    b = generate_series(30, seed="x")
    assert [p.cost for p in a] == [p.cost for p in b]


def test_different_seeds_differ():
    a = generate_series(30, seed="a")
    b = generate_series(30, seed="b")
    assert [p.cost for p in a] != [p.cost for p in b]


def test_costs_non_negative():
    series = generate_series(60, base=1.0, noise_amp=50.0, seed="noisy")
    assert all(p.cost >= 0 for p in series)


def test_anomaly_injected():
    clean = generate_series(30, seed="s")
    spiked = generate_series(30, seed="s", anomaly_day=15, anomaly_mult=3.0)
    assert spiked[15].cost == pytest.approx(clean[15].cost * 3.0, rel=1e-3)
    assert spiked[14].cost == clean[14].cost


def test_fleet_has_services():
    fleet = generate_fleet()
    assert {"compute", "object-storage", "autonomous-db", "genai-inference"} <= set(fleet)
    assert all(len(v) == 90 for v in fleet.values())


def test_invalid_days():
    with pytest.raises(ValueError):
        generate_series(0)
