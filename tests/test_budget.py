import pytest

from forecast_guard.budget import assess_budget


def test_clearly_under_budget_is_ok():
    risk = assess_budget(spent_to_date=100.0, forecast=[5.0] * 10,
                         residual_sd=0.5, budget=1000.0)
    assert risk.alert_level == "ok"
    assert risk.p_over_budget < 0.2


def test_clearly_over_budget_is_critical():
    risk = assess_budget(spent_to_date=900.0, forecast=[50.0] * 10,
                         residual_sd=1.0, budget=1000.0)
    assert risk.alert_level == "critical"
    assert risk.p_over_budget >= 0.8


def test_near_line_is_watch_or_warning():
    risk = assess_budget(spent_to_date=600.0, forecast=[40.0] * 10,
                         residual_sd=8.0, budget=1000.0)
    assert risk.alert_level in {"watch", "warning"}
    assert 0.2 <= risk.p_over_budget <= 0.8


def test_zero_sd_deterministic():
    risk = assess_budget(spent_to_date=100.0, forecast=[1.0] * 5,
                         residual_sd=0.0, budget=50.0)
    assert risk.p_over_budget == 1.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        assess_budget(0.0, [1.0], 1.0, budget=0)
    with pytest.raises(ValueError):
        assess_budget(-1.0, [1.0], 1.0, budget=100)
