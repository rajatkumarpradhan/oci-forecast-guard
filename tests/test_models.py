import pytest

from forecast_guard.models import holt_winters, mae, mape, seasonal_naive
from forecast_guard.synth import generate_series


@pytest.fixture
def series():
    return [p.cost for p in generate_series(90)]


def test_seasonal_naive_repeats_week(series):
    out = seasonal_naive(series[:70], 7, 7)
    assert out == series[63:70]


def test_holt_winters_beats_naive_on_synthetic(series):
    train, holdout = series[:76], series[76:]
    fc = holt_winters(train).forecast(14)
    naive = seasonal_naive(train, 7, 14)
    assert mae(holdout, fc) < mae(holdout, naive)


def test_holt_winters_tracks_trend():
    # pure trend + weekly season, no noise: MAPE should be tiny
    pts = [p.cost for p in generate_series(84, noise_amp=0.0, seed="t")]
    fc = holt_winters(pts[:70]).forecast(14)
    assert mape(pts[70:], fc) < 5.0


def test_short_series_rejected():
    with pytest.raises(ValueError):
        holt_winters([1.0] * 10, season=7)


def test_mae_mape_validation():
    with pytest.raises(ValueError):
        mae([1], [])
    with pytest.raises(ValueError):
        mape([0.0], [1.0])
