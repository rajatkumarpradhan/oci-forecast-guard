import json

from forecast_guard.cli import main


def test_generate_csv(capsys):
    assert main(["generate"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "day,service,compartment,cost"
    assert len(out) == 1 + 4 * 90


def test_forecast_with_eval(capsys):
    assert main(["forecast", "--days", "90", "--horizon", "14"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["eval"]["beats_baseline"] is True
    assert len(result["forecast"]) == 14


def test_budget_report(capsys):
    assert main(["budget", "--budget", "1500"]) == 0
    risk = json.loads(capsys.readouterr().out)
    assert risk["days_remaining"] == 10
    assert risk["alert_level"] in {"ok", "watch", "warning", "critical"}
