"""Command line entry points: generate synthetic data, forecast, budget report."""
from __future__ import annotations

import argparse
import csv
import json
import sys

from .budget import assess_budget
from .models import holt_winters, mae, mape, seasonal_naive
from .synth import generate_fleet, generate_series


def _series_from_csv(path: str) -> list[float]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if "cost" not in rows[0]:
        raise ValueError("CSV must have a 'cost' column")
    return [float(r["cost"]) for r in rows]


def cmd_generate(args: argparse.Namespace) -> int:
    fleet = generate_fleet()
    writer = csv.writer(sys.stdout)
    writer.writerow(["day", "service", "compartment", "cost"])
    for service, points in fleet.items():
        for p in points:
            writer.writerow([p.day, p.service, p.compartment, p.cost])
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    if args.csv:
        series = _series_from_csv(args.csv)
    else:
        series = [p.cost for p in generate_series(args.days)]
    if len(series) <= args.horizon + 14:
        train = series
        holdout: list[float] = []
    else:
        train = series[:-args.horizon]
        holdout = series[-args.horizon:]
    fit = holt_winters(train)
    forecast = fit.forecast(args.horizon)
    result = {
        "train_points": len(train),
        "horizon": args.horizon,
        "forecast": [round(v, 2) for v in forecast],
        "residual_sd": round(fit.residual_sd(train), 3),
    }
    if holdout:
        naive = seasonal_naive(train, 7, args.horizon)
        result["eval"] = {
            "holt_winters": {"mae": round(mae(holdout, forecast), 3),
                             "mape": round(mape(holdout, forecast), 2)},
            "seasonal_naive": {"mae": round(mae(holdout, naive), 3),
                               "mape": round(mape(holdout, naive), 2)},
            "beats_baseline": mae(holdout, forecast) < mae(holdout, naive),
        }
    print(json.dumps(result, indent=2))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    if args.csv:
        series = _series_from_csv(args.csv)
    else:
        series = [p.cost for p in generate_series(90)]
    days_in_month = args.days_in_month
    spent = sum(series[:args.elapsed_days])
    train = series[:args.elapsed_days]
    fit = holt_winters(train)
    forecast = fit.forecast(days_in_month - args.elapsed_days)
    risk = assess_budget(spent, forecast, fit.residual_sd(train), args.budget)
    print(json.dumps(risk.__dict__, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast-guard",
                                     description="Offline OCI usage-cost forecasting + budget risk")
    sub = parser.add_subparsers(required=True)
    g = sub.add_parser("generate", help="print a synthetic fleet usage CSV")
    g.set_defaults(func=cmd_generate)
    f = sub.add_parser("forecast", help="forecast a series and self-evaluate vs seasonal naive")
    f.add_argument("--csv")
    f.add_argument("--days", type=int, default=90)
    f.add_argument("--horizon", type=int, default=14)
    f.set_defaults(func=cmd_forecast)
    b = sub.add_parser("budget", help="assess month-end budget risk")
    b.add_argument("--csv")
    b.add_argument("--budget", type=float, required=True)
    b.add_argument("--elapsed-days", type=int, default=20)
    b.add_argument("--days-in-month", type=int, default=30)
    b.set_defaults(func=cmd_budget)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
