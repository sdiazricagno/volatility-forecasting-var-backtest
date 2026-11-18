#!/usr/bin/env python3
"""Run the volatility and VaR study and write tables and charts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from volforecast.config import StudyConfig
from volforecast.data import load_prices
from volforecast.plots import save_figures
from volforecast.study import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility forecasts and one-day VaR backtests")
    parser.add_argument(
        "--synthetic", action="store_true", help="Ignore the cache and simulate a GARCH sample"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    prices, source = load_prices(ROOT / "data" / "prices.csv", force_synthetic=args.synthetic)
    result = run_study(prices, StudyConfig(), source=source, verbose=True)
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    result.vol_scores.to_csv(out / "vol_scores.csv", index=False)
    result.dm_tests.to_csv(out / "dm_tests.csv", index=False)
    result.var_tests.to_csv(out / "var_tests.csv", index=False)
    result.crisis.to_csv(out / "crisis.csv", index=False)
    result.tail_gaps.to_csv(out / "tail_gaps.csv", index=False)
    result.ranking.to_csv(out / "limit_ranking.csv", index=False)
    _write_forecasts(result.forecasts, out / "forecasts.csv.gz")
    paths = save_figures(result.forecasts, result.vol_scores, result.var_tests, ROOT / "docs")
    elapsed = time.perf_counter() - started
    meta = dict(result.meta)
    meta["elapsed_seconds"] = round(elapsed, 1)
    meta["charts"] = [str(path.relative_to(ROOT)) for path in paths]
    (out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    _print_summary(result.vol_scores, result.var_tests, result.ranking, result.tail_gaps, meta)


def _write_forecasts(forecasts: pd.DataFrame, path: Path) -> None:
    slim = forecasts.copy()
    slim["date"] = pd.to_datetime(slim["date"]).dt.strftime("%Y-%m-%d")
    numeric = [column for column in slim.columns if column not in {"date", "ticker", "model"}]
    slim[numeric] = slim[numeric].astype(float).round(6)
    slim.to_csv(path, index=False, compression="gzip")


def _print_summary(
    vol_scores: pd.DataFrame,
    var_tests: pd.DataFrame,
    ranking: pd.DataFrame,
    tail_gaps: pd.DataFrame,
    meta: dict,
) -> None:
    print(f"source {meta['source']}  oos {meta['oos_start']} to {meta['oos_end']}")
    print(f"best QLIKE {meta['best_qlike_model']}  preferred 99% limit {meta['preferred_limit_model']}")
    print(
        f"refits {meta['refits']}  skipped blocks {meta['fit_failures']}  "
        f"nonconverged {meta['nonconverged']}  elapsed {meta['elapsed_seconds']}s"
    )
    print("\nEqual-weight QLIKE")
    print(
        vol_scores.groupby("model")["qlike"].mean().reindex(ranking.sort_values("qlike")["model"]).to_string()
    )
    focus = var_tests.loc[
        np.isclose(var_tests["level"], 0.99),
        [
            "ticker",
            "model",
            "hit_rate",
            "kupiec_pvalue",
            "ind_pvalue",
            "cc_pvalue",
            "basel_zone",
            "basel_gfc",
            "basel_covid",
            "basel_latest",
            "shortfall_ratio",
        ],
    ]
    print("\n99% VaR")
    print(focus.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("\nLimit ranking")
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    crisis = tail_gaps.loc[tail_gaps["window"].isin(["gfc_2008", "covid_2020", "full"])]
    print("\nStudent-t versus Gaussian 99% VaR")
    print(crisis.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


if __name__ == "__main__":
    main()
