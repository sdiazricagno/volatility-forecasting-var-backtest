"""Closed-form checks for the loss functions and the VaR backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from volforecast.metrics import (
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    diebold_mariano,
    kupiec_pof,
    mse,
    normal_var_es,
    qlike,
    student_var_es,
)
from volforecast.study import rank_limit_models


def test_qlike_and_mse_on_a_perfect_forecast() -> None:
    forecast = np.array([1.0, 2.0, 4.0])
    assert np.isclose(qlike(forecast, forecast), np.mean(np.log(forecast) + 1.0))
    assert np.isclose(mse(forecast, forecast), 0.0)
    assert mse(forecast, np.zeros(3)) == np.mean(forecast**2)


def test_kupiec_accepts_the_nominal_rate_and_rejects_a_large_miss() -> None:
    exact = np.zeros(1000)
    exact[:10] = 1.0
    passed = kupiec_pof(exact, 0.99)
    assert passed.statistic < 1e-8
    assert passed.p_value > 0.99
    crowded = np.zeros(1000)
    crowded[:40] = 1.0
    failed = kupiec_pof(crowded, 0.99)
    assert failed.p_value < 0.01


def test_christoffersen_rejects_clustered_hits_and_accepts_scattered_ones() -> None:
    clustered = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), 40)
    rejected = christoffersen_independence(clustered.astype(float))
    assert rejected.p_value < 1e-6
    scattered = np.zeros(4000)
    scattered[::100] = 1.0
    accepted = christoffersen_independence(scattered)
    assert accepted.p_value > 0.10
    joint = christoffersen_cc(clustered.astype(float), 0.99)
    assert joint.p_value < 0.01


def test_basel_zones_match_the_250_day_99_percent_table() -> None:
    assert basel_zone(4, 250, 0.01) == "green"
    assert basel_zone(5, 250, 0.01) == "yellow"
    assert basel_zone(9, 250, 0.01) == "yellow"
    assert basel_zone(10, 250, 0.01) == "red"
    assert basel_zone(0, 0, 0.01) == "undefined"


def test_gaussian_and_student_var_es() -> None:
    sigma = np.array([2.0])
    var, es = normal_var_es(sigma, 0.99)
    quantile = stats.norm.ppf(0.99)
    assert np.isclose(var[0], 2.0 * quantile)
    assert np.isclose(es[0], 2.0 * stats.norm.pdf(quantile) / 0.01)
    assert es[0] > var[0]
    student_var, student_es = student_var_es(np.array([1.0]), 8.0, 0.99)
    normal_var, normal_es = normal_var_es(np.array([1.0]), 0.99)
    assert student_var[0] > normal_var[0]
    assert student_es[0] > student_var[0]
    quadrature, _ = integrate.quad(
        lambda x: x * stats.t.pdf(x, 8.0), stats.t.ppf(0.99, 8.0), np.inf, limit=400
    )
    scale = np.sqrt((8.0 - 2.0) / 8.0)
    assert np.isclose(student_es[0], scale * quadrature / 0.01)


def test_diebold_mariano_sign() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=600)
    worse = baseline + 0.4
    test = diebold_mariano(worse, baseline)
    assert test.statistic > 0.0
    assert test.p_value < 0.01
    empty = diebold_mariano(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(empty.statistic)


def test_limit_ranking_prefers_a_green_calibrated_model() -> None:

    models = ["ewma", "garch_t"]
    var_rows = []
    score_rows = []
    for model, zone, hit, qlike_value in (
        ("ewma", "red", 0.02, 1.5),
        ("garch_t", "green", 0.01, 1.2),
    ):
        for ticker in ("SPX", "FEZ"):
            var_rows.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "level": 0.99,
                    "hit_rate": hit,
                    "kupiec_pvalue": 0.50,
                    "basel_gfc": zone,
                    "basel_covid": zone,
                }
            )
            score_rows.append({"ticker": ticker, "model": model, "qlike": qlike_value})
    ranking = rank_limit_models(pd.DataFrame(var_rows), pd.DataFrame(score_rows))
    assert list(ranking["model"]) == models[::-1] or ranking.iloc[0]["model"] == "garch_t"
    assert ranking.iloc[0]["model"] == "garch_t"
    assert ranking.iloc[0]["zone_penalty"] < ranking.iloc[1]["zone_penalty"]
