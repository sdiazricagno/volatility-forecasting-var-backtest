"""A short end-to-end run on the synthetic fallback."""

from __future__ import annotations

import numpy as np

from volforecast.config import StudyConfig
from volforecast.data import simulate_prices
from volforecast.study import run_study


def test_mini_study_scores_every_model() -> None:
    prices = simulate_prices(start="2016-01-04", end="2018-06-29", seed=3, tickers=("SPX", "FEZ"))
    config = StudyConfig(window=180, refit_every=120, gb_estimators=12, min_gb_train=60)
    result = run_study(prices, config, source="synthetic", verbose=False)
    assert set(result.vol_scores["model"]) == {
        "ewma",
        "garch_n",
        "garch_t",
        "gjr_n",
        "gjr_t",
        "egarch_n",
        "egarch_t",
        "gb_rv",
    }
    assert np.isfinite(result.vol_scores["qlike"]).all()
    assert (result.vol_scores["mse"] >= 0).all()
    var99 = result.var_tests.loc[np.isclose(result.var_tests["level"], 0.99)]
    assert var99["hit_rate"].between(0.0, 1.0).all()
    assert (var99["mean_es"] >= var99["mean_var"] - 1e-8).all()
    assert result.ranking.iloc[0]["rank"] == 1
    assert result.meta["fit_failures"] <= 4
    assert result.meta["source"] == "synthetic"
