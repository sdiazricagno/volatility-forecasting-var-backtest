"""Forecasts use information through t-1 and match the EWMA recursion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.config import StudyConfig
from volforecast.data import log_returns, read_cache
from volforecast.models import ewma_variance, feature_frame, rolling_garch_variance, rolling_gb_variance

ROOT = Path(__file__).resolve().parents[1]

GARCH_N = {"vol": "garch", "p": 1, "o": 0, "q": 1, "dist": "normal"}


def _returns(n: int = 420, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, size=n)
    return pd.Series(values, index=pd.bdate_range("2014-01-01", periods=n), name="ret")


def test_ewma_matches_a_hand_recursion() -> None:
    returns = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=4))
    forecast = ewma_variance(returns, lam=0.94, burn_in=2)
    assert np.isnan(forecast.iloc[0]) and np.isnan(forecast.iloc[1])
    assert np.isclose(forecast.iloc[2], 0.5 * (1.0 + 4.0))
    expected = 0.94 * 2.5 + 0.06 * 9.0
    assert np.isclose(forecast.iloc[3], expected)


def test_ewma_forecast_ignores_the_same_day_return() -> None:
    base = _returns(80)
    shocked = base.copy()
    shocked.iloc[40:] = shocked.iloc[40:] + 5.0
    left = ewma_variance(base, burn_in=10)
    right = ewma_variance(shocked, burn_in=10)
    assert np.allclose(left.iloc[:41].to_numpy(), right.iloc[:41].to_numpy(), equal_nan=True)
    assert not np.isclose(left.iloc[41], right.iloc[41])


def test_features_are_lagged_one_day() -> None:
    returns = _returns(120, seed=9)
    frame = feature_frame(returns)
    lagged = returns.pow(2).shift(1)
    row = frame.index[80]
    assert np.isclose(frame.loc[row, "sq_l1"], lagged.loc[row])
    assert np.isclose(frame.loc[row, "rv5"], lagged.loc[:row].tail(5).mean())
    changed = returns.copy()
    changed.loc[row] = 25.0
    refit = feature_frame(changed)
    assert np.allclose(
        frame.loc[row, ["sq_l1", "rv5", "rv21", "down21"]].to_numpy(dtype=float),
        refit.loc[row, ["sq_l1", "rv5", "rv21", "down21"]].to_numpy(dtype=float),
    )


def test_garch_and_gb_forecasts_do_not_use_the_same_day_return() -> None:
    config = StudyConfig(window=180, refit_every=180, gb_estimators=15, min_gb_train=80)
    base = _returns(400, seed=11)
    shocked = base.copy()
    shocked.iloc[180:] = shocked.iloc[180:] + 3.0
    garch_base, _, _ = rolling_garch_variance(base, GARCH_N, config)
    garch_shock, _, _ = rolling_garch_variance(shocked, GARCH_N, config)
    assert np.isclose(garch_base.iloc[180], garch_shock.iloc[180])
    assert abs(garch_base.iloc[181] - garch_shock.iloc[181]) > 1e-6
    gb_base = rolling_gb_variance(base, config)
    gb_shock = rolling_gb_variance(shocked, config)
    assert np.isclose(gb_base.iloc[180], gb_shock.iloc[180])
    assert abs(gb_base.iloc[181] - gb_shock.iloc[181]) > 1e-8


def test_egarch_warm_starts_do_not_explode_on_spx() -> None:
    close = read_cache(ROOT / "data" / "prices.csv")
    close = close.loc[close["ticker"] == "SPX"].set_index("date")["close"].sort_index()
    variance, _, info = rolling_garch_variance(
        log_returns(close),
        {"vol": "egarch", "p": 1, "o": 1, "q": 1, "dist": "normal"},
        StudyConfig(),
    )
    fitted = variance.dropna()
    assert info["failures"] == 0
    assert fitted.max() < 1.0e4
    assert fitted.median() < 5.0
