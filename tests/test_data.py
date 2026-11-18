"""Cache integrity and the synthetic fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.data import load_prices, log_returns, read_cache, simulate_prices

ROOT = Path(__file__).resolve().parents[1]


def test_log_returns_match_a_hand_calculation() -> None:
    close = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    returns = log_returns(close)
    expected = 100.0 * np.log(1.1)
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], expected)
    assert np.isclose(returns.iloc[1], expected)


def test_cache_covers_both_crises_and_known_spx_sessions() -> None:
    prices = read_cache(ROOT / "data" / "prices.csv")
    assert set(prices["ticker"]) == {"SPX", "FEZ", "EEM"}
    assert (prices["high"] + 1e-8 >= prices["low"]).all()
    assert not prices.duplicated(["date", "ticker"]).any()
    for ticker in ("SPX", "FEZ", "EEM"):
        dates = prices.loc[prices["ticker"] == ticker, "date"]
        assert dates.min() <= pd.Timestamp("2007-01-01")
        assert dates.max() >= pd.Timestamp("2020-04-30")
        assert ((dates >= "2008-09-01") & (dates <= "2009-03-31")).sum() > 100
        assert ((dates >= "2020-02-15") & (dates <= "2020-04-30")).sum() > 40
    spx = prices.loc[prices["ticker"] == "SPX"].set_index("date")["close"].sort_index()
    returns = log_returns(spx)
    assert returns.loc["2020-03-16"] < -12.0
    assert returns.loc["2008-10-13"] > 10.0
    assert returns.loc["2008-10-15"] < -8.0


def test_missing_cache_uses_the_garch_fallback(tmp_path: Path) -> None:
    prices, source = load_prices(tmp_path / "absent.csv")
    assert source == "synthetic"
    assert set(prices["ticker"]) == {"SPX", "FEZ", "EEM"}
    assert (prices["close"] > 0).all()


def test_simulated_path_has_crisis_dates_and_finite_returns() -> None:
    prices = simulate_prices(start="2008-01-01", end="2008-12-31", seed=7, tickers=("SPX",))
    returns = log_returns(prices.set_index("date")["close"])
    crisis = returns.loc["2008-09-01":"2008-10-31"]
    calm = returns.loc["2008-01-01":"2008-06-30"]
    assert np.isfinite(returns).all()
    assert crisis.std() > calm.std()
