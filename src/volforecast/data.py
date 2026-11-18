"""Load the cached price sample or simulate a GJR-GARCH fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "ticker", "open", "high", "low", "close")

_SIM_PARAMS: dict[str, dict[str, float]] = {
    "SPX": {"omega": 0.020, "alpha": 0.03, "gamma": 0.08, "beta": 0.90, "nu": 8.0, "shock": 1.8},
    "FEZ": {"omega": 0.025, "alpha": 0.04, "gamma": 0.09, "beta": 0.88, "nu": 7.0, "shock": 1.9},
    "EEM": {"omega": 0.040, "alpha": 0.05, "gamma": 0.10, "beta": 0.86, "nu": 6.0, "shock": 2.2},
}

_CRISIS_SPANS = (("2008-09-01", "2009-03-31"), ("2020-02-15", "2020-04-30"))


def read_cache(path: Path) -> pd.DataFrame:
    """Read the long OHLC cache and drop duplicate sessions."""
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"price cache missing columns: {missing}")
    frame = frame.dropna(subset=["date", "ticker", "close"])
    frame = frame.loc[frame["close"] > 0].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame.drop_duplicates(["date", "ticker"], keep="last")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def log_returns(close: pd.Series) -> pd.Series:
    """Percent log returns, indexed by the session being earned."""
    returns = np.log(close.astype(float)).diff() * 100.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = "ret"
    return returns


def simulate_prices(
    start: str = "2004-01-02",
    end: str = "2024-12-31",
    seed: int = 42,
    tickers: tuple[str, ...] = ("SPX", "FEZ", "EEM"),
) -> pd.DataFrame:
    """Simulate split-free prices from a GJR-GARCH(1,1) with Student-t shocks."""
    dates = pd.bdate_range(start, end)
    rows: list[pd.DataFrame] = []
    for offset, ticker in enumerate(tickers):
        params = _SIM_PARAMS.get(ticker, _SIM_PARAMS["SPX"])
        closes = _simulate_gjr_path(dates, params, seed + offset)
        part = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_prices(cache_path: Path, force_synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Use the cache when it is present. Otherwise simulate an offline sample."""
    if not force_synthetic and cache_path.exists():
        return read_cache(cache_path), "cache"
    return simulate_prices(), "synthetic"


def _simulate_gjr_path(dates: pd.DatetimeIndex, params: dict[str, float], seed: int) -> np.ndarray:
    n = len(dates)
    rng = np.random.default_rng(seed)
    nu = params["nu"]
    shock = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    shock = np.clip(shock, -8.0, 8.0)
    scale = np.ones(n)
    for start, end in _CRISIS_SPANS:
        scale[(dates >= start) & (dates <= end)] = params["shock"]
    shock = shock * scale
    omega, alpha, gamma, beta = params["omega"], params["alpha"], params["gamma"], params["beta"]
    variance = np.empty(n)
    returns = np.empty(n)
    variance[0] = omega / max(1.0 - alpha - 0.5 * gamma - beta, 0.05)
    returns[0] = np.sqrt(variance[0]) * shock[0]
    for t in range(1, n):
        leverage = gamma if returns[t - 1] < 0.0 else 0.0
        variance[t] = omega + (alpha + leverage) * returns[t - 1] ** 2 + beta * variance[t - 1]
        variance[t] = min(max(variance[t], 1e-8), 250.0)
        returns[t] = np.sqrt(variance[t]) * shock[t]
    return 100.0 * np.exp(np.cumsum(returns / 100.0))
