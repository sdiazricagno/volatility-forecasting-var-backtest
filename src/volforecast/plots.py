"""Charts for the out-of-sample volatility and VaR comparison."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volforecast.config import MODEL_LABELS, MODEL_ORDER, TICKER_NAMES

_TICKER_COLORS = {"SPX": "#0072B2", "FEZ": "#D55E00", "EEM": "#009E73"}
_MODEL_COLORS = {
    "ewma": "#009E73",
    "garch_n": "#0072B2",
    "garch_t": "#882255",
    "gjr_n": "#E69F00",
    "gjr_t": "#D55E00",
    "egarch_n": "#CC79A7",
    "egarch_t": "#661100",
    "gb_rv": "#333333",
}
_PANELS = (
    ("2008–2009", "2008-07-01", "2009-06-30", "2008-09-15"),
    ("2020", "2020-01-15", "2020-06-30", "2020-02-20"),
)
_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 10,
}


def save_figures(
    forecasts: pd.DataFrame, vol_scores: pd.DataFrame, var_tests: pd.DataFrame, docs: Path
) -> list[Path]:
    """Write the four study charts and return their paths."""
    docs.mkdir(parents=True, exist_ok=True)
    paths = [
        _volatility_chart(forecasts, docs / "spx_crisis_volatility.png"),
        _qlike_chart(vol_scores, docs / "qlike_versus_ewma.png"),
        _var_chart(forecasts, docs / "spx_var_limits.png"),
        _hit_chart(var_tests, docs / "var99_hit_rates.png"),
    ]
    plt.close("all")
    return paths


def _volatility_chart(forecasts: pd.DataFrame, path: Path) -> Path:
    spx = forecasts.loc[forecasts["ticker"] == "SPX"].copy()
    spx["date"] = pd.to_datetime(spx["date"])
    realised = _trailing_vol(spx.loc[spx["model"] == "ewma", ["date", "ret"]])
    shown = ("ewma", "garch_n", "gjr_t")
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharey=True)
        for axis, (title, start, end, marker) in zip(axes, _PANELS, strict=True):
            window = realised.loc[start:end]
            axis.plot(
                window.index, window.to_numpy(), color="#8a8a8a", lw=1.3, label="Trailing 21-day realised"
            )
            for model in shown:
                series = _annualized_forecast(spx.loc[spx["model"] == model, ["date", "variance"]])
                series = series.loc[start:end]
                axis.plot(
                    series.index,
                    series.to_numpy(),
                    color=_MODEL_COLORS[model],
                    lw=1.4,
                    label=MODEL_LABELS[model],
                )
            axis.axvline(pd.Timestamp(marker), color="#999999", ls="--", lw=0.8)
            axis.set_title(title)
            axis.set_ylabel("Annualized volatility (%)")
        axes[0].legend(frameon=False, ncol=4, loc="upper right")
        fig.suptitle("S&P 500 one-day volatility forecasts", y=0.98)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    return path


def _qlike_chart(vol_scores: pd.DataFrame, path: Path) -> Path:
    scores = vol_scores.copy()
    ewma = scores.loc[scores["model"] == "ewma", ["ticker", "qlike"]].rename(columns={"qlike": "ewma"})
    scores = scores.loc[scores["model"] != "ewma"].merge(ewma, on="ticker")
    scores["gap"] = scores["qlike"] - scores["ewma"]
    models = [model for model in MODEL_ORDER if model != "ewma"]
    tickers = [ticker for ticker in ("SPX", "FEZ", "EEM") if ticker in set(scores["ticker"])]
    x = np.arange(len(models))
    width = 0.24
    with plt.rc_context(_STYLE):
        fig, axis = plt.subplots(figsize=(11.0, 5.2))
        for i, ticker in enumerate(tickers):
            part = scores.loc[scores["ticker"] == ticker].set_index("model").reindex(models)
            axis.bar(
                x + (i - (len(tickers) - 1) / 2) * width,
                part["gap"].to_numpy(),
                width=width,
                color=_TICKER_COLORS[ticker],
                label=TICKER_NAMES.get(ticker, ticker),
            )
        axis.axhline(0.0, color="#333333", lw=0.8)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=0)
        axis.set_ylabel("QLIKE minus EWMA")
        axis.set_title("Variance-forecast loss relative to RiskMetrics EWMA")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    return path


def _var_chart(forecasts: pd.DataFrame, path: Path) -> Path:
    spx = forecasts.loc[forecasts["ticker"] == "SPX"].copy()
    spx["date"] = pd.to_datetime(spx["date"])
    returns = (
        spx.loc[spx["model"] == "ewma", ["date", "ret"]].drop_duplicates("date").set_index("date")["ret"]
    )
    shown = ("garch_n", "garch_t")
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharey=True)
        for axis, (title, start, end, marker) in zip(axes, _PANELS, strict=True):
            window = returns.loc[start:end]
            axis.plot(window.index, window.to_numpy(), color="#4d4d4d", lw=0.9, label="Log return")
            for model in shown:
                limit = spx.loc[spx["model"] == model, ["date", "var_99"]].set_index("date")["var_99"]
                limit = -limit.loc[start:end]
                axis.plot(
                    limit.index,
                    limit.to_numpy(),
                    color=_MODEL_COLORS[model],
                    lw=1.4,
                    label=f"{MODEL_LABELS[model]} 99% limit",
                )
            axis.axvline(pd.Timestamp(marker), color="#999999", ls="--", lw=0.8)
            axis.set_title(title)
            axis.set_ylabel("Percent")
        axes[0].legend(frameon=False, ncol=3, loc="lower left")
        fig.suptitle("S&P 500 returns and the one-day 99% loss limit", y=0.98)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    return path


def _hit_chart(var_tests: pd.DataFrame, path: Path) -> Path:
    tests = var_tests.loc[np.isclose(var_tests["level"], 0.99)].copy()
    models = [model for model in MODEL_ORDER if model in set(tests["model"])]
    tickers = [ticker for ticker in ("SPX", "FEZ", "EEM") if ticker in set(tests["ticker"])]
    x = np.arange(len(models))
    width = 0.24
    with plt.rc_context(_STYLE):
        fig, axis = plt.subplots(figsize=(11.0, 5.2))
        for i, ticker in enumerate(tickers):
            part = tests.loc[tests["ticker"] == ticker].set_index("model").reindex(models)
            axis.bar(
                x + (i - (len(tickers) - 1) / 2) * width,
                part["hit_rate"].to_numpy() * 100.0,
                width=width,
                color=_TICKER_COLORS[ticker],
                label=TICKER_NAMES.get(ticker, ticker),
            )
        axis.axhline(1.0, color="#333333", lw=0.9, ls="--", label="Nominal 1%")
        axis.set_xticks(x, [MODEL_LABELS[model] for model in models])
        axis.set_ylabel("Exception rate (%)")
        axis.set_title("One-day 99% VaR exceptions over the full out-of-sample window")
        axis.legend(frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    return path


def _trailing_vol(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.drop_duplicates("date").set_index("date").sort_index()
    trail = ordered["ret"].pow(2).rolling(21).mean().shift(1)
    return np.sqrt(trail.clip(lower=0.0)) * np.sqrt(252.0)


def _annualized_forecast(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.drop_duplicates("date").set_index("date").sort_index()
    return np.sqrt(ordered["variance"].clip(lower=0.0)) * np.sqrt(252.0)
