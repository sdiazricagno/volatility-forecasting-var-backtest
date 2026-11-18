"""Rolling out-of-sample study: variance scores, VaR tests, and the limit ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from volforecast.config import (
    CRISIS_WINDOWS,
    MODEL_ORDER,
    REVIEW_LENGTH,
    REVIEW_ORIGINS,
    STUDENT_PAIRS,
    ZONE_PENALTY,
    StudyConfig,
)
from volforecast.data import log_returns
from volforecast.metrics import (
    attach_var_es,
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    diebold_mariano,
    kupiec_pof,
    level_tag,
    mse,
    qlike,
    qlike_series,
)
from volforecast.models import (
    GARCH_SPECS,
    ewma_variance,
    rolling_garch_variance,
    rolling_gb_variance,
)

_DM_PAIRS: tuple[tuple[str, str], ...] = (
    tuple((model, "ewma") for model in MODEL_ORDER if model != "ewma")
    + STUDENT_PAIRS
    + (("gjr_t", "garch_t"), ("egarch_t", "garch_t"))
)


@dataclass(frozen=True)
class StudyResult:
    """Tables from one offline run."""

    forecasts: pd.DataFrame
    vol_scores: pd.DataFrame
    dm_tests: pd.DataFrame
    var_tests: pd.DataFrame
    crisis: pd.DataFrame
    tail_gaps: pd.DataFrame
    ranking: pd.DataFrame
    meta: dict[str, Any]


def run_study(
    prices: pd.DataFrame,
    config: StudyConfig | None = None,
    source: str = "cache",
    verbose: bool = False,
) -> StudyResult:
    """Fit every model on each ticker and score the paired out-of-sample dates."""
    config = config or StudyConfig()
    forecasts, fit_info = _forecast_panel(prices, config, verbose)
    forecasts, common_start = _paired_sample(forecasts)
    vol_scores = _vol_scores(forecasts)
    dm_tests = _dm_tests(forecasts)
    var_tests = _var_tests(forecasts, config)
    crisis = _crisis_table(forecasts)
    tail_gaps = _tail_gaps(forecasts)
    ranking = rank_limit_models(var_tests, vol_scores, config.kupiec_alpha)
    meta = _meta(prices, forecasts, vol_scores, ranking, fit_info, source, common_start, config)
    return StudyResult(forecasts, vol_scores, dm_tests, var_tests, crisis, tail_gaps, ranking, meta)


def rank_limit_models(
    var_tests: pd.DataFrame,
    vol_scores: pd.DataFrame,
    kupiec_alpha: float = 0.05,
) -> pd.DataFrame:
    """Rank 99% limits by crisis traffic lights, then Kupiec failures, then calibration, then QLIKE."""
    tests = var_tests.loc[np.isclose(var_tests["level"], 0.99)].copy()
    qlike_mean = vol_scores.groupby("model")["qlike"].mean()
    rows: list[dict[str, Any]] = []
    for model, group in tests.groupby("model", sort=False):
        penalty = int(sum(group[column].map(ZONE_PENALTY).sum() for column in ("basel_gfc", "basel_covid")))
        rows.append(
            {
                "model": str(model),
                "zone_penalty": penalty,
                "kupiec_failures": int((group["kupiec_pvalue"] < kupiec_alpha).sum()),
                "hit_mad": float(np.mean(np.abs(group["hit_rate"] - 0.01))),
                "qlike": float(qlike_mean.loc[model]),
            }
        )
    ranked = pd.DataFrame(rows)
    order = {name: position for position, name in enumerate(MODEL_ORDER)}
    ranked["order"] = ranked["model"].map(order)
    ranked = ranked.sort_values(
        ["zone_penalty", "kupiec_failures", "hit_mad", "qlike", "order"],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.drop(columns="order")


def _forecast_panel(
    prices: pd.DataFrame,
    config: StudyConfig,
    verbose: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pieces: list[pd.DataFrame] = []
    refits = 0
    failures = 0
    nonconverged = 0
    for ticker, group in prices.groupby("ticker", sort=True):
        closes = group.sort_values("date").set_index("date")["close"]
        returns = log_returns(closes)
        returns.index.name = "date"
        variances: dict[str, pd.Series] = {
            "ewma": ewma_variance(returns, config.ewma_lambda, config.ewma_burn_in)
        }
        nus: dict[str, pd.Series] = {"ewma": pd.Series(np.nan, index=returns.index)}
        for name, spec in GARCH_SPECS.items():
            if verbose:
                print(f"fitting {ticker} {name}", flush=True)
            variance, nu, info = rolling_garch_variance(returns, spec, config)
            variances[name] = variance
            nus[name] = nu
            refits += info["refits"]
            failures += info["failures"]
            nonconverged += info["nonconverged"]
        if verbose:
            print(f"fitting {ticker} gb_rv", flush=True)
        variances["gb_rv"] = rolling_gb_variance(returns, config)
        nus["gb_rv"] = pd.Series(np.nan, index=returns.index)
        pieces.append(_stack_ticker(str(ticker), returns, variances, nus, config))
    return pd.concat(pieces, ignore_index=True), {
        "refits": refits,
        "failures": failures,
        "nonconverged": nonconverged,
    }


def _stack_ticker(
    ticker: str,
    returns: pd.Series,
    variances: dict[str, pd.Series],
    nus: dict[str, pd.Series],
    config: StudyConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        part = pd.DataFrame(
            {
                "ret": returns,
                "realised": returns.pow(2),
                "variance": variances[model],
                "nu": nus[model],
            }
        )
        part["ticker"] = ticker
        part["model"] = model
        rows.append(part.reset_index())
    stacked = pd.concat(rows, ignore_index=True)
    return attach_var_es(stacked, config.var_levels)


def _paired_sample(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    for _, group in forecasts.groupby("ticker", sort=True):
        wide = group.pivot(index="date", columns="model", values="variance").sort_index()
        complete = np.isfinite(wide.to_numpy(dtype=float)).all(axis=1)
        if not complete.any():
            raise RuntimeError("a ticker has no date on which every model produced a variance")
        starts.append(pd.Timestamp(wide.index[int(np.flatnonzero(complete)[0])]))
    common = max(starts)
    kept: list[pd.DataFrame] = []
    for _, group in forecasts.groupby("ticker", sort=True):
        wide = group.pivot(index="date", columns="model", values="variance").sort_index()
        complete = np.isfinite(wide.to_numpy(dtype=float)).all(axis=1)
        good = wide.index[complete & (wide.index >= common)]
        kept.append(group.loc[group["date"].isin(good)])
    paired = pd.concat(kept, ignore_index=True)
    return paired.sort_values(["ticker", "model", "date"]).reset_index(drop=True), common


def _vol_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ticker, model), group in forecasts.groupby(["ticker", "model"], sort=False):
        forecast = group["variance"].to_numpy(dtype=float)
        realised = group["realised"].to_numpy(dtype=float)
        rows.append(
            {
                "ticker": ticker,
                "model": model,
                "n": int(len(group)),
                "qlike": qlike(forecast, realised),
                "mse": mse(forecast, realised),
                "mean_nu": _mean_nu(group["nu"].to_numpy(dtype=float)),
            }
        )
    scores = pd.DataFrame(rows)
    return _order_models(scores)


def _dm_tests(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, group in forecasts.groupby("ticker", sort=True):
        variance = group.pivot(index="date", columns="model", values="variance").sort_index()
        realised = group.pivot(index="date", columns="model", values="realised").sort_index()
        target = realised["ewma"].to_numpy(dtype=float)
        losses = {
            model: qlike_series(variance[model].to_numpy(dtype=float), target) for model in variance.columns
        }
        for left, right in _DM_PAIRS:
            test = diebold_mariano(losses[left], losses[right])
            rows.append(
                {
                    "ticker": ticker,
                    "model": left,
                    "benchmark": right,
                    "dm_stat": test.statistic,
                    "p_value": test.p_value,
                    "mean_loss_diff": float(np.mean(losses[left] - losses[right])),
                    "n": test.n,
                }
            )
    return pd.DataFrame(rows)


def _var_tests(forecasts: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = forecasts.groupby(["ticker", "model"], sort=False)
    for (ticker, model), group in grouped:
        ordered = group.sort_values("date")
        dates = pd.to_datetime(ordered["date"])
        loss = -ordered["ret"].to_numpy(dtype=float)
        for level in config.var_levels:
            tag = level_tag(level)
            var = ordered[f"var_{tag}"].to_numpy(dtype=float)
            es = ordered[f"es_{tag}"].to_numpy(dtype=float)
            hits = loss > var
            n = int(hits.size)
            exceptions = int(hits.sum())
            kupiec = kupiec_pof(hits.astype(float), level)
            independence = christoffersen_independence(hits.astype(float))
            coverage = christoffersen_cc(hits.astype(float), level)
            tail = 1.0 - level
            latest_zone, latest_exceptions = _latest_zone(hits, tail)
            gfc_zone, gfc_exceptions = _origin_zone(dates, hits, REVIEW_ORIGINS["gfc_2008"], tail)
            covid_zone, covid_exceptions = _origin_zone(dates, hits, REVIEW_ORIGINS["covid_2020"], tail)
            breached = hits & np.isfinite(es)
            if breached.any() and np.isfinite(es[breached]).all() and float(es[breached].mean()) > 0:
                shortfall_ratio = float(loss[breached].mean() / es[breached].mean())
            else:
                shortfall_ratio = float("nan")
            rows.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "level": level,
                    "n": n,
                    "n_exceptions": exceptions,
                    "hit_rate": exceptions / n if n else float("nan"),
                    "kupiec_stat": kupiec.statistic,
                    "kupiec_pvalue": kupiec.p_value,
                    "ind_stat": independence.statistic,
                    "ind_pvalue": independence.p_value,
                    "cc_stat": coverage.statistic,
                    "cc_pvalue": coverage.p_value,
                    "basel_zone": basel_zone(exceptions, n, tail),
                    "basel_latest": latest_zone,
                    "exceptions_latest": latest_exceptions,
                    "basel_gfc": gfc_zone,
                    "exceptions_gfc": gfc_exceptions,
                    "basel_covid": covid_zone,
                    "exceptions_covid": covid_exceptions,
                    "mean_var": float(np.mean(var)),
                    "mean_es": float(np.mean(es)),
                    "es_breach_rate": float(np.mean(loss > es)),
                    "shortfall_ratio": shortfall_ratio,
                }
            )
    return _order_models(pd.DataFrame(rows))


def _crisis_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ticker, model), group in forecasts.groupby(["ticker", "model"], sort=False):
        for name, (start, end) in CRISIS_WINDOWS.items():
            window = group.loc[(group["date"] >= start) & (group["date"] <= end)]
            if len(window) < 15:
                continue
            loss = -window["ret"].to_numpy(dtype=float)
            rows.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "window": name,
                    "start": start,
                    "end": end,
                    "n": int(len(window)),
                    "qlike": qlike(
                        window["variance"].to_numpy(dtype=float), window["realised"].to_numpy(dtype=float)
                    ),
                    "hit_rate_99": float(np.mean(loss > window["var_99"].to_numpy(dtype=float))),
                    "hit_rate_975": float(np.mean(loss > window["var_975"].to_numpy(dtype=float))),
                    "mean_var_99": float(window["var_99"].mean()),
                    "mean_es_99": float(window["es_99"].mean()),
                    "mean_sigma": float(np.sqrt(window["variance"].clip(lower=0)).mean()),
                }
            )
    return _order_models(pd.DataFrame(rows))


def _tail_gaps(forecasts: pd.DataFrame) -> pd.DataFrame:
    windows = {"full": None, **CRISIS_WINDOWS}
    rows: list[dict[str, Any]] = []
    for ticker, group in forecasts.groupby("ticker", sort=True):
        for window_name, bounds in windows.items():
            if bounds is None:
                sample = group
            else:
                sample = group.loc[(group["date"] >= bounds[0]) & (group["date"] <= bounds[1])]
            if sample.empty:
                continue
            for student, normal in STUDENT_PAIRS:
                left = sample.loc[sample["model"] == student]
                right = sample.loc[sample["model"] == normal]
                if len(left) < 15 or len(right) < 15:
                    continue
                loss_t = -left["ret"].to_numpy(dtype=float)
                loss_n = -right["ret"].to_numpy(dtype=float)
                rows.append(
                    {
                        "ticker": ticker,
                        "window": window_name,
                        "student": student,
                        "normal": normal,
                        "n": int(len(left)),
                        "var_ratio_99": float(left["var_99"].mean() / right["var_99"].mean()),
                        "es_ratio_99": float(left["es_99"].mean() / right["es_99"].mean()),
                        "hit_rate_student": float(np.mean(loss_t > left["var_99"].to_numpy(dtype=float))),
                        "hit_rate_normal": float(np.mean(loss_n > right["var_99"].to_numpy(dtype=float))),
                        "mean_nu": float(left["nu"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _latest_zone(hits: np.ndarray, tail: float) -> tuple[str, int]:
    if hits.size < REVIEW_LENGTH:
        return "undefined", 0
    exceptions = int(hits[-REVIEW_LENGTH:].sum())
    return basel_zone(exceptions, REVIEW_LENGTH, tail), exceptions


def _origin_zone(dates: pd.Series, hits: np.ndarray, origin: str, tail: float) -> tuple[str, int]:
    chosen = np.flatnonzero(pd.to_datetime(dates).to_numpy() >= np.datetime64(origin))
    if chosen.size < REVIEW_LENGTH:
        return "undefined", 0
    first = pd.Timestamp(pd.to_datetime(dates).to_numpy()[chosen[0]])
    if first > pd.Timestamp(origin) + pd.Timedelta(days=10):
        return "undefined", 0
    exceptions = int(hits[chosen[:REVIEW_LENGTH]].sum())
    return basel_zone(exceptions, REVIEW_LENGTH, tail), exceptions


def _mean_nu(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _order_models(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model" not in frame.columns:
        return frame
    order = {name: position for position, name in enumerate(MODEL_ORDER)}
    out = frame.copy()
    out["_order"] = out["model"].map(order)
    sort_cols = ["ticker", "_order"] if "ticker" in out.columns else ["_order"]
    if "level" in out.columns:
        sort_cols.append("level")
    if "window" in out.columns:
        sort_cols.append("window")
    return out.sort_values(sort_cols, kind="mergesort").drop(columns="_order").reset_index(drop=True)


def _meta(
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    vol_scores: pd.DataFrame,
    ranking: pd.DataFrame,
    fit_info: dict[str, int],
    source: str,
    common_start: pd.Timestamp,
    config: StudyConfig,
) -> dict[str, Any]:
    counts = {ticker: int((prices["ticker"] == ticker).sum()) for ticker in sorted(prices["ticker"].unique())}
    oos_counts = {str(ticker): int(n) for ticker, n in forecasts.groupby("ticker")["date"].nunique().items()}
    qlike_mean = vol_scores.groupby("model")["qlike"].mean().sort_values()
    best_qlike = str(qlike_mean.index[0])
    preferred = str(ranking.iloc[0]["model"])
    return {
        "source": source,
        "price_rows": counts,
        "oos_start": common_start.strftime("%Y-%m-%d"),
        "oos_end": pd.Timestamp(forecasts["date"].max()).strftime("%Y-%m-%d"),
        "oos_days": oos_counts,
        "window": config.window,
        "refit_every": config.refit_every,
        "ewma_lambda": config.ewma_lambda,
        "refits": fit_info["refits"],
        "fit_failures": fit_info["failures"],
        "nonconverged": fit_info["nonconverged"],
        "best_qlike_model": best_qlike,
        "qlike_mean": {str(model): float(value) for model, value in qlike_mean.items()},
        "preferred_limit_model": preferred,
        "ranking": ranking.to_dict(orient="records"),
    }
