"""Loss functions, the Diebold-Mariano test, and VaR coverage tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class HypothesisTest:
    """One likelihood-ratio or Diebold-Mariano result."""

    statistic: float
    p_value: float
    n: int


def qlike(forecast: np.ndarray, realised: np.ndarray) -> float:
    """Mean Patton QLIKE of a variance forecast against a realised-variance proxy."""
    return float(np.mean(qlike_series(forecast, realised)))


def qlike_series(forecast: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Pointwise QLIKE, log(f) + y / f."""
    fitted = np.maximum(np.asarray(forecast, dtype=float), _VARIANCE_FLOOR)
    target = np.asarray(realised, dtype=float)
    return np.log(fitted) + target / fitted


def mse(forecast: np.ndarray, realised: np.ndarray) -> float:
    """Mean squared error of a variance forecast."""
    error = np.asarray(forecast, dtype=float) - np.asarray(realised, dtype=float)
    return float(np.mean(error**2))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> HypothesisTest:
    """Harvey-adjusted DM test of loss_a minus loss_b. Positive means A has higher loss."""
    gap = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    gap = gap[np.isfinite(gap)]
    n = int(gap.size)
    if n < 10:
        return HypothesisTest(float("nan"), float("nan"), n)
    mean_gap = float(gap.mean())
    lags = max(horizon - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    variance = _newey_west_mean_variance(gap, lags)
    if not np.isfinite(variance) or variance <= 0.0:
        return HypothesisTest(float("nan"), float("nan"), n)
    statistic = mean_gap / np.sqrt(variance)
    adjustment = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    adjusted = float(statistic * adjustment)
    p_value = float(2.0 * stats.t.sf(abs(adjusted), df=n - 1))
    return HypothesisTest(adjusted, p_value, n)


def kupiec_pof(hits: np.ndarray, level: float) -> HypothesisTest:
    """Kupiec proportion-of-failures test. Hits are 1 when the loss exceeds VaR."""
    exceptions = np.asarray(hits, dtype=float)
    exceptions = exceptions[np.isfinite(exceptions)]
    n = int(exceptions.size)
    if n == 0:
        return HypothesisTest(float("nan"), float("nan"), 0)
    failures = float(exceptions.sum())
    tail = 1.0 - level
    rate = failures / n
    statistic = -2.0 * (_binomial_ll(failures, n, tail) - _binomial_ll(failures, n, rate))
    statistic = float(max(statistic, 0.0))
    return HypothesisTest(statistic, float(stats.chi2.sf(statistic, 1)), n)


def christoffersen_independence(hits: np.ndarray) -> HypothesisTest:
    """Christoffersen independence test on the hit sequence. Undefined if a row has no visits."""
    sequence = np.asarray(hits, dtype=float)
    sequence = sequence[np.isfinite(sequence)].astype(int)
    if sequence.size < 2:
        return HypothesisTest(float("nan"), float("nan"), int(sequence.size))
    previous, current = sequence[:-1], sequence[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    stay = n00 + n01
    switch = n10 + n11
    if stay == 0 or switch == 0:
        return HypothesisTest(float("nan"), float("nan"), int(sequence.size))
    pooled = (n01 + n11) / (stay + switch)
    null = _binomial_ll(n01 + n11, stay + switch, pooled)
    alternative = _binomial_ll(n01, stay, n01 / stay) + _binomial_ll(n11, switch, n11 / switch)
    statistic = float(max(-2.0 * (null - alternative), 0.0))
    return HypothesisTest(statistic, float(stats.chi2.sf(statistic, 1)), int(sequence.size))


def christoffersen_cc(hits: np.ndarray, level: float) -> HypothesisTest:
    """Conditional coverage: Kupiec plus Christoffersen independence, chi-square with 2 df."""
    unconditional = kupiec_pof(hits, level)
    independence = christoffersen_independence(hits)
    if not np.isfinite(unconditional.statistic) or not np.isfinite(independence.statistic):
        return HypothesisTest(float("nan"), float("nan"), unconditional.n)
    statistic = float(unconditional.statistic + independence.statistic)
    return HypothesisTest(statistic, float(stats.chi2.sf(statistic, 2)), unconditional.n)


def basel_zone(n_exceptions: int, n_obs: int, tail: float = 0.01) -> str:
    """Green, yellow, or red from the Basel cumulative-probability cutoffs at this tail."""
    if n_obs <= 0 or n_exceptions < 0:
        return "undefined"
    cdf = float(stats.binom.cdf(n_exceptions, n_obs, tail))
    if cdf < 0.95:
        return "green"
    if cdf < 0.9999:
        return "yellow"
    return "red"


def normal_var_es(sigma: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean Gaussian loss VaR and ES. Sigma is the standard deviation of the percent return."""
    scale = np.asarray(sigma, dtype=float)
    quantile = float(stats.norm.ppf(level))
    shortfall = float(stats.norm.pdf(quantile) / (1.0 - level))
    return scale * quantile, scale * shortfall


def student_var_es(sigma: np.ndarray, nu: float, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean unit-variance Student-t loss VaR and ES."""
    scale = np.asarray(sigma, dtype=float)
    var_multiplier, es_multiplier = student_multipliers(nu, level)
    return scale * var_multiplier, scale * es_multiplier


def student_multipliers(nu: float, level: float) -> tuple[float, float]:
    """VaR and ES multipliers for a unit-variance Student-t innovation."""
    if nu <= 2.05:
        quantile = float(stats.norm.ppf(level))
        return quantile, float(stats.norm.pdf(quantile) / (1.0 - level))
    unit = float(np.sqrt((nu - 2.0) / nu))
    quantile = float(stats.t.ppf(level, nu))
    density = float(stats.t.pdf(quantile, nu))
    var_multiplier = unit * quantile
    es_multiplier = unit * density * (nu + quantile**2) / ((nu - 1.0) * (1.0 - level))
    return var_multiplier, es_multiplier


def attach_var_es(frame: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    """Add loss VaR and ES columns. Student-t where nu is finite, Gaussian otherwise."""
    out = frame.copy()
    sigma = np.sqrt(np.maximum(out["variance"].to_numpy(dtype=float), _VARIANCE_FLOOR))
    nu = out["nu"].to_numpy(dtype=float)
    normal = ~np.isfinite(nu)
    for level in levels:
        tag = level_tag(level)
        var_col = np.full(sigma.shape, np.nan)
        es_col = np.full(sigma.shape, np.nan)
        if normal.any():
            var_col[normal], es_col[normal] = normal_var_es(sigma[normal], level)
        finite_nu = nu[np.isfinite(nu)]
        for nu_value in np.unique(np.round(finite_nu, 6)):
            mask = np.isfinite(nu) & np.isclose(nu, nu_value, atol=5e-7, rtol=0.0)
            var_col[mask], es_col[mask] = student_var_es(sigma[mask], float(nu_value), level)
        out[f"var_{tag}"] = var_col
        out[f"es_{tag}"] = es_col
    return out


def level_tag(level: float) -> str:
    """Column suffix for a VaR level: 0.99 -> 99 and 0.975 -> 975."""
    percent = level * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return str(int(round(percent)))
    return str(int(round(level * 1000.0)))


def _binomial_ll(successes: float, trials: float, probability: float) -> float:
    failures = trials - successes
    if trials == 0:
        return 0.0
    if probability <= 0.0:
        return 0.0 if successes == 0 else -np.inf
    if probability >= 1.0:
        return 0.0 if failures == 0 else -np.inf
    return failures * np.log(1.0 - probability) + successes * np.log(probability)


def _newey_west_mean_variance(values: np.ndarray, lags: int) -> float:
    centered = values - values.mean()
    n = centered.size
    gamma0 = float(np.dot(centered, centered) / n)
    acc = gamma0
    for lag in range(1, lags + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        acc += 2.0 * (1.0 - lag / (lags + 1)) * gamma
    if acc <= 0.0:
        acc = gamma0
    return acc / n
