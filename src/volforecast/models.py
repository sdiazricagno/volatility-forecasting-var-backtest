"""EWMA, GARCH-family, and gradient-boosting variance forecasts."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import GradientBoostingRegressor

from volforecast.config import StudyConfig

GARCH_SPECS: dict[str, dict[str, Any]] = {
    "garch_n": {"vol": "garch", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    "garch_t": {"vol": "garch", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "gjr_n": {"vol": "garch", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    "gjr_t": {"vol": "garch", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "egarch_n": {"vol": "egarch", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    "egarch_t": {"vol": "egarch", "p": 1, "o": 1, "q": 1, "dist": "t"},
}

_GB_VARIANCE_FLOOR = 0.05

GB_FEATURES: tuple[str, ...] = (
    "sq_l1",
    "abs_l1",
    "neg_l1",
    "rv5",
    "rv21",
    "rv63",
    "abs21",
    "down21",
    "ewma",
)


def ewma_variance(returns: pd.Series, lam: float = 0.94, burn_in: int = 21) -> pd.Series:
    """RiskMetrics variance for date t, using returns through t-1 only."""
    data = returns.to_numpy(dtype=float)
    n = data.size
    out = np.full(n, np.nan)
    if n <= burn_in:
        return pd.Series(out, index=returns.index, name="variance")
    variance = float(np.mean(data[:burn_in] ** 2))
    out[burn_in] = variance
    for t in range(burn_in + 1, n):
        variance = lam * variance + (1.0 - lam) * data[t - 1] ** 2
        out[t] = variance
    return pd.Series(out, index=returns.index, name="variance")


def feature_frame(returns: pd.Series) -> pd.DataFrame:
    """Lagged realised-vol features. Row t predicts r_t squared and uses returns through t-1."""
    data = returns.astype(float)
    squared = data.pow(2)
    negative = (data < 0.0).astype(float)
    lagged = squared.shift(1)
    frame = pd.DataFrame(
        {
            "sq_l1": lagged,
            "abs_l1": data.abs().shift(1),
            "neg_l1": negative.shift(1),
            "rv5": lagged.rolling(5).mean(),
            "rv21": lagged.rolling(21).mean(),
            "rv63": lagged.rolling(63).mean(),
            "abs21": data.abs().shift(1).rolling(21).mean(),
            "down21": (squared * negative).shift(1).rolling(21).mean(),
            "ewma": ewma_variance(data),
        },
        index=returns.index,
    )
    frame["target"] = squared
    return frame


def rolling_gb_variance(returns: pd.Series, config: StudyConfig) -> pd.Series:
    """Gradient boosting of next-day squared return on lagged realised-vol features."""
    frame = feature_frame(returns)
    features = frame.loc[:, list(GB_FEATURES)].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=float)
    ready = np.isfinite(features).all(axis=1) & np.isfinite(target)
    n = len(returns)
    out = np.full(n, np.nan)
    positions = np.arange(n)
    origin = config.window
    while origin < n:
        block_end = min(origin + config.refit_every, n)
        train = np.where(ready & (positions >= origin - config.window) & (positions < origin))[0]
        predict = np.where(ready & (positions >= origin) & (positions < block_end))[0]
        if train.size >= config.min_gb_train and predict.size:
            model = GradientBoostingRegressor(
                n_estimators=config.gb_estimators,
                max_depth=config.gb_max_depth,
                learning_rate=config.gb_learning_rate,
                random_state=config.seed,
            )
            model.fit(features[train], target[train])
            out[predict] = model.predict(features[predict])
        origin = block_end
    capped = np.clip(out, _GB_VARIANCE_FLOOR, 1.0e4)
    capped[~np.isfinite(out)] = np.nan
    return pd.Series(capped, index=returns.index, name="variance")


def rolling_garch_variance(
    returns: pd.Series,
    spec: dict[str, Any],
    config: StudyConfig,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """One-step variance and Student-t nu. Parameters are refit every refit_every sessions."""
    data = returns.astype(float)
    n = len(data)
    variance = np.full(n, np.nan)
    nu = np.full(n, np.nan)
    model = arch_model(data, mean="zero", rescale=False, **spec)
    start_values = None
    refits = 0
    failures = 0
    nonconverged = 0
    origin = config.window
    while origin < n:
        block_end = min(origin + config.refit_every, n)
        first = origin - config.window
        result, block = _accepted_block(model, first, origin, block_end, n, start_values)
        warm_failed = result is not None and int(result.convergence_flag) != 0
        if (block is None or warm_failed) and start_values is not None:
            result, block = _accepted_block(model, first, origin, block_end, n, None)
        if result is None or block is None:
            failures += 1
            start_values = None
            origin = block_end
            continue
        if int(result.convergence_flag) != 0:
            nonconverged += 1
        variance[origin:block_end] = np.maximum(block, config.variance_floor)
        if "nu" in result.params.index:
            nu_hat = float(result.params["nu"])
            if nu_hat > 2.05:
                nu[origin:block_end] = nu_hat
        start_values = result.params
        refits += 1
        origin = block_end
    info = {"refits": refits, "failures": failures, "nonconverged": nonconverged}
    index = returns.index
    return pd.Series(variance, index=index, name="variance"), pd.Series(nu, index=index, name="nu"), info


def _accepted_block(
    model: Any,
    first: int,
    origin: int,
    block_end: int,
    n: int,
    start_values: Any,
) -> tuple[Any, np.ndarray] | tuple[None, None]:
    result = _fit_garch(model, first, origin, start_values)
    block = _forecast_block(result, n, origin, block_end)
    if result is None or block is None:
        return None, None
    return result, block


def _forecast_block(result: Any, n: int, origin: int, block_end: int) -> np.ndarray | None:
    if result is None or not _params_ok(result.params):
        return None
    try:
        forecasts = result.forecast(align="target", reindex=True)
    except (ValueError, RuntimeError):
        return None
    values = np.asarray(forecasts.variance.iloc[:, 0], dtype=float)
    if values.shape[0] != n:
        return None
    block = values[origin:block_end]
    if not (np.isfinite(block).all() and np.all(block > 1e-8) and np.all(block < 1.0e4)):
        return None
    return block


def _params_ok(params: pd.Series) -> bool:
    for name, value in params.items():
        number = float(value)
        if not np.isfinite(number):
            return False
        if name == "nu":
            if not 2.0 < number < 500.0:
                return False
        elif abs(number) > 5.0:
            return False
    return True


def _fit_garch(model: Any, first: int, last: int, start_values: Any) -> Any | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(
                first_obs=first,
                last_obs=last,
                disp="off",
                show_warning=False,
                starting_values=start_values,
                options={"maxiter": 400},
            )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return None
