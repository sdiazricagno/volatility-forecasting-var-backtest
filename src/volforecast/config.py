"""Study constants and the default estimation window."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ORDER: tuple[str, ...] = (
    "ewma",
    "garch_n",
    "garch_t",
    "gjr_n",
    "gjr_t",
    "egarch_n",
    "egarch_t",
    "gb_rv",
)

MODEL_LABELS: dict[str, str] = {
    "ewma": "EWMA",
    "garch_n": "GARCH-N",
    "garch_t": "GARCH-t",
    "gjr_n": "GJR-N",
    "gjr_t": "GJR-t",
    "egarch_n": "EGARCH-N",
    "egarch_t": "EGARCH-t",
    "gb_rv": "GB-RV",
}

TICKER_NAMES: dict[str, str] = {
    "SPX": "S&P 500",
    "FEZ": "EURO STOXX 50",
    "EEM": "Emerging markets",
}

CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "calm_2017": ("2017-01-01", "2017-12-31"),
}

REVIEW_ORIGINS: dict[str, str] = {
    "gfc_2008": "2008-09-01",
    "covid_2020": "2020-02-15",
}

REVIEW_LENGTH = 250

ZONE_PENALTY: dict[str, int] = {"green": 0, "yellow": 1, "red": 2, "undefined": 2}

STUDENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("garch_t", "garch_n"),
    ("gjr_t", "gjr_n"),
    ("egarch_t", "egarch_n"),
)


@dataclass(frozen=True)
class StudyConfig:
    """Rolling-window controls. Parameters refresh on refit_every; the variance filter is daily."""

    window: int = 1008
    refit_every: int = 21
    ewma_lambda: float = 0.94
    ewma_burn_in: int = 21
    var_levels: tuple[float, ...] = (0.99, 0.975)
    seed: int = 42
    gb_estimators: int = 80
    gb_max_depth: int = 2
    gb_learning_rate: float = 0.05
    min_gb_train: int = 120
    variance_floor: float = 1e-8
    kupiec_alpha: float = 0.05
