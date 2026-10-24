"""Descriptive Stage 9 helpers that need no per-flight test labels.

Test outcomes enter only through aggregates already saved by the single frozen evaluation
(e.g. monthly prevalence, reliability bins). Nothing here fits or changes a model, chooses a
threshold or recalibrates. The prior-shift functions are a consistency check, never applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from flight_delay.contract import FIELD_REGISTRY
from flight_delay.evaluation.calibration import logit

# Raw columns that may be read for label-free analysis: identifiers and published-schedule fields.
LABEL_FREE_RAW_COLUMNS = ("id", "year", "month", "day", "sched_dep_time", "sched_arr_time",
                          "carrier", "origin", "dest", "distance")


def assert_label_free_columns(columns) -> None:
    """Refuse any raw column that is an outcome or derived from one."""
    bad = [c for c in columns
           if c not in FIELD_REGISTRY or FIELD_REGISTRY[c].available_at_prediction != "yes"]
    if bad:
        raise ValueError(f"Columns not label-free / not known at prediction time: {bad}")


def prediction_summary(p, low: float = 0.05, high: float = 0.5) -> dict[str, Any]:
    p = np.asarray(p, dtype="float64")
    qs = np.quantile(p, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"n": int(p.size), "mean": float(p.mean()), "std": float(p.std()),
            "min": float(p.min()), "max": float(p.max()),
            "quantiles": dict(zip(["q01", "q10", "q25", "q50", "q75", "q90", "q99"], map(float, qs))),
            f"share_below_{low}": float((p < low).mean()), f"share_above_{high}": float((p > high).mean())}


def prior_odds_shift(pi_from: float, pi_to: float) -> float:
    """logit(pi_to) − logit(pi_from): the log-odds shift implied by a pure change in base rate."""
    return float(logit(np.array([pi_to]))[0] - logit(np.array([pi_from]))[0])


def prior_shift_adjust(q, pi_from: float, pi_to: float) -> np.ndarray:
    """Probabilities re-expressed for a new base rate under a pure prior (label) shift, i.e.
    p(x|y) unchanged: logit q' = logit q + logit(pi_to) − logit(pi_from) (Saerens et al., 2002).
    Descriptive consistency check only; never applied to the frozen model."""
    z = logit(np.asarray(q, dtype="float64")) + prior_odds_shift(pi_from, pi_to)
    return 1 / (1 + np.exp(-z))


def required_logit_shift(q, target_mean: float) -> float:
    """Constant log-odds shift δ for which mean(σ(logit q + δ)) equals target_mean.
    Describes the size of a level mismatch; it is not fitted to labels and never applied."""
    z = logit(np.asarray(q, dtype="float64"))
    f = lambda d: float(np.mean(1 / (1 + np.exp(-(z + d))))) - target_mean  # noqa: E731
    return float(brentq(f, -20, 20))


def reliability_gaps(bins: list[dict[str, Any]]) -> dict[str, Any]:
    """Observed − predicted per saved reliability bin, and the count-weighted mean gap."""
    gaps = [b["observed_rate"] - b["mean_predicted"] for b in bins]
    n = np.array([b["n"] for b in bins], dtype=float)
    return {"gaps": [float(g) for g in gaps], "weighted_mean_gap": float(np.dot(gaps, n) / n.sum()),
            "all_bins_under_predicted": bool(all(g > 0 for g in gaps))}


def group_mean_predictions(df: pd.DataFrame, group: str, pred: str, period: str,
                           min_n: int = 500) -> pd.DataFrame:
    """Label-free: mean prediction per group and period, keeping groups with >= min_n rows in
    every period."""
    t = df.groupby([group, period], observed=True)[pred].agg(["mean", "size"]).unstack(period)
    keep = (t["size"] >= min_n).all(axis=1)
    return t[keep]
