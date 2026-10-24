"""Threshold-free probability metrics (Stage 7). Threshold-based metrics belong to Stage 8.

All functions take labels in {0, 1} and predicted probabilities in [0, 1]. Inputs for which
a metric is undefined (empty, single class, out-of-range or non-finite probabilities) raise
`MetricUndefinedError` instead of returning a plausible-looking number.

Average Precision is scikit-learn's step-wise estimator, AP = Σ_n (R_n − R_{n−1}) P_n. It is
not the trapezoidal area under the precision–recall curve, which is not computed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROB_EPS = 1e-15  # clipping for log loss so a single 0/1 prediction cannot give infinity


class MetricUndefinedError(ValueError):
    """Raised when a metric is mathematically undefined for the given inputs."""


def _validate(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y).astype("float64"))
    p = np.asarray(p, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        raise MetricUndefinedError(f"Shape mismatch: labels {y.shape}, probabilities {p.shape}")
    if y.size == 0:
        raise MetricUndefinedError("No samples")
    if np.isnan(y).any() or not np.isin(y, (0.0, 1.0)).all():
        raise MetricUndefinedError("Labels must be 0/1 with no missing values")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise MetricUndefinedError("Probabilities must be finite and within [0, 1]")
    return y.astype(int), p


def _require_both_classes(y: np.ndarray, name: str) -> None:
    if y.min() == y.max():
        raise MetricUndefinedError(f"{name} is undefined when only one class is present")


def average_precision(y, p) -> float:
    y, p = _validate(y, p)
    _require_both_classes(y, "Average Precision")
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    y, p = _validate(y, p)
    _require_both_classes(y, "ROC AUC")
    return float(roc_auc_score(y, p))


def brier(y, p) -> float:
    y, p = _validate(y, p)
    return float(brier_score_loss(y, p))


def logloss(y, p) -> float:
    y, p = _validate(y, p)
    return float(log_loss(y, np.clip(p, PROB_EPS, 1 - PROB_EPS), labels=[0, 1]))


def probability_metrics(y, p) -> dict[str, float]:
    """The threshold-free metric set reported per model and partition."""
    y_arr, _ = _validate(y, p)
    return {"n": int(y_arr.size), "prevalence": float(y_arr.mean()),
            "average_precision": average_precision(y, p), "roc_auc": roc_auc(y, p),
            "brier": brier(y, p), "log_loss": logloss(y, p),
            "mean_predicted_probability": float(np.mean(p))}


def day_block_bootstrap(y, p, days, metric, n_boot: int = 200, seed: int = 0,
                        p_other=None) -> dict[str, float]:
    """95% percentile interval for `metric` by resampling whole flight dates with replacement.

    Flights on the same day share weather and congestion, so they are not independent;
    resampling days keeps that dependence. If `p_other` is given, the interval is for the
    paired difference metric(p) − metric(p_other) on the same resampled days.
    """
    y = np.asarray(pd.Series(y).astype(int))
    p = np.asarray(p, dtype="float64")
    q = None if p_other is None else np.asarray(p_other, dtype="float64")
    day_codes, uniques = pd.factorize(pd.Series(days))
    rows_by_day = [np.flatnonzero(day_codes == k) for k in range(len(uniques))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_by_day[k] for k in rng.integers(0, len(rows_by_day), len(rows_by_day))])
        value = metric(y[idx], p[idx])
        stats.append(value if q is None else value - metric(y[idx], q[idx]))
    point = metric(y, p) if q is None else metric(y, p) - metric(y, q)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_boot": n_boot, "resampling_unit": "flight date"}
