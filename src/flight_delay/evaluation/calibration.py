"""Probability calibration on validation predictions (Stage 8; decision D11).

Calibrators map a fitted model's predicted probability p to a calibrated probability:
* `none`: identity.
* `sigmoid` (Platt scaling): q = σ(a·logit(p) + b), fitted by unpenalised logistic regression.
* `isotonic`: a non-decreasing step function of p, fitted by isotonic regression, clipped to [0, 1].

Method selection follows D11: for each method, fit on one validation month and compute log
loss on the other (and the reverse). Select the lowest mean held-out log loss; within
`tie_tolerance`, prefer the simpler method (order given by config).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from flight_delay.evaluation.metrics import PROB_EPS, brier, logloss

METHODS = ("none", "sigmoid", "isotonic")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), PROB_EPS, 1 - PROB_EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method '{method}'")
        self.method = method
        self.model = None
        self.fit_info_: dict[str, Any] | None = None

    def fit(self, p, y) -> "Calibrator":
        p, y = np.asarray(p, dtype="float64"), np.asarray(y, dtype=int)
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)  # unpenalised
            self.model.fit(logit(p).reshape(-1, 1), y)
            params = {"a": float(self.model.coef_[0, 0]), "b": float(self.model.intercept_[0])}
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            self.model.fit(p, y)
            params = {"n_steps": int(len(np.unique(self.model.predict(p))))}
        else:
            params = {}
        self.fit_info_ = {"method": self.method, "n": int(len(y)), "positives": int(y.sum()), **params}
        return self

    def transform(self, p) -> np.ndarray:
        if self.fit_info_ is None:
            raise RuntimeError("Calibrator must be fitted first")
        p = np.asarray(p, dtype="float64")
        if self.method == "sigmoid":
            q = self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            q = self.model.predict(p)
        else:
            q = p.copy()
        return np.clip(q, 0.0, 1.0)


def cross_fitted(p, y, fold, method: str) -> np.ndarray:
    """Out-of-fold calibrated probabilities: rows in each fold are calibrated by a calibrator
    fitted on the other folds only, so no row's own label influences its calibrated value."""
    p, y, fold = np.asarray(p, dtype="float64"), np.asarray(y, dtype=int), np.asarray(fold)
    out = np.full(len(p), np.nan)
    for f in np.unique(fold):
        held = fold == f
        if held.all():
            raise ValueError("Cross-fitting needs at least two folds")
        out[held] = Calibrator(method).fit(p[~held], y[~held]).transform(p[held])
    return out


def compare_methods(p, y, fold, methods) -> dict[str, dict[str, Any]]:
    """Held-out log loss and Brier per method and fold (fold = month the metric is computed on)."""
    p, y, fold = np.asarray(p, dtype="float64"), np.asarray(y, dtype=int), np.asarray(fold)
    results = {}
    for method in methods:
        oof = cross_fitted(p, y, fold, method)
        per_fold = {}
        for f in np.unique(fold):
            m = fold == f
            per_fold[str(f)] = {"evaluated_on": str(f), "fitted_on": [str(g) for g in np.unique(fold) if g != f],
                                "n": int(m.sum()), "log_loss": logloss(y[m], oof[m]),
                                "brier": brier(y[m], oof[m]),
                                "mean_calibrated_p": float(oof[m].mean()), "prevalence": float(y[m].mean())}
        results[method] = {"per_fold": per_fold,
                           "mean_heldout_log_loss": float(np.mean([v["log_loss"] for v in per_fold.values()])),
                           "mean_heldout_brier": float(np.mean([v["brier"] for v in per_fold.values()])),
                           "pooled_oof_log_loss": logloss(y, oof)}
    return results


def select_method(results: dict[str, dict[str, Any]], order, tol: float) -> dict[str, Any]:
    """Lowest mean held-out log loss; methods within `tol` of the best go to the simplest."""
    best = min(results[m]["mean_heldout_log_loss"] for m in order)
    within = [m for m in order if results[m]["mean_heldout_log_loss"] <= best + tol]
    return {"selected": within[0], "best_mean_heldout_log_loss": best, "within_tolerance": within,
            "tie_break_applied": len(within) > 1}
