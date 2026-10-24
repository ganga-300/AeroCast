"""Pre-registered final-evaluation metrics (frozen proposal, D32/D34).

`evaluate_predictions` computes exactly the registered metric set from fixed predictions and
labels. It fits nothing used for prediction, selects no threshold and changes no decision. The
only estimation is the descriptive calibration intercept/slope (a logistic fit of outcomes on
logit(q)), which is reported and never applied.

Definitions (also stated in the proposal):
* Calibration-in-the-large: mean(q) − mean(y). Positive means over-prediction.
* Calibration intercept and slope: unpenalised logistic regression of y on logit(clip(q, 1e-15,
  1 − 1e-15)). Slope is the coefficient and intercept the constant; perfect calibration is
  (0, 1). Undefined for constant predictions, which are reported as "undefined (constant
  prediction)".
* Reliability bins: up to 10 quantile bins of q. Bins with duplicate edges are merged (step-shaped
  isotonic outputs can give fewer bins). Each bin reports n, mean q and observed rate.
* Bootstrap: 95% percentile interval from 200 resamples of whole flight dates with replacement,
  seed from config. Resamples where a metric is undefined are excluded and counted (n_undefined).
* Paired differences: metric(primary) − metric(other) on the same resampled days. AP and ROC AUC
  differences > 0 favour the primary model; Brier and log-loss differences < 0 favour it.
* Constant baselines: computed with the same code. ROC AUC of a constant prediction is 0.5 by
  construction (degenerate interval); Average Precision equals the prevalence.
* Threshold-dependent metrics: not computed (D32).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flight_delay.evaluation.calibration import logit
from flight_delay.evaluation.metrics import (average_precision, brier, day_block_bootstrap, logloss,
                                             probability_metrics, roc_auc)

MAX_RELIABILITY_BINS = 10
INTERVAL_METRICS = {"average_precision": average_precision, "roc_auc": roc_auc,
                    "brier": brier, "log_loss": logloss}
THRESHOLD_METRICS_STATUS = {
    "status": "NOT APPLICABLE (D32)",
    "statement": "No operating threshold — the 70% precision target was infeasible on validation.",
    "not_computed": ["recall at the 70% minimum-precision threshold", "precision", "recall", "F1",
                     "confusion matrix at an operating threshold"],
}
SIGN_CONVENTION = ("difference = metric(primary) − metric(other); AP/ROC AUC > 0 favour primary; "
                   "Brier/log loss < 0 favour primary")


def calibration_in_the_large(y, q) -> float:
    return float(np.mean(q) - np.mean(np.asarray(y, dtype=float)))


def calibration_intercept_slope(y, q) -> dict[str, Any]:
    q = np.asarray(q, dtype="float64")
    if np.ptp(q) == 0:
        return {"status": "undefined (constant prediction)"}
    fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    fit.fit(logit(q).reshape(-1, 1), np.asarray(y, dtype=int))
    return {"intercept": float(fit.intercept_[0]), "slope": float(fit.coef_[0, 0])}


def reliability_bins(y, q, max_bins: int = MAX_RELIABILITY_BINS) -> list[dict[str, Any]]:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "q": np.asarray(q, dtype=float)})
    if df["q"].nunique() == 1:
        return [{"n": int(len(df)), "mean_predicted": float(df["q"].iloc[0]),
                 "observed_rate": float(df["y"].mean())}]
    df["bin"] = pd.qcut(df["q"], max_bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return [{"n": int(n), "mean_predicted": float(mq), "observed_rate": float(my)}
            for n, mq, my in zip(g.size(), g["q"].mean(), g["y"].mean())]


def model_block(y, q, days, n_boot: int, seed: int, with_intervals: bool) -> dict[str, Any]:
    block = {"metrics": probability_metrics(y, q),
             "calibration_in_the_large": calibration_in_the_large(y, q),
             "calibration_intercept_slope": calibration_intercept_slope(y, q),
             "reliability": reliability_bins(y, q)}
    if with_intervals:
        block["intervals_95"] = {m: day_block_bootstrap(y, q, days, f, n_boot, seed)
                                 for m, f in INTERVAL_METRICS.items()}
    return block


def evaluate_predictions(y, predictions: dict[str, np.ndarray], days, months,
                         primary: str, with_intervals: list[str], secondary: list[str],
                         paired_against: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    """The full registered metric set for fixed predictions.

    predictions: name -> probabilities, aligned with y/days/months.
    with_intervals: models that get bootstrap intervals (primary, comparison, references).
    secondary: descriptive models (raw probabilities), point estimates only.
    paired_against: models compared with `primary` by paired day-block bootstrap.
    """
    y = np.asarray(y, dtype=int)
    days, months = np.asarray(days), np.asarray(months)
    out: dict[str, Any] = {
        "partition_summary": {"labelled_rows": int(len(y)), "positives": int(y.sum()),
                              "prevalence": float(y.mean())},
        "models": {}, "secondary_descriptive": {}, "paired_differences": {},
        "by_month": {}, "threshold_dependent_metrics": THRESHOLD_METRICS_STATUS,
        "bootstrap": {"resamples": n_boot, "seed": seed, "unit": "flight date",
                      "interval": "95% percentile", "undefined_resamples": "excluded and counted"},
        "sign_convention": SIGN_CONVENTION,
    }
    for name in with_intervals:
        out["models"][name] = model_block(y, predictions[name], days, n_boot, seed, True)
    for name in secondary:
        out["secondary_descriptive"][name] = model_block(y, predictions[name], days, n_boot, seed, False)
    for other in paired_against:
        out["paired_differences"][f"{primary}_minus_{other}"] = {
            m: day_block_bootstrap(y, predictions[primary], days, f, n_boot, seed, p_other=predictions[other])
            for m, f in INTERVAL_METRICS.items()}
    for month in sorted(set(months.tolist())):
        m = months == month
        out["by_month"][str(month)] = {
            name: {"metrics": probability_metrics(y[m], predictions[name][m]),
                   "calibration_in_the_large": calibration_in_the_large(y[m], predictions[name][m])}
            for name in [*with_intervals, *secondary]}
    return out
