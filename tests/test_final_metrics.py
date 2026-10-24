"""Registered final-evaluation metric definitions on synthetic data (no project data)."""

import numpy as np
import pytest

from flight_delay.evaluation.final import (calibration_in_the_large, calibration_intercept_slope,
                                           evaluate_predictions, reliability_bins)


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < q).astype(int)                 # q perfectly calibrated
    days = np.repeat(np.arange(n // 100), 100)
    months = np.where(days < (n // 200), "2013-11", "2013-12")
    return y, q, days, months


def test_calibration_in_the_large_sign():
    assert calibration_in_the_large([0, 0, 1, 1], [0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.1)


def test_intercept_slope_recovers_known_miscalibration():
    rng = np.random.default_rng(1)
    q = rng.uniform(0.02, 0.98, 40000)
    lq = np.log(q / (1 - q))
    y = (rng.random(q.size) < 1 / (1 + np.exp(-(0.5 + 0.7 * lq)))).astype(int)
    r = calibration_intercept_slope(y, q)
    assert r["intercept"] == pytest.approx(0.5, abs=0.05) and r["slope"] == pytest.approx(0.7, abs=0.05)
    assert calibration_intercept_slope([0, 1, 0], [0.3, 0.3, 0.3]) == {"status": "undefined (constant prediction)"}


def test_reliability_bins_merge_duplicate_edges():
    y = np.array([0, 1] * 50)
    step = np.repeat([0.1, 0.2, 0.3], [40, 40, 20])          # step function: only 3 distinct values
    bins = reliability_bins(y, step)
    assert 1 <= len(bins) <= 3 and sum(b["n"] for b in bins) == 100
    assert len(reliability_bins(*synthetic()[:2])) == 10
    assert len(reliability_bins(y, np.full(100, 0.2))) == 1  # constant baseline: one bin


def test_evaluate_predictions_registered_structure_and_signs():
    y, q, days, months = synthetic()
    preds = {"primary": q, "worse": np.clip(q + np.random.default_rng(2).normal(0, 0.2, q.size), 0.001, 0.999),
             "prior": np.full(q.size, y.mean()), "dummy": np.zeros(q.size), "primary_raw": np.clip(q * 1.3, 0, 1)}
    r = evaluate_predictions(y, preds, days, months, primary="primary",
                             with_intervals=["primary", "worse", "prior", "dummy"],
                             secondary=["primary_raw"], paired_against=["worse", "prior"],
                             n_boot=50, seed=42)
    assert set(r["models"]) == {"primary", "worse", "prior", "dummy"}
    assert "intervals_95" not in r["secondary_descriptive"]["primary_raw"]
    assert r["threshold_dependent_metrics"]["status"] == "NOT APPLICABLE (D32)"
    for m in ("average_precision", "roc_auc", "brier", "log_loss"):
        assert m in r["models"]["primary"]["intervals_95"]
    d = r["paired_differences"]["primary_minus_worse"]
    assert d["average_precision"]["point"] > 0 and d["roc_auc"]["point"] > 0     # primary better
    assert d["brier"]["point"] < 0 and d["log_loss"]["point"] < 0
    # Constant baselines through the same code path
    assert r["models"]["prior"]["metrics"]["roc_auc"] == 0.5
    assert r["models"]["prior"]["intervals_95"]["roc_auc"]["ci95_low"] == 0.5
    assert r["models"]["prior"]["calibration_intercept_slope"]["status"].startswith("undefined")
    assert r["models"]["prior"]["metrics"]["average_precision"] == pytest.approx(y.mean())
    assert set(r["by_month"]) == {"2013-11", "2013-12"}
    assert r["models"]["primary"]["intervals_95"]["average_precision"]["n_undefined"] == 0
    # A perfectly calibrated predictor has near-zero calibration-in-the-large
    assert abs(r["models"]["primary"]["calibration_in_the_large"]) < 0.02
