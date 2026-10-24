"""Calibration and threshold-selection tests (hand-calculated toy examples)."""

import numpy as np
import pytest

from flight_delay.evaluation.calibration import (Calibrator, compare_methods, cross_fitted,
                                                 select_method)
from flight_delay.evaluation.threshold import (metrics_at_threshold, recall_at_precision,
                                               select_threshold, threshold_table)

Y = [1, 1, 0, 1, 0, 0]
P = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def test_threshold_table_hand_calculated():
    t = threshold_table(Y, P)
    assert t["threshold"].tolist() == P
    assert t["precision"].round(4).tolist() == [1.0, 1.0, 0.6667, 0.75, 0.6, 0.5]
    assert t["recall"].round(4).tolist() == [0.3333, 0.6667, 0.6667, 1.0, 1.0, 1.0]


def test_select_threshold_maximises_recall_subject_to_precision():
    r = select_threshold(Y, P, 0.70)
    # Thresholds 0.9, 0.8, 0.6 meet precision >= 0.70; 0.6 has the highest recall (1.0)
    assert r["feasible"] and r["threshold"] == 0.6 and r["recall"] == 1.0 and r["precision"] == 0.75


def test_recall_ties_go_to_highest_threshold():
    y = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    p = [0.95, 0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.15, 0.1, 0.05]
    # At 0.80: P = 4/4, R = 4/5; at 0.75: P = 4/5, R = 4/5. Both meet 0.70 -> pick 0.80
    # At 0.30: P = 5/6 >= 0.70, R = 1.0 -> highest recall wins overall
    assert select_threshold(y, p, 0.70)["threshold"] == 0.3
    assert select_threshold(y[:5] + [0] * 5, p, 0.70)["threshold"] == 0.8


def test_exact_boundary_precision_counts_as_meeting():
    # Recall 1.0 is reached only at the lowest threshold, where precision is exactly 7/10.
    y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    p = list(np.linspace(0.9, 0.1, 10))
    r = select_threshold(y, p, 0.70)
    assert r["feasible"] and r["recall"] == 1.0 and r["precision"] == pytest.approx(0.7)
    assert r["threshold"] == pytest.approx(0.1)


def test_tied_scores_are_flagged_together():
    t = threshold_table([1, 0, 1], [0.5, 0.5, 0.2])
    assert t["n_flagged"].tolist() == [2, 3] and t["precision"].tolist()[0] == 0.5


def test_infeasible_requirement_is_reported_not_relaxed():
    r = select_threshold([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6], 0.70)
    assert r["feasible"] is False and r["max_attainable_precision"] == 0.5
    assert "threshold" not in r
    assert recall_at_precision([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6], 0.70) == 0.0


def test_metrics_at_threshold_confusion_matrix():
    m = metrics_at_threshold(Y, P, 0.6)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 0, 2)
    assert m["precision"] == 0.75 and m["recall"] == 1.0 and m["f1"] == pytest.approx(2 * 0.75 / 1.75)
    none = metrics_at_threshold(Y, P, 0.99)
    assert none["precision"] is None and none["n_flagged"] == 0     # undefined, not zero


def test_calibrators_basic_properties():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    true = 1 / (1 + np.exp(-(2 * np.log(p / (1 - p)) - 1)))       # known Platt map: a=2, b=-1
    y = (rng.random(p.size) < true).astype(int)
    np.testing.assert_array_equal(Calibrator("none").fit(p, y).transform(p), p)
    sig = Calibrator("sigmoid").fit(p, y)
    assert sig.fit_info_["a"] == pytest.approx(2, abs=0.1) and sig.fit_info_["b"] == pytest.approx(-1, abs=0.1)
    iso = Calibrator("isotonic").fit(p, y)
    q = iso.transform(np.sort(p))
    assert (np.diff(q) >= 0).all() and q.min() >= 0 and q.max() <= 1


def test_cross_fit_never_uses_a_rows_own_fold_labels():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < p).astype(int)
    fold = np.array(["sep"] * 200 + ["oct"] * 200)
    for method in ("sigmoid", "isotonic"):
        a = cross_fitted(p, y, fold, method)
        y2 = y.copy()
        y2[fold == "sep"] = 1 - y2[fold == "sep"]                  # change September labels only
        b = cross_fitted(p, y2, fold, method)
        np.testing.assert_array_equal(a[fold == "sep"], b[fold == "sep"])   # Sep rows unaffected
        assert not np.array_equal(a[fold == "oct"], b[fold == "oct"])       # Oct rows use Sep fit


def test_select_method_uses_mean_heldout_log_loss_and_simpler_tie_break():
    res = {"none": {"mean_heldout_log_loss": 0.4000004}, "sigmoid": {"mean_heldout_log_loss": 0.4},
           "isotonic": {"mean_heldout_log_loss": 0.39}}
    assert select_method(res, ["none", "sigmoid", "isotonic"], 1e-6)["selected"] == "isotonic"
    res["isotonic"]["mean_heldout_log_loss"] = 0.41
    s = select_method(res, ["none", "sigmoid", "isotonic"], 1e-6)
    assert s["selected"] == "none" and s["tie_break_applied"]       # 4e-7 apart -> simpler wins


def test_compare_methods_reports_each_fold_and_finite_losses():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 1.0, 600)
    y = (rng.random(600) < p ** 2).astype(int)
    fold = np.array(["sep"] * 300 + ["oct"] * 300)
    res = compare_methods(p, y, fold, ["none", "sigmoid", "isotonic"])
    for m in res.values():
        assert set(m["per_fold"]) == {"sep", "oct"}
        assert np.isfinite(m["mean_heldout_log_loss"])
        assert m["mean_heldout_log_loss"] == pytest.approx(
            np.mean([v["log_loss"] for v in m["per_fold"].values()]))
