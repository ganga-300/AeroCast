"""Stage 9 descriptive helpers (synthetic data only; no project test labels)."""

import math

import numpy as np
import pandas as pd
import pytest

from flight_delay.analysis.error_analysis import (LABEL_FREE_RAW_COLUMNS, assert_label_free_columns,
                                                  group_mean_predictions, prediction_summary,
                                                  prior_odds_shift, prior_shift_adjust,
                                                  reliability_gaps, required_logit_shift)


def test_label_free_column_guard():
    assert_label_free_columns(LABEL_FREE_RAW_COLUMNS)
    for bad in ("arr_delay", "dep_delay", "dep_time", "arr_time", "air_time", "tailnum", "not_a_column"):
        with pytest.raises(ValueError):
            assert_label_free_columns(["id", bad])


def test_prediction_summary_hand_values():
    s = prediction_summary([0.01, 0.2, 0.6, 0.3])
    assert s["n"] == 4 and s["mean"] == pytest.approx(0.2775)
    assert s["share_below_0.05"] == 0.25 and s["share_above_0.5"] == 0.25
    assert s["min"] == 0.01 and s["max"] == 0.6


def test_prior_shift_hand_calculation():
    # logit(0.5) - logit(0.2) = 0 - ln(0.25) = ln 4
    assert prior_odds_shift(0.2, 0.5) == pytest.approx(math.log(4))
    q = np.array([0.2])
    # odds 0.25 * 4 = 1 -> probability 0.5
    assert prior_shift_adjust(q, 0.2, 0.5)[0] == pytest.approx(0.5)
    np.testing.assert_allclose(prior_shift_adjust([0.1, 0.7], 0.3, 0.3), [0.1, 0.7])   # no shift


def test_required_logit_shift_recovers_known_shift():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.05, 0.6, 5000)
    z = np.log(q / (1 - q)) + 0.8
    target = float(np.mean(1 / (1 + np.exp(-z))))
    assert required_logit_shift(q, target) == pytest.approx(0.8, abs=1e-6)
    assert required_logit_shift(q, float(q.mean())) == pytest.approx(0.0, abs=1e-6)


def test_reliability_gaps_weighting():
    bins = [{"n": 100, "mean_predicted": 0.1, "observed_rate": 0.2},
            {"n": 300, "mean_predicted": 0.3, "observed_rate": 0.32}]
    r = reliability_gaps(bins)
    assert r["gaps"] == pytest.approx([0.1, 0.02])
    assert r["weighted_mean_gap"] == pytest.approx((0.1 * 100 + 0.02 * 300) / 400)
    assert r["all_bins_under_predicted"] is True


def test_group_mean_predictions_min_n():
    df = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 2, "period": ["x", "y"] * 4,
                       "p": [0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.5, 0.5]})
    t = group_mean_predictions(df, "g", "p", "period", min_n=2)
    assert list(t.index) == ["a"]                      # b has only 1 row per period
    assert t.loc["a", ("mean", "y")] == pytest.approx(0.3)
