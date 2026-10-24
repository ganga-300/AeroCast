import math

import numpy as np
import pytest

from flight_delay.evaluation.metrics import (MetricUndefinedError, average_precision, brier,
                                             day_block_bootstrap, logloss, probability_metrics,
                                             roc_auc)


def test_average_precision_hand_calculated():
    # Ranked by score: 0.9 (pos), 0.8 (neg), 0.7 (pos), 0.1 (neg)
    # Positives retrieved at ranks 1 and 3: AP = 0.5*(1/1) + 0.5*(2/3) = 5/6
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)


def test_roc_auc_hand_calculated():
    # Pairs (pos, neg): (0.9>0.8), (0.9>0.1), (0.7<0.8), (0.7>0.1) -> 3/4
    assert roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)


def test_brier_and_logloss_hand_calculated():
    y, p = [1, 0], [0.8, 0.4]
    assert brier(y, p) == pytest.approx(((1 - 0.8) ** 2 + 0.4 ** 2) / 2)
    assert logloss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)


def test_constant_prediction_baselines():
    y = [1, 0, 0, 0]
    assert average_precision(y, [0.25] * 4) == pytest.approx(0.25)   # AP of a constant = prevalence
    assert roc_auc(y, [0.25] * 4) == pytest.approx(0.5)


@pytest.mark.parametrize("y,p", [([], []), ([1, 1], [0.2, 0.3]), ([0, 1], [0.2]),
                                 ([0, 1], [0.2, 1.2]), ([0, 1], [0.2, float("nan")]),
                                 ([0, 2], [0.2, 0.3])])
def test_undefined_inputs_raise(y, p):
    with pytest.raises(MetricUndefinedError):
        probability_metrics(y, p)


def test_logloss_finite_for_hard_predictions():
    assert np.isfinite(logloss([1, 0], [0.0, 1.0]))


def test_day_bootstrap_interval_contains_point_and_paired_zero():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(30), 50)
    y = rng.integers(0, 2, days.size)
    p = np.clip(y * 0.3 + rng.random(days.size) * 0.7, 0, 1)
    r = day_block_bootstrap(y, p, days, average_precision, n_boot=100, seed=1)
    assert r["ci95_low"] <= r["point"] <= r["ci95_high"]
    same = day_block_bootstrap(y, p, days, average_precision, n_boot=50, seed=1, p_other=p)
    assert same["point"] == 0 and same["ci95_low"] == same["ci95_high"] == 0
