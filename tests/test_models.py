"""Model-wrapper tests on toy data. Toy results are not project results."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_flights
from flight_delay.data.load import prepare_flights
from flight_delay.experiment import development_partitions, holdout_features_without_labels
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.models import FlightDelayModel, training_scale_pos_weight

XGB = dict(max_depth=3, min_child_weight=1, scale_pos_weight="train_balanced", learning_rate=0.3,
           subsample=0.8, colsample_bytree=0.8, n_estimators_max=30, early_stopping_rounds=5,
           early_stopping_metric="aucpr")
LR = dict(C=1.0, class_weight=None, max_iter=2000)


@pytest.fixture
def table(cfg):
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2013-01-01", "2013-12-31", freq="D"):
        for carrier, dest, hour in [("AA", "BOS", 6), ("UA", "ORD", 12), ("B6", "MIA", 18), ("DL", "ATL", 20)]:
            late = rng.random() < 0.15 + 0.02 * (hour - 6)
            rows.append(dict(month=d.month, day=d.day, carrier=carrier, dest=dest,
                             sched_dep_time=hour * 100, arr_delay=float(40 if late else -5)))
    return build_feature_table(prepare_flights(make_flights(rows), cfg), cfg)


def fit(family, params, table, cfg, seed=0):
    train, val = development_partitions(table, cfg)
    return FlightDelayModel(family, params, feature_specs(cfg), cfg, seed).fit(
        train, val if family == "xgboost" else None)


@pytest.mark.parametrize("family,params", [("logistic_regression", LR), ("xgboost", XGB)])
def test_fit_rejects_non_training_rows(family, params, table, cfg):
    train, val = development_partitions(table, cfg)
    m = FlightDelayModel(family, params, feature_specs(cfg), cfg, 0)
    with pytest.raises(AssertionError):
        m.fit(pd.concat([train, val.head(2)]), val if family == "xgboost" else None)


def test_early_stopping_rows_must_be_validation(table, cfg):
    train, val = development_partitions(table, cfg)
    specs = feature_specs(cfg)
    with pytest.raises(AssertionError, match="validation"):
        FlightDelayModel("xgboost", XGB, specs, cfg, 0).fit(train, train.tail(50))
    with pytest.raises(ValueError):
        FlightDelayModel("xgboost", XGB, specs, cfg, 0).fit(train)           # XGBoost needs eval rows
    with pytest.raises(ValueError):
        FlightDelayModel("logistic_regression", LR, specs, cfg, 0).fit(train, val)


@pytest.mark.parametrize("family,params", [("logistic_regression", LR), ("xgboost", XGB),
                                           ("dummy_most_frequent", {}), ("prior_baseline", {})])
def test_predictions_valid_and_aligned_by_row(family, params, table, cfg):
    m = fit(family, params, table, cfg)
    _, val = development_partitions(table, cfg)
    p = m.predict_proba(val)
    assert p.index.equals(val.index) and len(p) == len(val)
    assert np.isfinite(p).all() and p.between(0, 1).all()
    shuffled = val.sample(frac=1.0, random_state=3)
    pd.testing.assert_series_equal(m.predict_proba(shuffled).sort_index(), p.sort_index())


@pytest.mark.parametrize("family,params", [("logistic_regression", LR), ("xgboost", XGB)])
def test_deterministic_given_seed(family, params, table, cfg):
    _, val = development_partitions(table, cfg)
    a, b = fit(family, params, table, cfg, seed=7), fit(family, params, table, cfg, seed=7)
    np.testing.assert_array_equal(a.predict_proba(val).to_numpy(), b.predict_proba(val).to_numpy())


def test_class_weight_uses_training_labels_only(table, cfg):
    train, val = development_partitions(table, cfg)
    flipped = val.copy()
    flipped["y"] = (1 - flipped["y"].astype(int)).astype("Int8")
    a = FlightDelayModel("xgboost", XGB, feature_specs(cfg), cfg, 0).fit(train, val)
    b = FlightDelayModel("xgboost", XGB, feature_specs(cfg), cfg, 0).fit(train, flipped)
    expected = training_scale_pos_weight(train["y"])
    assert a.train_info_["resolved_params"]["scale_pos_weight"] == pytest.approx(expected)
    assert b.train_info_["resolved_params"]["scale_pos_weight"] == pytest.approx(expected)
    y = train["y"].astype(int)
    assert expected == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_logistic_regression_never_sees_validation_labels(table, cfg):
    train, val = development_partitions(table, cfg)
    a = FlightDelayModel("logistic_regression", LR, feature_specs(cfg), cfg, 0).fit(train)
    altered = table.copy()
    altered.loc[altered["split"] == "validation", "y"] = 1
    train2, val2 = development_partitions(altered, cfg)
    b = FlightDelayModel("logistic_regression", LR, feature_specs(cfg), cfg, 0).fit(train2)
    pd.testing.assert_series_equal(a.predict_proba(val), b.predict_proba(val2))


def test_reference_baselines(table, cfg):
    train, val = development_partitions(table, cfg)
    prev = train["y"].astype(int).mean()
    assert prev < 0.5
    assert (fit("dummy_most_frequent", {}, table, cfg).predict_proba(val) == 0).all()
    np.testing.assert_allclose(fit("prior_baseline", {}, table, cfg).predict_proba(val), prev)


def test_no_refit_and_no_predict_before_fit(table, cfg):
    train, val = development_partitions(table, cfg)
    m = FlightDelayModel("logistic_regression", LR, feature_specs(cfg), cfg, 0)
    with pytest.raises(RuntimeError):
        m.predict_proba(val)
    m.fit(train)
    with pytest.raises(RuntimeError):
        m.fit(train)


def test_frozen_model_scores_rows_without_labels(table, cfg):
    """A fitted model can score label-free holdout features (for the later final evaluation)."""
    m = fit("logistic_regression", LR, table, cfg)
    x = holdout_features_without_labels(table, cfg)
    assert "y" not in x.columns and len(m.predict_proba(x)) == len(x)
