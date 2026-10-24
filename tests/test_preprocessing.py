"""Preprocessing tests on toy data (no model is trained)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import make_flights
from flight_delay.contract import PROHIBITED_INPUTS
from flight_delay.data.load import prepare_flights
from flight_delay.experiment import development_partitions, holdout_features_without_labels
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.features.specs import FeatureSpec
from flight_delay.preprocessing import FAMILIES, TrainOnlyPreprocessor, column_groups


@pytest.fixture
def pcfg(cfg):
    cfg["preprocessing"]["min_category_frequency"] = 3
    return cfg


@pytest.fixture
def table(pcfg):
    """Three flights per day all year. 'RR' is a rare training carrier (2 flights); 'ZZ'
    appears only in validation; 'NEW' destination appears only in validation/test."""
    rows = []
    for i, d in enumerate(pd.date_range("2013-01-01", "2013-12-31", freq="D")):
        for j, (carrier, dest) in enumerate([("AA", "BOS"), ("UA", "ORD"), ("B6", "MIA")]):
            rows.append(dict(month=d.month, day=d.day, carrier=carrier, dest=dest,
                             sched_dep_time=600 + 300 * j, distance=200 + 500 * j,
                             arr_delay=float((i * 7 + j * 11) % 50 - 15)))
    rows[10]["carrier"] = rows[20]["carrier"] = "RR"
    for r in rows:
        if (r["month"], r["day"]) in {(9, 10), (9, 11), (11, 5)}:
            r["carrier"], r["dest"] = "ZZ", "NEW"
    return build_feature_table(prepare_flights(make_flights(rows), pcfg), pcfg)


def fitted(family, table, cfg):
    train, _ = development_partitions(table, cfg)
    return TrainOnlyPreprocessor(family, feature_specs(cfg), cfg).fit(train)


@pytest.mark.parametrize("family", FAMILIES)
def test_fit_rejects_non_training_rows(family, table, pcfg):
    train, val = development_partitions(table, pcfg)
    with pytest.raises(AssertionError):
        TrainOnlyPreprocessor(family, feature_specs(pcfg), pcfg).fit(pd.concat([train, val.head(3)]))
    with pytest.raises(AssertionError):
        TrainOnlyPreprocessor(family, feature_specs(pcfg), pcfg).fit(val)


@pytest.mark.parametrize("family", FAMILIES)
def test_no_refit_and_no_transform_before_fit(family, table, pcfg):
    train, val = development_partitions(table, pcfg)
    p = TrainOnlyPreprocessor(family, feature_specs(pcfg), pcfg)
    with pytest.raises(RuntimeError):
        p.transform(val)
    p.fit(train)
    with pytest.raises(RuntimeError):
        p.fit(train)


@pytest.mark.parametrize("family", FAMILIES)
def test_transform_never_changes_learned_parameters(family, table, pcfg):
    p = fitted(family, table, pcfg)
    _, val = development_partitions(table, pcfg)
    before = pickle.dumps(p.transformer)
    shifted = val.copy()
    shifted["distance"] *= 100                      # wildly different distribution
    p.transform(shifted)
    p.transform(holdout_features_without_labels(table, pcfg))
    assert pickle.dumps(p.transformer) == before


def test_scaling_uses_training_statistics_only(table, pcfg):
    p = fitted("logistic_regression", table, pcfg)
    train, val = development_partitions(table, pcfg)
    mu, sd = train["distance"].mean(), train["distance"].std(ddof=0)
    out = p.transform(val)
    np.testing.assert_allclose(out["distance"], (val["distance"] - mu) / sd)


@pytest.mark.parametrize("family", FAMILIES)
def test_rare_and_unseen_categories(family, table, pcfg):
    p = fitted(family, table, pcfg)
    train, val = development_partitions(table, pcfg)
    state = p.learned_state()
    assert state["one_hot_infrequent"]["carrier"] == ["RR"]        # learned from training counts
    out_v = p.transform(val)
    out_t = p.transform(holdout_features_without_labels(table, pcfg))
    assert list(out_v.columns) == list(out_t.columns) == p.feature_names_out_
    zz = (val["carrier"] == "ZZ").to_numpy()
    assert zz.sum() == 6
    # unseen carrier -> the training-learned infrequent column
    assert (out_v.loc[zz, "carrier_infrequent_sklearn"] == 1).all()
    assert (out_v.loc[zz, ["carrier_AA", "carrier_UA", "carrier_B6"]] == 0).all().all()
    # unseen destination with no infrequent destination in training -> all-zero destination block
    dest_cols = [c for c in out_v.columns if c.startswith("dest_")]
    assert "dest_infrequent_sklearn" not in dest_cols
    assert (out_v.loc[zz, dest_cols] == 0).all().all()


def test_cold_start_imputation_logistic_vs_xgboost(table, pcfg):
    train, _ = development_partitions(table, pcfg)
    cold = train[train["hist_all_rate_exp"].isna()]
    assert len(cold) == 6                                           # Jan 1-2, three flights each
    lr = fitted("logistic_regression", table, pcfg)
    xgb = fitted("xgboost", table, pcfg)
    out_lr, out_x = lr.transform(train), xgb.transform(train)
    # LR: median of the TRAINING column, then standardised with training statistics
    imp = lr.transformer.named_transformers_["hist_rates"]
    med = train["hist_carrier_rate_28d"].median()
    assert imp.named_steps["impute"].statistics_[
        lr.hist_rate_columns.index("hist_carrier_rate_28d")] == pytest.approx(med)
    assert not out_lr.isna().any().any()
    # XGBoost: NaN preserved for native handling
    assert out_x.loc[cold.index, "hist_carrier_rate_28d"].isna().all()
    # Same indicator definition for both families
    for out in (out_lr, out_x):
        assert (out.loc[cold.index, "history_unavailable"] == 1).all()
        assert (out.drop(index=cold.index)["history_unavailable"] == 0).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_column_order_and_extra_columns_do_not_matter(family, table, pcfg):
    p = fitted(family, table, pcfg)
    _, val = development_partitions(table, pcfg)
    reordered = val[list(reversed(val.columns))].assign(arr_delay=999.0, dep_delay=999.0)
    pd.testing.assert_frame_equal(p.transform(val), p.transform(reordered))


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_input_column_raises(family, table, pcfg):
    p = fitted(family, table, pcfg)
    _, val = development_partitions(table, pcfg)
    with pytest.raises(ValueError, match="missing required"):
        p.transform(val.drop(columns="hist_dest_rate_28d"))


@pytest.mark.parametrize("family", FAMILIES)
def test_output_types_and_determinism(family, table, pcfg):
    a, b = fitted(family, table, pcfg), fitted(family, table, pcfg)
    _, val = development_partitions(table, pcfg)
    snapshot = val.copy(deep=True)
    out = a.transform(val)
    pd.testing.assert_frame_equal(val, snapshot)                    # input not mutated
    pd.testing.assert_frame_equal(out, b.transform(val))            # identical refits
    assert all(t == "float64" for t in out.dtypes)
    assert out.shape == (len(val), len(a.feature_names_out_))
    assert a.fit_info_["date_max"] <= "2013-08-31"


def test_inputs_are_declared_permitted_features_only(pcfg):
    specs = feature_specs(pcfg)
    names = {s.name for s in specs}
    for family in FAMILIES:
        p = TrainOnlyPreprocessor(family, specs, pcfg)
        assert set(p.input_columns) <= names
        assert not set(p.input_columns) & PROHIBITED_INPUTS
        assert "route" not in p.input_columns                       # D19


def test_every_feature_placed_exactly_once(pcfg):
    specs = feature_specs(pcfg)
    column_groups(specs)
    with pytest.raises(ValueError):
        column_groups(specs + [FeatureSpec("extra", "numeric", "schedule", ("distance",), "")])
