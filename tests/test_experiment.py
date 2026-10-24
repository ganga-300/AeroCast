import json

import pandas as pd
import pytest

from conftest import NAN, make_flights
from flight_delay.data.load import prepare_flights
from flight_delay.experiment import (HoldoutLockedError, assert_fit_rows, development_partitions,
                                     partition, holdout_features_without_labels, unlock_test_labels)
from flight_delay.features.build import build_feature_table


@pytest.fixture
def table(cfg):
    """One labelled flight per day for the whole year, plus one unlabelled flight."""
    days = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    rows = [dict(month=d.month, day=d.day, arr_delay=float(20 if d.day % 3 == 0 else 0))
            for d in days]
    rows.append(dict(month=6, day=1, arr_delay=NAN, air_time=NAN))
    return build_feature_table(prepare_flights(make_flights(rows), cfg), cfg)


def test_development_partitions_never_contain_test(table, cfg):
    train, val = development_partitions(table, cfg)
    assert train["flight_date"].max() == pd.Timestamp("2013-08-31")
    assert val["flight_date"].min() == pd.Timestamp("2013-09-01")
    assert val["flight_date"].max() == pd.Timestamp("2013-10-31")
    assert not (train["split"] == "test").any() and not (val["split"] == "test").any()
    assert train["y"].notna().all() and val["y"].notna().all()   # unlabelled row excluded


def test_holdout_features_carry_no_labels(table, cfg):
    t = holdout_features_without_labels(table, cfg)
    assert "y" not in t.columns and "label_status" not in t.columns
    assert t["flight_date"].min() == pd.Timestamp("2013-11-01") and len(t) == 61


def test_partition_detects_missing_dates(table, cfg):
    gap = table[table["flight_date"] != pd.Timestamp("2013-09-15")]
    with pytest.raises(AssertionError, match="no labelled flights"):
        partition(gap, cfg)


def test_partition_detects_overlap(table, cfg):
    bad = table.copy()
    bad["split"] = bad["split"].astype(object)
    bad.loc[bad["flight_date"] == pd.Timestamp("2013-10-31"), "split"] = "train"
    with pytest.raises(AssertionError):
        partition(bad, cfg)


def test_fit_rows_must_be_training(table, cfg):
    train, val = development_partitions(table, cfg)
    assert_fit_rows(train, cfg)
    with pytest.raises(AssertionError):
        assert_fit_rows(pd.concat([train, val.head(1)]), cfg)


def test_test_labels_locked_without_complete_manifest(table, cfg, tmp_path):
    log = tmp_path / "access.jsonl"
    with pytest.raises(HoldoutLockedError):
        unlock_test_labels(table, cfg, tmp_path / "missing.json", log, "unit test")
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"frozen_at": "2026-10-24", "model": "xgb"}))
    with pytest.raises(HoldoutLockedError, match="incomplete"):
        unlock_test_labels(table, cfg, partial, log, "unit test")
    assert not log.exists()   # failed attempts release nothing and log nothing


def test_unlock_releases_labels_and_logs_each_access(table, cfg, tmp_path):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"frozen_at": "2026-10-24", "model": "m", "features": ["f"],
                                    "preprocessing": "p", "hyperparameters": {"a": 1},
                                    "threshold": 0.5, "calibration": "none"}))
    log = tmp_path / "access.jsonl"
    y = unlock_test_labels(table, cfg, manifest, log, "unit test")
    assert len(y) == 61 and y.notna().all()
    unlock_test_labels(table, cfg, manifest, log, "second access")
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert [e["purpose"] for e in entries] == ["unit test", "second access"]
