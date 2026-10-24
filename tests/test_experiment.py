"""Partition and test-label-lock tests. Lock tests run in a temporary synthetic project root."""

import json

import pandas as pd
import pytest

from conftest import NAN, make_flights
from flight_delay.data.load import prepare_flights
from flight_delay.experiment import (ACCESS_LOG_REL, D32_STATEMENT, MANIFEST_REL, PROPOSAL_REL,
                                     HoldoutLockedError, assert_fit_rows, build_manifest, code_sha256,
                                     development_partitions, holdout_features_without_labels,
                                     partition, sha256_file, test_partition_sufficient,
                                     unlock_test_labels, validate_manifest)
from flight_delay.features.build import build_feature_table


@pytest.fixture
def table(cfg):
    """One labelled flight per day for the whole year, plus one unlabelled flight."""
    days = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    rows = [dict(month=d.month, day=d.day, arr_delay=float(20 if d.day % 3 == 0 else 0))
            for d in days]
    rows.append(dict(month=6, day=1, arr_delay=NAN, air_time=NAN))
    return build_feature_table(prepare_flights(make_flights(rows), cfg), cfg)


# ------------------------------------------------------------------ partitions

def test_development_partitions_never_contain_test(table, cfg):
    train, val = development_partitions(table, cfg)
    assert train["flight_date"].max() == pd.Timestamp("2013-08-31")
    assert val["flight_date"].min() == pd.Timestamp("2013-09-01")
    assert val["flight_date"].max() == pd.Timestamp("2013-10-31")
    assert train["y"].notna().all() and val["y"].notna().all()   # unlabelled row excluded


def test_partition_never_returns_test_labels(table, cfg):
    parts = partition(table, cfg)
    assert "y" not in parts["test"].columns and "label_status" not in parts["test"].columns
    assert "y" in parts["train"].columns and "y" in parts["validation"].columns
    t = holdout_features_without_labels(table, cfg)
    assert "y" not in t.columns and len(t) == 61


def test_test_sufficiency_is_pass_fail_only(table, cfg):
    assert test_partition_sufficient(table, cfg, 1) is True
    assert test_partition_sufficient(table, cfg, 10_000) is False


def test_partition_detects_missing_dates_and_overlap(table, cfg):
    with pytest.raises(AssertionError, match="no labelled flights"):
        partition(table[table["flight_date"] != pd.Timestamp("2013-09-15")], cfg)
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


# ------------------------------------------------------------------ lock

def write(root, rel, content):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def make_root(tmp_path, statement=D32_STATEMENT, features=("f1",), operating_point=None):
    """Synthetic project: config, raw data, artifacts, evaluation script, package code, proposal."""
    root = tmp_path / "proj"
    write(root, "configs/config.yaml", "seed: 42\n")
    write(root, "data/raw/flights.csv", "id\n1\n")
    write(root, "scripts/evaluate_test.py", "# evaluation\n")
    write(root, "src/flight_delay/mod.py", "X = 1\n")
    models = {}
    for fam, method in (("logistic_regression", "sigmoid"), ("xgboost", "isotonic")):
        arts = {k: write(root, f"models/{fam}_{k}.joblib", f"{fam}-{k}") for k in ("model", "calibrator")}
        models[fam] = {"hyperparameters": {"p": 1}, "calibration": {"method": method},
                       "operating_threshold": {"status": "none", "statement": statement},
                       "artifacts": {k: {"path": str(p.relative_to(root)), "sha256": sha256_file(p)}
                                     for k, p in arts.items()}}
    proposal = {
        "status": "PROPOSAL - FOR USER REVIEW", "primary_model": "logistic_regression",
        "comparison_model": "xgboost", "models": models,
        "operating_point_D32": operating_point if operating_point is not None else statement,
        "prediction_contract": {"features": list(features), "raw_path": "data/raw/flights.csv"},
        "calibration_and_threshold_procedure_D11": {"criterion": "mean_heldout_log_loss"},
        "test_evaluation_plan": {"metrics": ["average_precision"]},
        "evaluation_script": {"path": "scripts/evaluate_test.py",
                              "sha256": sha256_file(root / "scripts/evaluate_test.py")},
        "code_sha256": code_sha256(root),
        "reproducibility": {"config_sha256": sha256_file(root / "configs/config.yaml"),
                            "raw_sha256": sha256_file(root / "data/raw/flights.csv")},
    }
    write(root, PROPOSAL_REL, json.dumps(proposal))
    return root


def freeze(root):
    manifest = build_manifest("approved in test", root)
    write(root, MANIFEST_REL, json.dumps(manifest))
    return manifest


def test_no_manifest_keeps_labels_locked(table, cfg, tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(HoldoutLockedError, match="No frozen manifest"):
        unlock_test_labels(table, cfg, "unit test", root=root)
    assert not (root / ACCESS_LOG_REL).exists()


def test_valid_manifest_unlocks_once_and_logs_hashes(table, cfg, tmp_path):
    root = make_root(tmp_path)
    manifest = freeze(root)
    y = unlock_test_labels(table, cfg, "unit test", root=root)
    assert len(y) == 61 and y.notna().all()
    entry = json.loads((root / ACCESS_LOG_REL).read_text().splitlines()[0])
    assert entry["access_number"] == 1
    assert entry["manifest_sha256"] == sha256_file(root / MANIFEST_REL)
    assert entry["approved_proposal_sha256"] == manifest["approved_proposal_sha256"]
    with pytest.raises(HoldoutLockedError, match="already released"):
        unlock_test_labels(table, cfg, "second", root=root)
    unlock_test_labels(table, cfg, "second", root=root, repeat_reason="disclosed defect fix")
    entries = [json.loads(l) for l in (root / ACCESS_LOG_REL).read_text().splitlines()]
    assert [e["access_number"] for e in entries] == [1, 2]
    assert entries[1]["repeat_reason"] == "disclosed defect fix"


def test_proposal_copied_as_manifest_is_rejected(table, cfg, tmp_path):
    root = make_root(tmp_path)
    write(root, MANIFEST_REL, (root / PROPOSAL_REL).read_text())
    with pytest.raises(HoldoutLockedError, match="not 'FROZEN'"):
        unlock_test_labels(table, cfg, "unit test", root=root)


MANIFEST_TAMPERING = {
    "lowercase status": lambda m: m.update(status="frozen"),
    "naive timestamp": lambda m: m.update(frozen_at="2026-10-24T10:00:00"),
    "non-timestamp": lambda m: m.update(frozen_at="x"),
    "empty approval note": lambda m: m.update(approval_note="  "),
    "wrong proposal hash": lambda m: m.update(approved_proposal_sha256="0" * 64),
    "threshold added": lambda m: m["models"]["logistic_regression"].update(
        operating_threshold={"status": "frozen", "value": 0.5}),
    "calibration changed": lambda m: m["models"]["xgboost"]["calibration"].update(method="sigmoid"),
    "missing field": lambda m: m.pop("code_sha256"),
}


@pytest.mark.parametrize("name", MANIFEST_TAMPERING)
def test_tampered_manifest_is_rejected(name, tmp_path):
    root = make_root(tmp_path)
    manifest = build_manifest("approved in test", root)
    MANIFEST_TAMPERING[name](manifest)
    with pytest.raises(HoldoutLockedError):
        validate_manifest(manifest, root)


FILE_TAMPERING = {
    "artifact": "models/logistic_regression_model.joblib",
    "config": "configs/config.yaml",
    "raw data": "data/raw/flights.csv",
    "evaluation script": "scripts/evaluate_test.py",
    "package code": "src/flight_delay/mod.py",
    "proposal": str(PROPOSAL_REL),
}


@pytest.mark.parametrize("name", FILE_TAMPERING)
def test_files_changed_after_freezing_block_release(name, table, cfg, tmp_path):
    root = make_root(tmp_path)
    freeze(root)
    path = root / FILE_TAMPERING[name]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(HoldoutLockedError):
        unlock_test_labels(table, cfg, "unit test", root=root)
    assert not (root / ACCESS_LOG_REL).exists()


def test_malformed_manifest_files_fail_cleanly(table, cfg, tmp_path):
    root = make_root(tmp_path)
    for content in ("[1, 2]", "{not json", '"FROZEN"'):
        write(root, MANIFEST_REL, content)
        with pytest.raises(HoldoutLockedError):
            unlock_test_labels(table, cfg, "unit test", root=root)
    assert not (root / ACCESS_LOG_REL).exists()


def test_semantic_requirements_checked_even_when_proposal_agrees(tmp_path):
    root = make_root(tmp_path, statement="Threshold 0.5 selected")
    with pytest.raises(HoldoutLockedError, match="D32"):
        validate_manifest(build_manifest("approved", root), root)
    # Top-level operating point wrong while every model still says "none"
    root3 = make_root(tmp_path / "c", operating_point="Threshold 0.5 selected")
    with pytest.raises(HoldoutLockedError, match="not the D32 statement"):
        validate_manifest(build_manifest("approved", root3), root3)
    root2 = make_root(tmp_path / "b", features=())
    with pytest.raises(HoldoutLockedError, match="Feature list is empty"):
        validate_manifest(build_manifest("approved", root2), root2)
