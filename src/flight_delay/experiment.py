"""Chronological experimental design (Stage 5).

Three responsibilities:

1. **Partitions.** Split the labelled modelling population by flight date and assert the
   chronological guarantees (no overlap, one partition per date, contiguous date coverage).
2. **History vs. fitting data.** Historical features are computed once over all dates
   (`features.build`). Model fitting may only ever see training rows (`assert_fit_rows`).
3. **Test-set lock.** Development code obtains data through `development_partitions`, which
   never returns test rows. Test features can be scored without labels
   (`holdout_features_without_labels`). Test labels are released only by `unlock_test_labels`,
   which requires a frozen-decision manifest and appends every access to an audit log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.data.load import modelling_population
from flight_delay.data.split import SPLIT_ORDER, assert_chronological, parse_boundaries

LABEL_COLUMNS = ("y", "label_status")
FREEZE_REQUIRED_KEYS = ("frozen_at", "model", "features", "preprocessing",
                        "hyperparameters", "threshold", "calibration")


class HoldoutLockedError(RuntimeError):
    """Raised when code tries to read test labels before modelling decisions are frozen."""


def partition(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Labelled rows per partition, with chronological assertions. Internal: callers outside
    the final evaluation should use `development_partitions` instead."""
    pop = modelling_population(table)
    assert_chronological(pop["flight_date"], pop["split"])
    bounds = parse_boundaries(cfg["split"])
    parts = {}
    for name in SPLIT_ORDER:
        part = pop[pop["split"] == name]
        r = bounds[name]
        if not part["flight_date"].between(r.start, r.end).all():
            raise AssertionError(f"Rows outside the configured range in '{name}'")
        expected_days = pd.date_range(r.start, r.end, freq="D")
        missing = expected_days.difference(pd.DatetimeIndex(part["flight_date"].unique()))
        if len(missing):
            raise AssertionError(f"'{name}' has dates with no labelled flights: {list(missing[:5])}")
        parts[name] = part
    ids = [set(p["id"]) for p in parts.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1:]):
        raise AssertionError("A flight id appears in more than one partition")
    return parts


def development_partitions(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) labelled rows. The only data source for any modelling decision."""
    parts = partition(table, cfg)
    return parts["train"], parts["validation"]


def holdout_features_without_labels(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Test-period rows of the modelling population with label columns removed, so a frozen
    pipeline can score them without the labels being visible."""
    return partition(table, cfg)["test"].drop(columns=list(LABEL_COLUMNS))


def assert_fit_rows(fit: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Model fitting (and any learned preprocessing) may use labelled training rows only."""
    r = parse_boundaries(cfg["split"])["train"]
    if not (fit["split"] == "train").all():
        raise AssertionError("Fitting data contains non-training rows")
    if not fit["flight_date"].between(r.start, r.end).all():
        raise AssertionError("Fitting data contains dates outside the training range")
    if fit["y"].isna().any():
        raise AssertionError("Fitting data contains unlabelled rows")


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = [k for k in FREEZE_REQUIRED_KEYS if manifest.get(k) in (None, "", {})]
    if missing:
        raise HoldoutLockedError(f"Frozen manifest incomplete; missing: {missing}")


def unlock_test_labels(table: pd.DataFrame, cfg: dict[str, Any], manifest_path: Path,
                       access_log: Path, purpose: str) -> pd.Series:
    """Release test labels (indexed by flight id) only after decisions are frozen.

    Every call is appended to `access_log` with the manifest it was made under, so the
    number of times the test set was evaluated is auditable.
    """
    if not manifest_path.exists():
        raise HoldoutLockedError(f"No frozen manifest at {manifest_path}; test labels stay locked")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    test = partition(table, cfg)["test"]
    access_log.parent.mkdir(parents=True, exist_ok=True)
    with access_log.open("a") as f:
        f.write(json.dumps({"accessed_at": datetime.now(timezone.utc).isoformat(),
                            "purpose": purpose, "manifest": str(manifest_path),
                            "manifest_frozen_at": manifest["frozen_at"],
                            "n_rows": int(len(test))}) + "\n")
    return test.set_index("id")["y"]
