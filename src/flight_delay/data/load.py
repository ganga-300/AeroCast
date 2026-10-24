"""Read the raw CSV (read-only) and assemble the audited flight table."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.config import resolve_path
from flight_delay.data.split import assign_split, parse_boundaries
from flight_delay.data.target import LABELLED, build_target, label_status
from flight_delay.data.timing import flight_date, scheduled_departure_ts

EXPECTED_COLUMNS = (
    "id", "year", "month", "day", "dep_time", "sched_dep_time", "dep_delay",
    "arr_time", "sched_arr_time", "arr_delay", "carrier", "flight", "tailnum",
    "origin", "dest", "air_time", "distance", "hour", "minute", "time_hour", "name",
)
STRING_COLUMNS = ("carrier", "tailnum", "origin", "dest", "time_hour", "name")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Load flights.csv after verifying its checksum and exact column set."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}; see data/README.md")
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}")
    df = pd.read_csv(path, dtype={c: "string" for c in STRING_COLUMNS})
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or unexpected:
        raise ValueError(f"Schema mismatch. Missing: {missing}; unexpected: {unexpected}")
    non_numeric = [c for c in EXPECTED_COLUMNS
                   if c not in STRING_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns are not numeric: {non_numeric}")


def prepare_flights(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add flight date, scheduled timestamp, label status, target and split.

    Returns a new frame (the input is not modified), sorted by scheduled departure time and
    then `id` so downstream code never depends on the raw file's row order. Unlabelled rows
    are kept, with y = <NA>, so exclusions remain countable.
    """
    df = raw.copy()
    df["flight_date"] = flight_date(df)
    df["sched_dep_ts"] = scheduled_departure_ts(df)
    df["label_status"] = label_status(df)
    df["y"] = build_target(df[cfg["target"]["source_column"]], cfg["target"]["threshold_minutes"])
    df["split"] = assign_split(df["flight_date"], parse_boundaries(cfg["split"]))
    df = df.sort_values(["sched_dep_ts", "id"], kind="mergesort").reset_index(drop=True)
    if (df["y"].isna() != (df["label_status"] != LABELLED)).any():
        raise AssertionError("Missing labels and label_status disagree")
    return df


def load_flights(cfg: dict[str, Any]) -> pd.DataFrame:
    """Checksum-verified raw load followed by `prepare_flights`."""
    raw = load_raw(resolve_path(cfg["data"]["raw_path"]), cfg["data"]["raw_sha256"])
    return prepare_flights(raw, cfg)


def modelling_population(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for supervised training and evaluation (decision D4)."""
    out = df[df["label_status"] == LABELLED]
    if out["y"].isna().any():
        raise AssertionError("Modelling population contains missing labels")
    return out
