"""Stage 2: verify the prediction contract against the real data and the Stage 1 audit.

Checks that the config, feature registry, target, population and split are mutually
consistent with the actual schema, and measures the outcome-availability evidence for the
historical cutoff (decision D10). Writes:
    reports/contract/contract_check.json
    docs/feature_availability.md   (generated from src/flight_delay/contract.py)
Exits with status 1 if any check fails.

Usage:
    .venv/bin/python scripts/check_contract.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from flight_delay.config import PROJECT_ROOT, load_config, resolve_path
from flight_delay.contract import (PROHIBITED_INPUTS, PERMITTED_INPUTS, assert_registry_covers,
                                   availability_table)
from flight_delay.data.load import file_sha256, load_raw, modelling_population, prepare_flights
from flight_delay.data.split import split_summary
from flight_delay.data.timing import landing_lower_bound

AUDIT_JSON = PROJECT_ROOT / "reports" / "audit" / "dataset_audit.json"
OUT_DIR = PROJECT_ROOT / "reports" / "contract"
AVAILABILITY_MD = PROJECT_ROOT / "docs" / "feature_availability.md"
BUFFERS_MIN = (0, 60, 180)  # unrecorded taxi-in / reporting allowance added to the lower bound


def markdown_table(table: pd.DataFrame) -> str:
    """Minimal Markdown renderer (avoids adding `tabulate` as a dependency)."""
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * table.shape[1]]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |"
              for row in table.itertuples(index=False)]
    return "\n".join(lines)


def outcome_availability(df: pd.DataFrame, lags=(1, 2)) -> dict:
    """For each candidate lag L, count labelled flights whose arrival outcome provably was
    not yet known when the first prediction allowed to use it was made (00:00 on date d+L)."""
    lab = df[df["label_status"] == "labelled"]
    land = landing_lower_bound(lab)
    out = {"labelled": int(len(lab)),
           "max_landing_lower_bound_hours_after_flight_date_start":
               round(float(((land - lab["flight_date"]).dt.total_seconds() / 3600).max()), 2)}
    for lag in lags:
        first_use = lab["flight_date"] + pd.Timedelta(days=lag)
        for buf in BUFFERS_MIN:
            late = land + pd.Timedelta(minutes=buf) >= first_use
            key = f"lag_{lag}_buffer_{buf}min"
            out[key] = {"violations": int(late.sum()),
                        "pct_of_labelled": round(100 * float(late.mean()), 3),
                        "positive_share_among_violations":
                            round(float(lab.loc[late, "y"].mean()), 3) if late.any() else None}
    return out


def main() -> int:
    cfg = load_config()
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    raw = load_raw(raw_path, cfg["data"]["raw_sha256"])
    audit = json.loads(AUDIT_JSON.read_text())
    checks: list[dict] = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    try:
        assert_registry_covers(raw.columns)
        check("registry classifies every raw column exactly", True)
    except ValueError as e:
        check("registry classifies every raw column exactly", False, str(e))
    check("no field is both permitted and unavailable",
          not (PERMITTED_INPUTS & PROHIBITED_INPUTS), sorted(PERMITTED_INPUTS & PROHIBITED_INPUTS))
    check("target source column is not a permitted input",
          cfg["target"]["source_column"] not in PERMITTED_INPUTS)

    raw_snapshot = raw.copy(deep=True)
    df = prepare_flights(raw, cfg)
    check("raw frame not modified by preparation", raw.equals(raw_snapshot) and len(raw) == len(df))

    status_counts = df["label_status"].value_counts().to_dict()
    ug = audit["unlabelled_groups"]
    check("label counts match Stage 1 audit",
          status_counts["labelled"] == audit["target_feasibility"]["labelled_rows"]
          and status_counts["no_departure"] == ug["no_departure_time (cancellation proxy)"]
          and status_counts["departed_no_arrival"] == ug["departed_but_no_arrival_time"]
          and status_counts["arrived_no_delay"] == ug["arrival_time_but_no_arr_delay (diversion proxy)"],
          {k: int(v) for k, v in status_counts.items()})
    check("positive count matches Stage 1 audit",
          int((df["y"] == 1).sum()) == audit["target_feasibility"]["positive"], int((df["y"] == 1).sum()))
    check("missing labels never counted as negatives",
          int((df["y"] == 0).sum()) == audit["target_feasibility"]["negative"]
          and int(df["y"].isna().sum()) == audit["unlabelled_groups"]["total_without_arr_delay"])

    data_min, data_max = df["flight_date"].min(), df["flight_date"].max()
    check("split ranges cover exactly the data's date range",
          str(data_min.date()) == cfg["split"]["train"]["start"]
          and str(data_max.date()) == cfg["split"]["test"]["end"],
          [str(data_min.date()), str(data_max.date())])

    summary = split_summary(df["flight_date"], df["split"], df["y"])
    check("every partition has both classes among labelled rows",
          bool(((summary["positives"] > 0) & (summary["negatives"] > 0)).all()))
    check("labelled rows across partitions sum to modelling population",
          int(summary["rows_labelled"].sum()) == len(modelling_population(df)))

    # Row-order independence: shuffled raw rows must give identical per-id results.
    shuffled = prepare_flights(raw.sample(frac=1.0, random_state=cfg["seed"]), cfg)
    cols = ["id", "flight_date", "sched_dep_ts", "label_status", "y", "split"]
    check("preparation is independent of raw row order",
          df[cols].equals(shuffled[cols]))
    check("prepared frame sorted by scheduled departure",
          bool(df["sched_dep_ts"].is_monotonic_increasing))

    exclusions = (df[df["label_status"] != "labelled"]
                  .groupby(["split", "label_status"], observed=False).size()
                  .unstack(fill_value=0))
    availability = outcome_availability(df)
    sha_after = file_sha256(raw_path)
    check("raw file checksum unchanged", sha_before == sha_after == cfg["data"]["raw_sha256"])

    result = {
        "config": cfg,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "split_summary": summary.to_dict(orient="records"),
        "exclusions_by_split": {s: {k: int(v) for k, v in row.items()}
                                for s, row in exclusions.iterrows()},
        "unlabelled_share_by_carrier_pct": (df["y"].isna().groupby(df["carrier"]).mean() * 100)
                                           .round(2).sort_values().to_dict(),
        "outcome_availability_evidence_D10": availability,
        "permitted_inputs": sorted(PERMITTED_INPUTS),
        "prohibited_inputs": sorted(PROHIBITED_INPUTS),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "contract_check.json").write_text(json.dumps(result, indent=2, default=str))

    table = availability_table()
    AVAILABILITY_MD.write_text(
        "# Feature Availability Table\n\n"
        "Generated by `scripts/check_contract.py` from `src/flight_delay/contract.py`; edit the "
        "registry, not this file.\n\n"
        "Prediction timestamp: end of the day before the flight (decision D3). "
        "`history_eligible` outcomes may only enter historical aggregates of earlier flights, "
        "subject to the historical-outcome cutoff (decision D10).\n\n"
        + markdown_table(table) + "\n")

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}"
              + (f" -> {c['detail']}" if c["detail"] is not None else ""))
    print(summary.to_string(index=False))
    print(json.dumps(availability, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
