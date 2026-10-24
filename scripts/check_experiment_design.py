"""Stage 5: verify the chronological experimental design on the real data.

Checks (exit status 1 if any fails):
  1. Partitions are chronological, contiguous in date, disjoint by flight id, and match the
     configured boundaries exactly.
  2. Every partition has at least MIN_CLASS_COUNT positives and negatives. For the test
     partition only PASS/FAIL is reported: test-label statistics stay withheld.
  3. The fitting set is labelled training rows only.
  4. The test lock holds: development data excludes test rows, test labels cannot be
     unlocked without a frozen manifest, and no test-label access has been logged.
Also reports (label-free or train/validation only):
  * where validation flights' history comes from (training vs validation period dates);
  * validation flights affected by the train/validation boundary embargo;
  * validation sub-period (month) sizes, relevant to the calibration protocol.

Output: reports/experiment/design_check.json
Usage:  .venv/bin/python scripts/check_experiment_design.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from flight_delay.config import PROJECT_ROOT, load_config, require_history_lag
from flight_delay.data.load import load_flights
from flight_delay.data.split import parse_boundaries
from flight_delay.experiment import (HoldoutLockedError, assert_fit_rows, development_partitions,
                                     partition, unlock_test_labels)
from flight_delay.features.build import build_feature_table

OUT_JSON = PROJECT_ROOT / "reports" / "experiment" / "design_check.json"
MANIFEST = PROJECT_ROOT / "reports" / "final" / "frozen_manifest.json"
ACCESS_LOG = PROJECT_ROOT / "reports" / "final" / "test_label_access_log.jsonl"
# Floor for metric stability: a precision/recall estimate at a 70%-precision operating point
# needs hundreds of positives. A sufficiency check, not a tuned value.
MIN_CLASS_COUNT = 500


def history_composition(val: pd.DataFrame, train_end: pd.Timestamp, lag: int, window: int) -> dict:
    """Label-free: share of each validation flight's history window that lies in the training
    period (dates <= train_end) vs. the validation period itself."""
    end = val["flight_date"] - pd.Timedelta(days=lag)
    start = end - pd.Timedelta(days=window - 1)
    days_in_train = ((train_end - start).dt.days + 1).clip(lower=0, upper=window)
    share = days_in_train / window
    return {"window_days": window,
            "share_of_window_in_training_period": {
                "min": round(float(share.min()), 3), "median": round(float(share.median()), 3),
                "max": round(float(share.max()), 3)},
            "validation_rows_with_window_fully_in_training": int((share == 1).sum()),
            "validation_rows_with_window_fully_in_validation": int((share == 0).sum())}


def main() -> int:
    cfg = load_config()
    lag = require_history_lag(cfg)
    table = build_feature_table(load_flights(cfg), cfg)
    bounds = parse_boundaries(cfg["split"])
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    parts = partition(table, cfg)  # raises on overlap, gaps or id reuse
    check("partitions chronological, contiguous, disjoint by id", True)
    check("partition date ranges equal configured boundaries",
          all(parts[s]["flight_date"].min() == bounds[s].start and
              parts[s]["flight_date"].max() == bounds[s].end for s in parts),
          {s: [str(bounds[s].start.date()), str(bounds[s].end.date())] for s in parts})

    sizes = {}
    for s in ("train", "validation"):
        y = parts[s]["y"]
        sizes[s] = {"rows": int(len(y)), "days": int(parts[s]["flight_date"].nunique()),
                    "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
                    "prevalence": round(float(y.mean()), 4)}
    y_test = parts["test"]["y"]
    test_ok = bool(min((y_test == 1).sum(), (y_test == 0).sum()) >= MIN_CLASS_COUNT)
    sizes["test"] = {"rows": int(len(y_test)), "days": int(parts["test"]["flight_date"].nunique()),
                     "positives": "withheld", "negatives": "withheld", "prevalence": "withheld"}
    del y_test
    check(f"train and validation each have >= {MIN_CLASS_COUNT} positives and negatives",
          all(min(sizes[s]["positives"], sizes[s]["negatives"]) >= MIN_CLASS_COUNT
              for s in ("train", "validation")))
    check(f"test has >= {MIN_CLASS_COUNT} positives and negatives (counts withheld)", test_ok)

    train, val = development_partitions(table, cfg)
    try:
        assert_fit_rows(train, cfg)
        check("fitting set = labelled training rows only", True, {"rows": int(len(train))})
    except AssertionError as e:
        check("fitting set = labelled training rows only", False, str(e))

    check("development partitions contain no test-period rows",
          not ((train["split"] == "test").any() or (val["split"] == "test").any()))
    if MANIFEST.exists():
        check("no frozen manifest exists yet (decisions not frozen)", False, str(MANIFEST))
    else:
        try:
            unlock_test_labels(table, cfg, MANIFEST, ACCESS_LOG, "stage 5 lock check")
            check("test labels locked without a frozen manifest", False)
        except HoldoutLockedError:
            check("test labels locked without a frozen manifest", True)
    check("no test-label access has been logged", not ACCESS_LOG.exists())

    train_end = bounds["train"].end
    embargo = val["flight_date"] < train_end + pd.Timedelta(days=lag)
    months = {m: val[val["flight_date"].dt.month == m]["y"] for m in (9, 10)}
    result = {
        "checks": checks, "all_passed": all(c["passed"] for c in checks),
        "min_class_count": MIN_CLASS_COUNT,
        "partition_sizes_labelled": sizes,
        "history_sources_for_validation": [
            history_composition(val, train_end, lag, w) for w in cfg["features"]["history"]["entity_windows_days"]],
        "boundary_embargo": {
            "rule": f"a model predicting date D could only have been trained on labels dated <= D-{lag}",
            "affected_validation_dates": sorted(val.loc[embargo, "flight_date"].dt.date.astype(str).unique()),
            "affected_validation_rows": int(embargo.sum()),
            "share_of_validation": round(float(embargo.mean()), 4),
            "test_affected": False,
        },
        "validation_sub_periods": {
            {9: "2013-09", 10: "2013-10"}[m]: {"rows": int(len(y)), "positives": int((y == 1).sum()),
                                              "prevalence": round(float(y.mean()), 4)}
            for m, y in months.items()},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    print(json.dumps({k: result[k] for k in ("partition_sizes_labelled", "history_sources_for_validation",
                                             "boundary_embargo", "validation_sub_periods")}, indent=1))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
