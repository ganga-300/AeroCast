"""Stage 6: verify the training-only preprocessing pipelines on the real data.

For each model family (Logistic Regression, XGBoost):
  1. Fit on labelled training rows only; fit metadata must match the training partition.
  2. Transform train, validation, and test-period features with labels removed.
  3. Output columns identical in name and order across partitions.
  4. Learned parameters unchanged after transforming validation and test (no refitting).
  5. Missing values: none for Logistic Regression; for XGBoost only in historical-rate
     columns and only on cold-start rows.
  6. history_unavailable equals the cold-start definition in every partition.
Plus: inputs are declared permitted features; test-label lock intact; raw file unchanged.

No model is trained, and no metric is computed. Test-period results are PASS/FAIL only;
reported statistics are training-derived parameters and validation-only category counts.

Output: reports/preprocessing/preprocessing_check.json
Usage:  .venv/bin/python scripts/check_preprocessing.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys

from flight_delay.config import PROJECT_ROOT, load_config, resolve_path
from flight_delay.contract import PROHIBITED_INPUTS
from flight_delay.data.load import file_sha256, load_flights
from flight_delay.experiment import development_partitions, holdout_features_without_labels
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.preprocessing import FAMILIES, INDICATOR, TrainOnlyPreprocessor

OUT_JSON = PROJECT_ROOT / "reports" / "preprocessing" / "preprocessing_check.json"
FINAL_DIR = PROJECT_ROOT / "reports" / "final"


def state_hash(p: TrainOnlyPreprocessor) -> str:
    return hashlib.sha256(pickle.dumps(p.transformer)).hexdigest()


def main() -> int:
    cfg = load_config()
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    table = build_feature_table(load_flights(cfg), cfg)
    specs = feature_specs(cfg)
    train, val = development_partitions(table, cfg)
    test_x = holdout_features_without_labels(table, cfg)
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    cold = {"train": train["hist_all_rate_exp"].isna(), "validation": val["hist_all_rate_exp"].isna(),
            "test": test_x["hist_all_rate_exp"].isna()}
    families = {}
    for family in FAMILIES:
        p = TrainOnlyPreprocessor(family, specs, cfg).fit(train)
        info = p.fit_info_
        check(f"[{family}] fitted on the training partition only",
              info["n_rows"] == len(train) and info["date_min"] == cfg["split"]["train"]["start"]
              and info["date_max"] == cfg["split"]["train"]["end"], info)
        check(f"[{family}] inputs are declared features, none prohibited, route not encoded",
              not set(p.input_columns) & PROHIBITED_INPUTS and "route" not in p.input_columns)
        before = state_hash(p)
        outs = {"train": p.transform(train), "validation": p.transform(val), "test": p.transform(test_x)}
        check(f"[{family}] learned parameters unchanged after transforming validation and test",
              state_hash(p) == before)
        check(f"[{family}] identical output columns across partitions",
              all(list(o.columns) == p.feature_names_out_ for o in outs.values()),
              {"n_output_columns": len(p.feature_names_out_)})
        nan_ok = {}
        for part, o in outs.items():
            rows_with_nan = o.isna().any(axis=1)
            nan_ok[part] = (not rows_with_nan.any()) if family == "logistic_regression" else \
                bool((rows_with_nan.to_numpy() == cold[part].to_numpy()).all())
        check(f"[{family}] missing values only where allowed (LR none; XGBoost cold-start history only)",
              all(nan_ok.values()), {k: ("PASS" if v else "FAIL") for k, v in nan_ok.items()})
        ind_ok = {part: bool(((o[INDICATOR] == 1).to_numpy() == cold[part].to_numpy()).all())
                  for part, o in outs.items()}
        check(f"[{family}] history_unavailable matches cold-start definition in every partition",
              all(ind_ok.values()), {k: ("PASS" if v else "FAIL") for k, v in ind_ok.items()})

        state = p.learned_state()
        inf_col = {c: f"{c}_infrequent_sklearn" for c in ("carrier", "origin", "dest")}
        val_mapping = {}
        for c in ("carrier", "origin", "dest"):
            seen = set(state["one_hot_categories"][c])
            unseen_rows = ~val[c].isin(seen)
            infrequent_rows = val[c].isin(set(state["one_hot_infrequent"][c]))
            val_mapping[c] = {
                "validation_rows_unseen_in_training": int(unseen_rows.sum()),
                "validation_rows_in_training_infrequent_group": int(infrequent_rows.sum()),
                "unseen_mapped_to": inf_col[c] if inf_col[c] in p.feature_names_out_ else "all-zero block",
            }
        families[family] = {
            "fit_info": info, "input_columns": p.input_columns,
            "output_columns": p.feature_names_out_, "training_learned_state": state,
            "validation_category_mapping": val_mapping,
            "cold_start_rows_train": int(cold["train"].sum()),
        }

    check("test-label lock intact (no frozen manifest, no access log)",
          not (FINAL_DIR / "frozen_manifest.json").exists()
          and not (FINAL_DIR / "test_label_access_log.jsonl").exists())
    check("raw file unchanged", sha_before == file_sha256(raw_path) == cfg["data"]["raw_sha256"])

    result = {"checks": checks, "all_passed": all(c["passed"] for c in checks),
              "min_category_frequency": cfg["preprocessing"]["min_category_frequency"],
              "families": families}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
