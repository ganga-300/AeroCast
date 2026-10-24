"""Stage 7: train baseline models on Jan-Aug and compare them on Sep-Oct validation.

* Reference models: most-frequent dummy and training-prevalence constant.
* Logistic Regression and XGBoost search spaces from configs/config.yaml (`models`).
* Selection per family: highest validation Average Precision (D21). Exact ties (1e-9) go
  to the simpler configuration, in the config grid's listed order.
* Uncertainty: day-block bootstrap on validation (whole flight dates resampled).

Test data are never loaded with labels, scored or summarised. No threshold, calibration or
threshold-based metric is computed (Stage 8).

Outputs:
    reports/models/baseline_runs.json          every configuration, train + validation metrics
    reports/models/baseline_selection.json     selected configs, CIs, checks, reproducibility
    reports/models/validation_predictions.csv  id, flight_date, y, p_<model> (validation only)
    models/<model>.joblib                       fitted selected models (git-ignored)
Usage:
    .venv/bin/python scripts/train_baselines.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from flight_delay.config import PROJECT_ROOT, load_config, resolve_path
from flight_delay.data.load import file_sha256, load_flights
from flight_delay.evaluation.metrics import (average_precision, day_block_bootstrap,
                                             probability_metrics, roc_auc)
from flight_delay.experiment import development_partitions, partition
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.models import FlightDelayModel, training_scale_pos_weight

OUT_DIR = PROJECT_ROOT / "reports" / "models"
MODEL_DIR = PROJECT_ROOT / "models"
FINAL_DIR = PROJECT_ROOT / "reports" / "final"
TIE_TOL = 1e-9


def lr_grid(cfg):
    g = cfg["models"]["logistic_regression"]
    # Simpler first: stronger regularisation, then no class weighting.
    return [dict(C=c, class_weight=w, max_iter=g["max_iter"])
            for c, w in itertools.product(sorted(g["C"]), g["class_weight"])]


def xgb_grid(cfg):
    g = cfg["models"]["xgboost"]
    fixed = {k: g[k] for k in ("learning_rate", "subsample", "colsample_bytree", "n_estimators_max",
                               "early_stopping_rounds", "early_stopping_metric")}
    # Simpler first: shallower trees, larger min_child_weight, then no class weighting.
    return [dict(max_depth=d, min_child_weight=m, scale_pos_weight=s, **fixed)
            for d, m, s in itertools.product(sorted(g["max_depth"]),
                                             sorted(g["min_child_weight"], reverse=True),
                                             g["scale_pos_weight"])]


def select(runs):
    """Highest validation AP; exact ties resolved by grid order (simpler first)."""
    best = runs[0]
    for r in runs[1:]:
        if r["validation"]["average_precision"] > best["validation"]["average_precision"] + TIE_TOL:
            best = r
    return best


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    table = build_feature_table(load_flights(cfg), cfg)
    specs = feature_specs(cfg)
    train, val = development_partitions(table, cfg)
    test_ids = set(partition(table, cfg)["test"]["id"])  # ids only, to assert they are never used
    y_tr, y_va = train["y"].astype(int), val["y"].astype(int)
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def run(family, params):
        model = FlightDelayModel(family, params, specs, cfg, seed).fit(
            train, val if family == "xgboost" else None)
        p_tr, p_va = model.predict_proba(train), model.predict_proba(val)
        rec = {"family": family, "params": params, "train_info": model.train_info_,
               "train": probability_metrics(y_tr, p_tr), "validation": probability_metrics(y_va, p_va)}
        return rec, model, p_va

    runs, models, preds = [], {}, {}
    for family in ("dummy_most_frequent", "prior_baseline"):
        rec, models[family], preds[family] = run(family, {})
        runs.append(rec)
    by_family = {}
    for family, grid in (("logistic_regression", lr_grid(cfg)), ("xgboost", xgb_grid(cfg))):
        fam_runs = []
        for i, params in enumerate(grid):
            rec, model, p_va = run(family, params)
            rec["config_id"] = f"{family}_{i:02d}"
            fam_runs.append((rec, model, p_va))
            v = rec["validation"]
            print(f"{rec['config_id']}: val AP {v['average_precision']:.4f} ROC {v['roc_auc']:.4f} "
                  f"{json.dumps({k: params[k] for k in params if k in ('C', 'class_weight', 'max_depth', 'min_child_weight', 'scale_pos_weight')})}")
        best = select([r for r, _, _ in fam_runs])
        for rec, model, p_va in fam_runs:
            rec["selected"] = rec is best
            runs.append(rec)
            if rec is best:
                models[family], preds[family] = model, p_va
        by_family[family] = best

    # --- checks ---
    fit_rows = {r["train_info"]["train_rows"] for r in runs}
    check("every model fitted on exactly the labelled training partition",
          fit_rows == {len(train)} and all(r["train_info"]["preprocessing"]["date_max"] == cfg["split"]["train"]["end"]
                                           for r in runs), {"train_rows": len(train)})
    check("no test-period flight used for fitting, early stopping or prediction",
          not (set(train["id"]) | set(val["id"])) & test_ids)
    check("validation predictions aligned to validation rows",
          all(p.index.equals(val.index) for p in preds.values()))
    check("all predicted probabilities finite and within [0, 1]",
          all(np.isfinite(p).all() and p.between(0, 1).all() for p in preds.values()))
    lr_runs = [r for r in runs if r["family"] == "logistic_regression"]
    check("all Logistic Regression fits converged",
          all(r["train_info"]["converged"] for r in lr_runs),
          {r["config_id"]: r["train_info"]["n_iter"] for r in lr_runs})
    spw = training_scale_pos_weight(train["y"])
    balanced = [r for r in runs if r["family"] == "xgboost" and r["params"]["scale_pos_weight"] == "train_balanced"]
    check("XGBoost class weight equals training negatives/positives",
          all(abs(r["train_info"]["resolved_params"]["scale_pos_weight"] - spw) < 1e-12 for r in balanced),
          {"train_neg_over_pos": spw})
    determinism = {}
    for family in ("logistic_regression", "xgboost"):
        again = FlightDelayModel(family, by_family[family]["params"], specs, cfg, seed).fit(
            train, val if family == "xgboost" else None)
        determinism[family] = float(np.max(np.abs(again.predict_proba(val).to_numpy() - preds[family].to_numpy())))
    check("refitting selected models reproduces validation predictions", max(determinism.values()) == 0.0,
          {"max_abs_difference": determinism})
    check("test-label lock intact (no frozen manifest, no access log)",
          not (FINAL_DIR / "frozen_manifest.json").exists()
          and not (FINAL_DIR / "test_label_access_log.jsonl").exists())
    check("raw file unchanged", sha_before == file_sha256(raw_path) == cfg["data"]["raw_sha256"])

    # --- comparison with day-block bootstrap ---
    n_boot = cfg["models"]["bootstrap_resamples"]
    days = val["flight_date"].to_numpy()
    comparison = {}
    for name, p in preds.items():
        comparison[name] = {
            "average_precision": day_block_bootstrap(y_va, p, days, average_precision, n_boot, seed),
            "roc_auc": (day_block_bootstrap(y_va, p, days, roc_auc, n_boot, seed)
                        if p.nunique() > 1 else {"point": 0.5, "note": "constant prediction"}),
        }
    paired = {m: day_block_bootstrap(y_va, preds["xgboost"], days, f, n_boot, seed,
                                     p_other=preds["logistic_regression"])
              for m, f in (("average_precision", average_precision), ("roc_auc", roc_auc))}
    monthly = {}
    for name in ("logistic_regression", "xgboost"):
        monthly[name] = {}
        for month in (9, 10):
            m = (val["flight_date"].dt.month == month).to_numpy()
            monthly[name][f"2013-{month:02d}"] = probability_metrics(y_va[m], preds[name][m])

    imbalance = {}
    for family, key in (("logistic_regression", "class_weight"), ("xgboost", "scale_pos_weight")):
        fam = [r for r in runs if r["family"] == family]
        unweighted = [r for r in fam if r["params"][key] in (None, 1.0)]
        weighted = [r for r in fam if r["params"][key] not in (None, 1.0)]
        imbalance[family] = {
            "best_unweighted_val_ap": max(r["validation"]["average_precision"] for r in unweighted),
            "best_weighted_val_ap": max(r["validation"]["average_precision"] for r in weighted),
            "best_unweighted_val_log_loss": min(r["validation"]["log_loss"] for r in unweighted),
            "best_weighted_val_log_loss": min(r["validation"]["log_loss"] for r in weighted),
        }

    # --- save artefacts ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for name, model in models.items():
        path = MODEL_DIR / f"baseline_{name}.joblib"
        joblib.dump(model, path)
        model_files[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
    vp = val[["id", "flight_date", "y"]].copy()
    for name, p in preds.items():
        vp[f"p_{name}"] = p.round(10)
    vp.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    repro = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "raw_sha256": sha_before,
        "config_sha256": hashlib.sha256((PROJECT_ROOT / "configs" / "config.yaml").read_bytes()).hexdigest(),
        "train_ids_sha256": hashlib.sha256(np.sort(train["id"].to_numpy()).tobytes()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(np.sort(val["id"].to_numpy()).tobytes()).hexdigest(),
        "feature_columns": [s.name for s in specs],
        "model_input_columns": {f: models[f].pre.feature_names_out_ for f in ("logistic_regression", "xgboost")},
    }
    partitions = {"train": {"rows": len(train), "positives": int(y_tr.sum()), "prevalence": float(y_tr.mean()),
                            "dates": [cfg["split"]["train"]["start"], cfg["split"]["train"]["end"]]},
                  "validation": {"rows": len(val), "positives": int(y_va.sum()), "prevalence": float(y_va.mean()),
                                 "dates": [cfg["split"]["validation"]["start"], cfg["split"]["validation"]["end"]]},
                  "test": "not used in Stage 7"}
    (OUT_DIR / "baseline_runs.json").write_text(json.dumps(runs, indent=2, default=str))
    selection = {
        "checks": checks, "all_passed": all(c["passed"] for c in checks),
        "partitions": partitions,
        "selected": {f: {"config_id": by_family[f]["config_id"], "params": by_family[f]["params"],
                         "train_info": by_family[f]["train_info"], "train": by_family[f]["train"],
                         "validation": by_family[f]["validation"]} for f in by_family},
        "references": {f: {"train": r["train"], "validation": r["validation"]}
                       for r in runs for f in [r["family"]] if f in ("dummy_most_frequent", "prior_baseline")},
        "validation_bootstrap_ci": comparison,
        "paired_difference_xgboost_minus_logistic_regression": paired,
        "validation_by_month": monthly,
        "class_imbalance_effect": imbalance,
        "determinism": determinism,
        "model_files": model_files,
        "reproducibility": repro,
        "note": "Validation metrics are development estimates: validation was used for "
                "configuration selection and XGBoost early stopping, so they are optimistic (D21).",
    }
    (OUT_DIR / "baseline_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if selection["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
