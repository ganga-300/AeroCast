"""Build and verify the frozen-decisions PROPOSAL for user review (no test access).

This is NOT the file that unlocks test labels. The unlocking manifest
(reports/final/frozen_manifest.json, status "FROZEN", built by experiment.build_manifest) is
created only after explicit user approval. The lock rejects this proposal (status is not
"FROZEN"), and a manifest is accepted only if it binds to this proposal's SHA-256 and to the
hashes of the config, raw data, artifacts, evaluation script and package code.

Verifies, without touching test labels:
  * config, decision register outcomes and Stage 7/8 results agree with the decisions to freeze;
  * saved model and calibrator artifacts are hash-pinned, reproduce the recorded validation
    predictions, and match a fresh deterministic refit from the config;
  * the test-label lock is intact and the raw file unchanged.

Output: reports/freeze/frozen_decisions_proposal.json
Usage:  .venv/bin/python scripts/prepare_frozen_proposal.py
"""

from __future__ import annotations

import ast
import hashlib
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
from flight_delay.experiment import (ACCESS_LOG_REL, D32_STATEMENT, EVALUATION_SCRIPT_REL, MANIFEST_REL,
                                     PROPOSAL_REL, HoldoutLockedError, code_sha256,
                                     development_partitions, sha256_file, unlock_test_labels,
                                     validate_manifest)
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.features.specs import EXCLUDED_FEATURES
from flight_delay.models import FlightDelayModel

OUT = PROJECT_ROOT / PROPOSAL_REL
MANIFEST = PROJECT_ROOT / MANIFEST_REL
ACCESS_LOG = PROJECT_ROOT / ACCESS_LOG_REL
EVAL_SCRIPT = PROJECT_ROOT / EVALUATION_SCRIPT_REL
# Calls the evaluation script must never make (no tuning, calibration fitting or threshold choice).
FORBIDDEN_CALLS = {"select_threshold", "recall_at_precision", "metrics_at_threshold", "threshold_table",
                   "compare_methods", "cross_fitted", "select_method", "Calibrator", "partition",
                   "_labelled_partitions", "build_manifest", "validate_config"}
STAGE7 = PROJECT_ROOT / "reports" / "models" / "baseline_selection.json"
STAGE7_PREDS = PROJECT_ROOT / "reports" / "models" / "validation_predictions.csv"
STAGE8 = PROJECT_ROOT / "reports" / "calibration" / "stage8_results.json"
EXPECTED = {"logistic_regression": "sigmoid", "xgboost": "isotonic"}  # D31, as approved
PROPOSAL_STATUS = ("PROPOSAL - FOR USER REVIEW. NOT THE UNLOCKING MANIFEST. "
                   "Cannot release test labels (status is not 'FROZEN').")
D32_TEXT = D32_STATEMENT

TEST_EVALUATION_PLAN = {
    "implementation": "scripts/evaluate_test.py (hash-pinned in this proposal) using "
                      "flight_delay.evaluation.final.evaluate_predictions; not run before approval",
    "order_of_operations": [
        "1. load and validate the FROZEN manifest (bound to this proposal and all hashes) before loading data",
        "2. load hash-pinned artifacts; a deterministic refit on training rows must reproduce their validation predictions exactly; calibrator parameters must match",
        "3. score test-period features without labels; save predictions (no labels) and their SHA-256",
        "4. release test labels once via unlock_test_labels (logged with manifest hash and predictions hash)",
        "5. compute the registered metrics below and write reports/final/test_evaluation.json and figures",
    ],
    "single_access": "Exactly one release. The lock refuses a second release unless a repeat "
                     "reason is given; a repeat is permitted only to fix a software defect that "
                     "changes no frozen decision, and both outputs must be disclosed.",
    "models_scored": {
        "primary": "logistic_regression_sigmoid: Logistic Regression (C=0.01, L2, lbfgs, unweighted) + sigmoid calibrator fitted on all validation predictions",
        "comparison": "xgboost_isotonic: XGBoost (xgboost_01) + isotonic calibrator fitted on all validation predictions",
        "references": ["prior_baseline: constant training prevalence (P = 0.2637)",
                       "dummy_most_frequent: constant P = 0"],
        "secondary_descriptive": ["logistic_regression_raw", "xgboost_raw (uncalibrated probabilities)"],
    },
    "partition_summary": ["labelled test rows", "positives", "prevalence"],
    "metric_definitions": {
        "average_precision": "step-wise AP (scikit-learn average_precision_score); not trapezoidal",
        "roc_auc": "scikit-learn roc_auc_score",
        "brier": "mean squared error of probabilities",
        "log_loss": "binary cross-entropy with probabilities clipped to [1e-15, 1 - 1e-15]",
        "calibration_in_the_large": "mean(q) - mean(y); positive = over-prediction",
        "calibration_intercept_slope": "unpenalised logistic regression of y on logit(clip(q, 1e-15, 1-1e-15)); "
                                       "slope = coefficient, intercept = constant; ideal (0, 1); reported as "
                                       "'undefined (constant prediction)' for constant predictions; descriptive only, never applied",
        "reliability": "up to 10 quantile bins of q; bins with duplicate edges merged (fewer bins possible for "
                       "step-shaped isotonic outputs); per bin: n, mean q, observed rate",
        "mean_predicted_probability": "mean(q)",
    },
    "per_model_metrics": ["average_precision", "roc_auc", "brier", "log_loss", "mean_predicted_probability",
                          "calibration_in_the_large", "calibration_intercept_slope", "reliability"],
    "secondary_descriptive": ["the same point metrics for raw (uncalibrated) LR and XGBoost",
                              "point metrics and calibration-in-the-large by month (2013-11, 2013-12)",
                              "validation vs test: prevalence and mean predicted probability"],
    "threshold_dependent_metrics": {
        "status": "NOT APPLICABLE (D32)",
        "statement": D32_STATEMENT,
        "not_computed": ["recall at the 70% minimum-precision threshold", "precision", "recall",
                         "F1", "confusion matrix at an operating threshold"],
        "figures": "Precision-recall and ROC curves are shown descriptively with the 0.70 "
                   "precision line; no threshold is read from them or reported as an operating point.",
    },
    "uncertainty": {
        "method": "95% percentile intervals; day-block bootstrap: 200 resamples of whole test flight "
                  "dates with replacement; seed 42 (D28); the same resampled days for every model and metric",
        "undefined_resamples": "resamples in which a metric is undefined are excluded and counted (n_undefined)",
        "interval_models": ["logistic_regression_sigmoid", "xgboost_isotonic", "prior_baseline", "dummy_most_frequent"],
        "interval_metrics": ["average_precision", "roc_auc", "brier", "log_loss"],
        "paired_differences": ["logistic_regression_sigmoid - xgboost_isotonic",
                               "logistic_regression_sigmoid - prior_baseline"],
        "paired_sign_convention": "difference = metric(primary) - metric(other); AP and ROC AUC > 0 favour "
                                  "the primary model; Brier and log loss < 0 favour it",
        "constant_baselines": "computed with the same code; ROC AUC of a constant prediction is 0.5 by "
                              "construction (degenerate interval); AP equals prevalence; calibration slope undefined",
    },
    "error_analysis": "Any later test-set error analysis (plan Stage 9) is exploratory and "
                      "cannot change the frozen pipeline.",
}

PRIOR_TEST_EXPOSURE = [
    "Stage 1 dataset audit: monthly delay prevalence incl. November (0.181) and December (0.336); whole-year delay distribution.",
    "Stage 2 prediction contract: test labelled rows 53,991, positives 13,946, prevalence 0.2583; the D10 positive share (53.8%) computed over all partitions.",
    "Stage 3 EDA: daily and monthly prevalence through December (descriptive figure), label-free category mix and unseen categories.",
    "Stage 4 feature validation (earlier version): test-period feature ranges incl. historical rates built from test-period outcomes, fallback shares, unseen categories.",
    "No test predictions, metrics, calibration results or error analyses have ever been computed or seen.",
    "Mitigation: calibration-method and threshold rules (D11, D7) were fixed before any model existed and applied with validation-only criteria; D17 used label-free evidence only.",
]

STRICT_RULES = [
    "Test results cannot be used to alter the model, its hyperparameters, preprocessing, feature "
    "definitions, historical windows, calibration method or calibrator, threshold rule, operating "
    "point, split or prediction contract.",
    "The test evaluation is reported once, as obtained, including if results are disappointing.",
    "Any analysis or model change after test access is a new, non-final experiment and must be "
    "labelled as such; its results may not be presented as an untouched test evaluation.",
    "No refit on training + validation (D20); the frozen artifacts are the ones scored.",
    "The scored pipeline must be the hash-pinned artifacts listed here; a deterministic refit from "
    "the config must reproduce their validation predictions before scoring.",
]


def sha_text(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> int:
    cfg = load_config()
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    stage7, stage8 = json.loads(STAGE7.read_text()), json.loads(STAGE8.read_text())
    preds7 = pd.read_csv(STAGE7_PREDS).set_index("id")
    table = build_feature_table(load_flights(cfg), cfg)
    specs = feature_specs(cfg)
    train, val = development_partitions(table, cfg)
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    cal = cfg["calibration"]
    check("config: D−2 cutoff, 15-min target, 0.70 precision, confirmed split",
          cfg["prediction"]["history_lag_days"] == 2 and cfg["target"]["threshold_minutes"] == 15
          and cfg["operating_point"]["min_precision"] == 0.70
          and cfg["split"] == {"train": {"start": "2013-01-01", "end": "2013-08-31"},
                               "validation": {"start": "2013-09-01", "end": "2013-10-31"},
                               "test": {"start": "2013-11-01", "end": "2013-12-31"}})
    check("primary = Logistic Regression, comparison = XGBoost (D29)",
          cal["primary_model"] == "logistic_regression" and cal["comparison_model"] == "xgboost")

    models = {}
    for family, method in EXPECTED.items():
        s8 = stage8["models"][family]
        check(f"[{family}] Stage 8 selected calibration = {method} (D31)",
              s8["calibration_selection"]["selected"] == method, s8["calibration_selection"])
        check(f"[{family}] threshold infeasible on validation (D32)",
              s8["threshold_selection"]["feasible"] is False,
              {k: s8["threshold_selection"][k] for k in ("max_attainable_precision", "min_precision")})
        check(f"[{family}] Stage 8 used the Stage 7 selected configuration",
              s8["stage7_config_id"] == stage7["selected"][family]["config_id"]
              and s8["params"] == stage7["selected"][family]["params"])

        model_path = PROJECT_ROOT / "models" / f"stage8_{family}_model.joblib"
        cal_path = PROJECT_ROOT / "models" / f"stage8_{family}_calibrator_{method}.joblib"
        model, calibrator = joblib.load(model_path), joblib.load(cal_path)
        p_loaded = model.predict_proba(val).to_numpy()
        recorded = preds7.loc[val["id"], f"p_{family}"].to_numpy()
        refit = FlightDelayModel(family, s8["params"], specs, cfg, cfg["seed"]).fit(
            train, val if family == "xgboost" else None)
        p_refit = refit.predict_proba(val).to_numpy()
        check(f"[{family}] saved model reproduces recorded validation predictions and a fresh refit",
              np.max(np.abs(p_loaded - recorded)) < 1e-9 and np.array_equal(p_loaded, p_refit),
              {"max_abs_vs_recorded": float(np.max(np.abs(p_loaded - recorded))),
               "identical_to_refit": bool(np.array_equal(p_loaded, p_refit))})
        check(f"[{family}] saved calibrator matches Stage 8 final calibrator",
              calibrator.method == method and calibrator.fit_info_ == s8["final_calibrator"],
              calibrator.fit_info_)
        check(f"[{family}] preprocessing fitted on training rows only",
              model.pre.fit_info_["n_rows"] == len(train)
              and model.pre.fit_info_["date_max"] == cfg["split"]["train"]["end"])
        models[family] = {
            "role": "primary" if family == cal["primary_model"] else "comparison",
            "stage7_config_id": s8["stage7_config_id"],
            "hyperparameters": s8["params"],
            "resolved_params": model.train_info_["resolved_params"],
            "seed": cfg["seed"],
            "training_rows": model.train_info_["train_rows"],
            "training_positives": model.train_info_["train_positives"],
            "preprocessing": {
                "family": model.pre.family, "fit_info": model.pre.fit_info_,
                "min_category_frequency": cfg["preprocessing"]["min_category_frequency"],
                "n_output_columns": len(model.pre.feature_names_out_),
                "output_columns_sha256": sha_text(model.pre.feature_names_out_),
            },
            "calibration": {"method": method, "fitted_on": "all validation predictions, 2013-09-01..2013-10-31",
                            "parameters": calibrator.fit_info_,
                            "selection": s8["calibration_selection"],
                            "mean_heldout_log_loss": {m: v["mean_heldout_log_loss"]
                                                      for m, v in s8["calibration_comparison"].items()}},
            "operating_threshold": {"status": "none", "statement": D32_TEXT,
                                    "validation_evidence": s8["threshold_selection"]},
            "validation_development_metrics": s8["eligible_probability_metrics"],
            "artifacts": {"model": {"path": str(model_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(model_path)},
                          "calibrator": {"path": str(cal_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(cal_path)}},
        }

    proposal = {
        "status": PROPOSAL_STATUS,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "unlocking_manifest": {
            "path": str(MANIFEST_REL), "access_log": str(ACCESS_LOG_REL),
            "exists": MANIFEST.exists(),
            "created_only": "after explicit user approval, via experiment.build_manifest: status 'FROZEN', "
                            "ISO-8601 frozen_at with time zone, non-empty approval_note, this proposal's "
                            "SHA-256, and an exact copy of the bound decision fields",
            "lock_checks": ["status == 'FROZEN'", "required fields present", "frozen_at ISO-8601 with time zone",
                            "approved_proposal_sha256 equals this file on disk",
                            "bound fields equal this proposal exactly",
                            "operating point equals the D32 statement; no model has a threshold",
                            "config, raw data, model and calibrator files match their SHA-256",
                            "evaluation script and src package code match their SHA-256",
                            "a repeat release needs a logged reason",
                            "each release logged with manifest and proposal hashes"],
        },
        "prediction_contract": {
            "prediction_timestamp": "end of the day before the scheduled flight date (D3)",
            "target": "y = 1 if arr_delay >= 15 minutes, 0 if below; flights without a recorded arrival delay excluded (D4, D6)",
            "split": cfg["split"], "history_lag_days": cfg["prediction"]["history_lag_days"],
            "history_rule": "historical features use only flights with flight date <= D-2 (D10)",
            "features": [s.name for s in specs], "excluded_features": EXCLUDED_FEATURES,
            "raw_path": cfg["data"]["raw_path"],
            "feature_config": cfg["features"],
        },
        "primary_model": "logistic_regression", "comparison_model": "xgboost",
        "models": models,
        "calibration_and_threshold_procedure_D11": {
            "methods": cal["methods"], "folds": cal["folds"], "criterion": cal["criterion"],
            "tie_tolerance": cal["tie_tolerance"],
            "threshold_rule": "on validation predictions unseen by the calibrator (raw if 'none', otherwise "
                              "cross-fitted by month), choose the threshold with precision >= 0.70 and maximum "
                              "recall; ties to the highest threshold; infeasible if none (D7)",
            "final_calibrator": "selected method refitted on all validation predictions",
            "limitation_D33": "cross-fitting with month-specific maps reorders flights between months; on raw LR "
                              "scores 70% precision occurs only at the top 7 validation flights (5 delayed), in 60% "
                              "of day resamples; D11 kept unchanged",
        },
        "operating_point_D32": D32_TEXT,
        "evaluation_script": {"path": str(EVALUATION_SCRIPT_REL), "sha256": sha256_file(EVAL_SCRIPT)},
        "code_sha256": code_sha256(),
        "test_evaluation_plan": TEST_EVALUATION_PLAN,
        "prior_test_period_exposure": PRIOR_TEST_EXPOSURE,
        "strict_rules": STRICT_RULES,
        "reproducibility": {
            "python": platform.python_version(),
            "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
            "system_dependency": "libomp (Homebrew) for XGBoost on macOS",
            "raw_sha256": sha_before,
            "config_sha256": file_sha256(PROJECT_ROOT / "configs" / "config.yaml"),
            "train_ids_sha256": stage7["reproducibility"]["train_ids_sha256"],
            "validation_ids_sha256": stage7["reproducibility"]["validation_ids_sha256"],
            "stage7_selection_sha256": file_sha256(STAGE7),
            "stage8_results_sha256": file_sha256(STAGE8),
        },
    }

    try:
        validate_manifest(proposal)
        check("this proposal is rejected by the lock if used as a manifest", False)
    except HoldoutLockedError as e:
        check("this proposal is rejected by the lock if used as a manifest", True, str(e))
    if MANIFEST.exists():
        check("no unlocking manifest exists", False, str(MANIFEST_REL))
    else:
        try:
            unlock_test_labels(table, cfg, "proposal-stage lock check")
            check("official lock refuses release without a manifest", False)
        except HoldoutLockedError:
            check("official lock refuses release without a manifest", True)
    check("no test-label access logged", not ACCESS_LOG.exists())

    # Static audit of the evaluation script (compiled and parsed, never executed).
    source = EVAL_SCRIPT.read_text()
    compile(source, str(EVAL_SCRIPT), "exec")
    tree = ast.parse(source)
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    fit_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in ("fit", "fit_transform")]
    order = [name for name in ("load_and_validate_manifest", "load_flights", "holdout_features_without_labels",
                               "unlock_test_labels", "evaluate_predictions")
             if name in source]
    positions = [source.index(f"{name}(", source.index("def main")) for name in order]
    check("evaluation script: no forbidden tuning/calibration/threshold calls",
          not (called & FORBIDDEN_CALLS), sorted(called & FORBIDDEN_CALLS))
    check("evaluation script: exactly one .fit call (the training-only refit check)", len(fit_calls) == 1,
          {"fit_calls": len(fit_calls)})
    check("evaluation script: manifest validated before data load; predictions saved before labels released",
          order == ["load_and_validate_manifest", "load_flights", "holdout_features_without_labels",
                    "unlock_test_labels", "evaluate_predictions"] and positions == sorted(positions)
          and source.index("test_predictions.csv") < source.index("unlock_test_labels(", source.index("def main")))
    check("raw file unchanged", sha_before == file_sha256(raw_path) == cfg["data"]["raw_sha256"])

    proposal["checks"] = checks
    proposal["all_checks_passed"] = all(c["passed"] for c in checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(proposal, indent=2, default=str))
    print(f"proposal SHA-256: {file_sha256(OUT)}")
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if proposal["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
