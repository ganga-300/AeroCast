"""Final, single test evaluation of the frozen pipeline (pre-registered; D32, D34).

DO NOT RUN until the user has approved the proposal and the FROZEN manifest exists at
reports/final/frozen_manifest.json. Without a valid manifest this script stops at step 1,
before loading any data.

Steps:
  1. Load and validate the FROZEN manifest. It must bind to the approved proposal hash, config,
     raw data, model/calibrator artifacts, this script's own SHA-256 and the src package code.
  2. Integrity: load the hash-pinned artifacts; a deterministic refit from the frozen
     configuration (training rows only) must reproduce their validation predictions exactly;
     calibrator parameters must equal the manifest.
  3. Score test-period features WITHOUT labels through the frozen models and calibrators, and
     save the predictions (no labels) with their SHA-256 BEFORE any label is released.
  4. Release test labels once via unlock_test_labels (logged with the manifest hash).
  5. Compute exactly the registered metrics (flight_delay.evaluation.final) and write results.
Nothing is fitted on test data except the descriptive calibration intercept/slope; no
calibration, threshold or decision is selected or changed (D32: no operating threshold).

Outputs (reports/final/): test_predictions.csv, test_evaluation.json, figures/*.png
Usage (only after approval):
    .venv/bin/python scripts/evaluate_test.py [--repeat-reason "disclosed defect fix ..."]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from flight_delay.config import PROJECT_ROOT, load_config  # noqa: E402
from flight_delay.data.load import load_flights  # noqa: E402
from flight_delay.evaluation.final import evaluate_predictions  # noqa: E402
from flight_delay.experiment import (D32_STATEMENT, development_partitions,  # noqa: E402
                                     holdout_features_without_labels, load_and_validate_manifest,
                                     sha256_file, unlock_test_labels)
from flight_delay.features.build import build_feature_table, feature_specs  # noqa: E402
from flight_delay.models import FlightDelayModel  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "final"
FIG_DIR = OUT_DIR / "figures"
PRIMARY = "logistic_regression_sigmoid"
COMPARISON = "xgboost_isotonic"
REFERENCES = ["prior_baseline", "dummy_most_frequent"]
SECONDARY = ["logistic_regression_raw", "xgboost_raw"]
COLORS = {PRIMARY: "#2a78d6", COMPARISON: "#eb6834", "prior_baseline": "#9b9a95", "dummy_most_frequent": "#52514e"}


class IntegrityError(RuntimeError):
    """Raised (before any label is released) when frozen artifacts fail verification."""


def load_frozen(manifest: dict, family: str):
    m = manifest["models"][family]
    loaded = {}
    for kind in ("model", "calibrator"):
        path = PROJECT_ROOT / m["artifacts"][kind]["path"]
        if sha256_file(path) != m["artifacts"][kind]["sha256"]:
            raise IntegrityError(f"{family} {kind} hash changed after manifest validation")
        loaded[kind] = joblib.load(path)
    if loaded["calibrator"].method != m["calibration"]["method"] or \
            loaded["calibrator"].fit_info_ != m["calibration"]["parameters"]:
        raise IntegrityError(f"{family} calibrator differs from the manifest")
    return loaded["model"], loaded["calibrator"]


def figures(y, preds, result) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "savefig.dpi": 200, "savefig.bbox": "tight"})
    paths = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name in (PRIMARY, COMPARISON):
        pr, rc, _ = precision_recall_curve(y, preds[name])
        a1.plot(rc[:-1], pr[:-1], color=COLORS[name], lw=1.5, label=name)
        fpr, tpr, _ = roc_curve(y, preds[name])
        a2.plot(fpr, tpr, color=COLORS[name], lw=1.5, label=name)
    a1.axhline(0.70, color="#0b0b0b", lw=0.8, ls="--", label="0.70 precision (no threshold frozen; D32)")
    a1.axhline(float(np.mean(y)), color="#9b9a95", lw=0.8, ls=":", label="test prevalence")
    a1.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), title="Precision–recall (test, descriptive)")
    a2.plot([0, 1], [0, 1], color="#9b9a95", lw=0.8, ls="--")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    a1.legend(frameon=False, fontsize=6); a2.legend(frameon=False, fontsize=6, loc="lower right")
    p = FIG_DIR / "test_pr_roc.png"; fig.savefig(p); plt.close(fig); paths.append(str(p.relative_to(PROJECT_ROOT)))
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for name in (PRIMARY, COMPARISON, "prior_baseline"):
        rel = result["models"][name]["reliability"]
        ax.plot([b["mean_predicted"] for b in rel], [b["observed_rate"] for b in rel], marker="o", ms=3,
                color=COLORS[name], label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="#9b9a95", lw=0.8, ls="--")
    ax.set(xlabel="Mean predicted probability (≤10 quantile bins)", ylabel="Observed delay share (test)",
           title="Reliability (test)")
    ax.legend(frameon=False, fontsize=6)
    p = FIG_DIR / "test_reliability.png"; fig.savefig(p); plt.close(fig); paths.append(str(p.relative_to(PROJECT_ROOT)))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat-reason", default=None,
                    help="Only for a disclosed software-defect re-run after a previous release")
    args = ap.parse_args()

    # 1. Manifest first: nothing else happens without a valid FROZEN manifest.
    manifest, manifest_sha = load_and_validate_manifest()
    cfg = load_config()
    seed, n_boot = cfg["seed"], cfg["models"]["bootstrap_resamples"]

    # 2. Integrity of frozen artifacts (training/validation data only).
    table = build_feature_table(load_flights(cfg), cfg)
    specs = feature_specs(cfg)
    train, val = development_partitions(table, cfg)
    frozen = {}
    for family in (manifest["primary_model"], manifest["comparison_model"]):
        model, calibrator = load_frozen(manifest, family)
        refit = FlightDelayModel(family, manifest["models"][family]["hyperparameters"], specs, cfg, seed).fit(
            train, val if family == "xgboost" else None)
        if not np.array_equal(model.predict_proba(val).to_numpy(), refit.predict_proba(val).to_numpy()):
            raise IntegrityError(f"{family}: frozen artifact does not reproduce a deterministic refit")
        frozen[family] = (model, calibrator)

    # 3. Score test features without labels; save predictions before any label is released.
    test_x = holdout_features_without_labels(table, cfg)
    if {"y", "label_status"} & set(test_x.columns):
        raise IntegrityError("Test features unexpectedly contain label columns")
    lr_model, lr_cal = frozen["logistic_regression"]
    xgb_model, xgb_cal = frozen["xgboost"]
    lr_raw, xgb_raw = lr_model.predict_proba(test_x).to_numpy(), xgb_model.predict_proba(test_x).to_numpy()
    preds = {PRIMARY: lr_cal.transform(lr_raw), COMPARISON: xgb_cal.transform(xgb_raw),
             "prior_baseline": np.full(len(test_x), float(train["y"].astype(int).mean())),
             "dummy_most_frequent": np.zeros(len(test_x)),
             "logistic_regression_raw": lr_raw, "xgboost_raw": xgb_raw}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "test_predictions.csv"
    pd.DataFrame({"id": test_x["id"].to_numpy(), "flight_date": test_x["flight_date"].dt.date.astype(str).to_numpy(),
                  **{k: np.round(v, 12) for k, v in preds.items()}}).to_csv(pred_path, index=False)
    pred_sha = sha256_file(pred_path)

    # 4. Single logged release of test labels.
    y_by_id = unlock_test_labels(table, cfg, purpose=f"final test evaluation; predictions sha256={pred_sha}",
                                 repeat_reason=args.repeat_reason)
    y = y_by_id.loc[test_x["id"].to_numpy()]
    if len(y) != len(test_x) or y.isna().any():
        raise IntegrityError("Released labels do not align with scored rows")
    y = y.astype(int).to_numpy()

    # 5. Registered metrics only.
    result = evaluate_predictions(
        y, preds, days=test_x["flight_date"].to_numpy(),
        months=test_x["flight_date"].dt.strftime("%Y-%m").to_numpy(),
        primary=PRIMARY, with_intervals=[PRIMARY, COMPARISON, *REFERENCES], secondary=SECONDARY,
        paired_against=[COMPARISON, "prior_baseline"], n_boot=n_boot, seed=seed)
    result["validation_vs_test"] = {
        family: {"validation_prevalence": manifest["models"][family]["validation_development_metrics"]["prevalence"],
                 "validation_mean_predicted": manifest["models"][family]["validation_development_metrics"]["mean_predicted_probability"],
                 "test_prevalence": float(y.mean()),
                 "test_mean_predicted": float(np.mean(preds[name]))}
        for family, name in (("logistic_regression", PRIMARY), ("xgboost", COMPARISON))}
    result.update({
        "operating_point": D32_STATEMENT,
        "manifest_sha256": manifest_sha,
        "approved_proposal_sha256": manifest["approved_proposal_sha256"],
        "test_predictions_sha256": pred_sha,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Single pre-registered evaluation. Test results may not be used to alter the "
                "model, calibration, threshold, features or any other frozen decision.",
    })
    result["figures"] = figures(y, preds, result)
    (OUT_DIR / "test_evaluation.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {OUT_DIR / 'test_evaluation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
