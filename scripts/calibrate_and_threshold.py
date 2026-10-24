"""Stage 8: calibration-method selection and operating-threshold selection on validation (D7, D11).

For the primary model (Logistic Regression, D29) and the comparison model (XGBoost):
  1. Refit the Stage 7 selected configuration on Jan-Aug (no search, no train+validation refit)
     and confirm its validation predictions match the recorded Stage 7 predictions.
  2. Compare none / sigmoid / isotonic by Sep<->Oct cross-fitting; select by mean held-out
     log loss with the simpler-method tie-break.
  3. Choose the 70%-precision threshold on validation predictions the calibrator did not see
     (raw if "none", otherwise cross-fitted), maximising recall; report infeasibility honestly.
  4. Fit the final calibrator on all validation predictions and report the scale mismatch at
     the frozen threshold.
Test data are never scored or summarised. No frozen manifest is written: a proposal awaits
the user's approval.

Outputs:
    reports/calibration/stage8_results.json
    reports/calibration/frozen_decisions_proposal.json
    reports/figures/stage8/*.png
    models/stage8_*.joblib (git-ignored)
Usage:
    .venv/bin/python scripts/calibrate_and_threshold.py
"""

from __future__ import annotations

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

from flight_delay.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from flight_delay.data.load import file_sha256, load_flights  # noqa: E402
from flight_delay.evaluation.calibration import (Calibrator, compare_methods,  # noqa: E402
                                                 cross_fitted, select_method)
from flight_delay.evaluation.metrics import (average_precision, day_block_bootstrap,  # noqa: E402
                                             probability_metrics, roc_auc)
from flight_delay.evaluation.threshold import (metrics_at_threshold, recall_at_precision,  # noqa: E402
                                               select_threshold, threshold_table)
from flight_delay.experiment import development_partitions, partition  # noqa: E402
from flight_delay.features.build import build_feature_table, feature_specs  # noqa: E402
from flight_delay.models import FlightDelayModel  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "calibration"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "stage8"
MODEL_DIR = PROJECT_ROOT / "models"
STAGE7 = PROJECT_ROOT / "reports" / "models"
FINAL_DIR = PROJECT_ROOT / "reports" / "final"
METHOD_COLORS = {"none": "#2a78d6", "sigmoid": "#eb6834", "isotonic": "#1baf7a"}
MODEL_COLORS = {"logistic_regression": "#2a78d6", "xgboost": "#eb6834"}
LABELS = {"logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}
INK_2, GRID, NEUTRAL = "#52514e", "#e4e3df", "#9b9a95"
N_BINS = 10

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def reliability(y, p, n_bins=N_BINS) -> list[dict]:
    """Quantile-binned mean prediction vs observed rate (validation)."""
    bins = pd.qcut(pd.Series(p), n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins.to_numpy()})
    g = df.groupby("bin", observed=True)
    return [{"mean_predicted": float(r.p), "observed_rate": float(r.y), "n": int(n)}
            for (r, n) in zip(g[["y", "p"]].mean().itertuples(), g.size())]


def main() -> int:
    cfg = load_config()
    seed, cal_cfg = cfg["seed"], cfg["calibration"]
    min_precision = cfg["operating_point"]["min_precision"]
    n_boot = cfg["models"]["bootstrap_resamples"]
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    table = build_feature_table(load_flights(cfg), cfg)
    specs = feature_specs(cfg)
    train, val = development_partitions(table, cfg)
    test_ids = set(partition(table, cfg)["test"]["id"])
    y = val["y"].astype(int).to_numpy()
    days = val["flight_date"].to_numpy()
    stage7 = json.loads((STAGE7 / "baseline_selection.json").read_text())
    stage7_preds = pd.read_csv(STAGE7 / "validation_predictions.csv")
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Folds by calendar month (config), covering validation exactly once.
    fold = pd.Series(pd.NA, index=val.index, dtype="object")
    for name, r in cal_cfg["folds"].items():
        fold[val["flight_date"].between(pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))] = name
    check("calibration folds partition the validation set", fold.notna().all(),
          fold.value_counts().to_dict())
    fold = fold.to_numpy()

    results, figs_data, proposal_models = {}, {}, {}
    for role, family in (("primary", cal_cfg["primary_model"]), ("comparison", cal_cfg["comparison_model"])):
        params = stage7["selected"][family]["params"]
        model = FlightDelayModel(family, params, specs, cfg, seed).fit(
            train, val if family == "xgboost" else None)
        p_raw = model.predict_proba(val)
        saved = stage7_preds.set_index("id").loc[val["id"], f"p_{family}"].to_numpy()
        check(f"[{family}] refit reproduces Stage 7 validation predictions",
              np.max(np.abs(p_raw.to_numpy() - saved)) < 1e-9,
              {"max_abs_diff": float(np.max(np.abs(p_raw.to_numpy() - saved)))})
        p = p_raw.to_numpy()

        comparison = compare_methods(p, y, fold, cal_cfg["methods"])
        choice = select_method(comparison, cal_cfg["methods"], cal_cfg["tie_tolerance"])
        method = choice["selected"]
        oof = {m: (p if m == "none" else cross_fitted(p, y, fold, m)) for m in cal_cfg["methods"]}
        eligible = oof[method]

        # Independence check on the real data: a fold's eligible predictions must not change
        # when that fold's labels change.
        if method != "none":
            y_flip = y.copy()
            sep = fold == list(cal_cfg["folds"])[0]
            y_flip[sep] = 1 - y_flip[sep]
            unchanged = np.array_equal(cross_fitted(p, y_flip, fold, method)[sep], eligible[sep])
        else:
            unchanged = True  # raw predictions: the calibrator sees nothing
        check(f"[{family}] threshold predictions are out-of-fold (own-month labels unused)", unchanged)

        thr = select_threshold(y, eligible, min_precision)
        final_cal = Calibrator(method).fit(p, y)
        at_thr = metrics_at_threshold(y, eligible, thr["threshold"]) if thr["feasible"] else None
        mismatch = (metrics_at_threshold(y, final_cal.transform(p), thr["threshold"])
                    if thr["feasible"] else None)

        uncertainty = {
            "recall_at_min_precision_reselected": day_block_bootstrap(
                y, eligible, days, lambda a, b: recall_at_precision(a, b, min_precision), n_boot, seed),
            "average_precision": day_block_bootstrap(y, eligible, days, average_precision, n_boot, seed),
            "roc_auc": day_block_bootstrap(y, eligible, days, roc_auc, n_boot, seed),
        }
        if thr["feasible"]:
            t = thr["threshold"]

            def prec(a, b, t=t):
                m = metrics_at_threshold(a, b, t)
                return np.nan if m["precision"] is None else m["precision"]

            uncertainty["precision_at_frozen_threshold"] = day_block_bootstrap(y, eligible, days, prec, n_boot, seed)
            uncertainty["recall_at_frozen_threshold"] = day_block_bootstrap(
                y, eligible, days, lambda a, b, t=t: metrics_at_threshold(a, b, t)["recall"], n_boot, seed)

        diagnostic = {}
        for m in cal_cfg["methods"]:
            d = select_threshold(y, oof[m], min_precision)
            diagnostic[m] = {"max_attainable_precision": float(threshold_table(y, oof[m])["precision"].max()),
                             "recall_at_min_precision": d["recall"] if d["feasible"] else 0.0,
                             "n_flagged_at_selected": d.get("n_flagged"), "tp_at_selected": d.get("tp")}
        # Stability of the raw-score diagnostic (not used for any decision): share of day
        # resamples in which any raw-score threshold reaches the precision target.
        rng = np.random.default_rng(seed)
        day_codes, uniq = pd.factorize(pd.Series(days))
        rows_by_day = [np.flatnonzero(day_codes == k) for k in range(len(uniq))]
        feasible_draws = 0
        for _ in range(n_boot):
            idx = np.concatenate([rows_by_day[k] for k in rng.integers(0, len(rows_by_day), len(rows_by_day))])
            feasible_draws += select_threshold(y[idx], p[idx], min_precision)["feasible"]
        diagnostic["none"]["share_of_day_resamples_feasible"] = feasible_draws / n_boot
        by_month = {}
        for f in cal_cfg["folds"]:
            m = fold == f
            t_month = select_threshold(y[m], eligible[m], min_precision)
            by_month[f] = {"prevalence": float(y[m].mean()),
                           "max_attainable_precision": float(threshold_table(y[m], eligible[m])["precision"].max()),
                           "recall_at_min_precision": t_month["recall"] if t_month["feasible"] else 0.0,
                           "at_frozen_threshold": (metrics_at_threshold(y[m], eligible[m], thr["threshold"])
                                                   if thr["feasible"] else None)}

        results[family] = {
            "role": role, "stage7_config_id": stage7["selected"][family]["config_id"], "params": params,
            "calibration_comparison": comparison, "calibration_selection": choice,
            "final_calibrator": final_cal.fit_info_,
            "eligible_predictions": "raw model probabilities" if method == "none"
                                    else f"cross-fitted {method} (each month calibrated on the other)",
            "eligible_probability_metrics": probability_metrics(y, eligible),
            "raw_probability_metrics": probability_metrics(y, p),
            "threshold_selection": thr, "metrics_at_threshold_eligible": at_thr,
            "final_calibrator_in_sample_at_threshold": mismatch,
            "uncertainty_day_bootstrap": uncertainty,
            "diagnostic_not_used_for_selection": diagnostic, "by_month": by_month,
            "reliability": {m: reliability(y, oof[m]) for m in cal_cfg["methods"]},
        }
        figs_data[family] = {"oof": oof, "eligible": eligible, "thr": thr}
        MODEL_DIR.mkdir(exist_ok=True)
        cal_path = MODEL_DIR / f"stage8_{family}_calibrator_{method}.joblib"
        mod_path = MODEL_DIR / f"stage8_{family}_model.joblib"
        joblib.dump(final_cal, cal_path)
        joblib.dump(model, mod_path)
        proposal_models[family] = {
            "role": role, "family": family, "stage7_config_id": results[family]["stage7_config_id"],
            "hyperparameters": params, "resolved_params": model.train_info_["resolved_params"],
            "preprocessing": {"family": model.pre.family, "fit_info": model.pre.fit_info_,
                              "min_category_frequency": cfg["preprocessing"]["min_category_frequency"],
                              "output_columns": model.pre.feature_names_out_},
            "calibration": {"method": method, "final_calibrator": final_cal.fit_info_,
                            "fitted_on": "all validation predictions (2013-09-01..10-31)"},
            "threshold": (thr["threshold"] if thr["feasible"] else
                          {"status": "infeasible", "detail": thr}),
            "artifacts": {"model": {"path": str(mod_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(mod_path)},
                          "calibrator": {"path": str(cal_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(cal_path)}},
        }

    check("only validation rows used for calibration and threshold selection",
          not set(val["id"]) & test_ids and len(val) == len(y))
    check("test-label lock intact (no frozen manifest, no access log)",
          not (FINAL_DIR / "frozen_manifest.json").exists()
          and not (FINAL_DIR / "test_label_access_log.jsonl").exists())
    check("raw file unchanged", sha_before == file_sha256(raw_path) == cfg["data"]["raw_sha256"])

    # --- figures (validation only) ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, family in zip(axes, results):
        for m in cal_cfg["methods"]:
            rel = results[family]["reliability"][m]
            ax.plot([r["mean_predicted"] for r in rel], [r["observed_rate"] for r in rel], marker="o", ms=3,
                    lw=1.5, color=METHOD_COLORS[m],
                    label=f"{m} (held-out log loss {results[family]['calibration_comparison'][m]['mean_heldout_log_loss']:.4f})")
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color=NEUTRAL, lw=1, ls="--")
        sel = results[family]["calibration_selection"]["selected"]
        ax.set_title(f"({'a' if family == 'logistic_regression' else 'b'}) {LABELS[family]}: selected {sel}", loc="left")
        ax.set_xlabel("Mean predicted probability (decile bins; out-of-fold)")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Observed delay share (validation)")
    fig.savefig(FIG_DIR / "stage8_reliability.png"); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    prev = float(y.mean())
    for family in results:
        e = figs_data[family]["eligible"]
        pr, rc, _ = precision_recall_curve(y, e)
        pr, rc = pr[:-1], rc[:-1]  # drop scikit-learn's appended (recall 0, precision 1) convention point
        a1.plot(rc, pr, color=MODEL_COLORS[family], lw=1.5,
                label=f"{LABELS[family]} (AP {results[family]['eligible_probability_metrics']['average_precision']:.3f})")
        fpr, tpr, _ = roc_curve(y, e)
        a2.plot(fpr, tpr, color=MODEL_COLORS[family], lw=1.5,
                label=f"{LABELS[family]} (AUC {results[family]['eligible_probability_metrics']['roc_auc']:.3f})")
        t = figs_data[family]["thr"]
        if t["feasible"]:
            a1.plot(t["recall"], t["precision"], marker="o", ms=6, color=MODEL_COLORS[family],
                    markeredgecolor="white")
    a1.axhline(min_precision, color="#0b0b0b", lw=0.8, ls="--", label=f"precision target {min_precision:.2f}")
    a1.axhline(prev, color=NEUTRAL, lw=0.8, ls=":", label=f"prevalence {prev:.3f}")
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision"); a1.set_ylim(0, 1.02)
    a1.set_title("(a) Precision–recall (validation)", loc="left")
    a1.legend(frameon=False, loc="upper right")
    a2.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_title("(b) ROC (validation)", loc="left"); a2.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "stage8_pr_roc_validation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    tt = threshold_table(y, figs_data["logistic_regression"]["eligible"])
    ax.plot(tt["threshold"], tt["precision"], color=MODEL_COLORS["logistic_regression"], lw=1.5, label="Precision")
    ax.plot(tt["threshold"], tt["recall"], color="#1baf7a", lw=1.5, label="Recall")
    ax.axhline(min_precision, color="#0b0b0b", lw=0.8, ls="--", label=f"precision target {min_precision:.2f}")
    t = figs_data["logistic_regression"]["thr"]
    if t["feasible"]:
        ax.axvline(t["threshold"], color=INK_2, lw=0.8, ls=":", label=f"selected threshold {t['threshold']:.3f}")
    ax.set_xlabel("Threshold on threshold-eligible probability (flag if p ≥ threshold)")
    ax.set_ylabel("Value"); ax.set_ylim(0, 1.02)
    ax.set_title("Logistic Regression: precision and recall by threshold (validation)", loc="left")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(FIG_DIR / "stage8_threshold_sweep_lr.png"); plt.close(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"checks": checks, "all_passed": all(c["passed"] for c in checks),
           "min_precision": min_precision, "calibration_config": cal_cfg,
           "validation": {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean())},
           "models": results,
           "note": "Validation-only development results; optimistic because validation was used for "
                   "model selection, early stopping, calibration selection and threshold selection."}
    (OUT_DIR / "stage8_results.json").write_text(json.dumps(out, indent=2, default=str))
    proposal = {
        "status": "PROPOSAL - awaiting user approval; not a frozen manifest; test labels remain locked",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": cal_cfg["primary_model"], "comparison_model": cal_cfg["comparison_model"],
        "features": [s.name for s in specs], "history_lag_days": cfg["prediction"]["history_lag_days"],
        "split": cfg["split"], "operating_point": cfg["operating_point"],
        "models": proposal_models, "config_sha256": file_sha256(PROJECT_ROOT / "configs" / "config.yaml"),
        "raw_sha256": sha_before,
    }
    (OUT_DIR / "frozen_decisions_proposal.json").write_text(json.dumps(proposal, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
