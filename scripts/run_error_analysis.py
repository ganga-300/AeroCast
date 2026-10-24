"""Stage 9: descriptive error analysis WITHOUT another test-label release.

Inputs (read-only):
  * reports/final/test_evaluation.json  - saved aggregate test outcomes from the single evaluation
  * reports/final/test_predictions.csv  - saved test predictions (no labels)
  * reports/models/validation_predictions.csv - development (validation) predictions and labels
  * raw schedule columns only, read with `usecols` that exclude every outcome column
  * frozen model/calibrator artifacts (coefficients, importances, calibrator transform)
Never touched: test labels, unlock_test_labels, test-period historical-rate features (they are
lagged summaries of test outcomes). Frozen outputs are hash-checked before and after.

Output: reports/error_analysis/stage9_results.json, reports/figures/stage9/*.png
Usage:  .venv/bin/python scripts/run_error_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flight_delay.analysis.error_analysis import (LABEL_FREE_RAW_COLUMNS,  # noqa: E402
                                                  assert_label_free_columns, group_mean_predictions,
                                                  prediction_summary, prior_odds_shift,
                                                  prior_shift_adjust, reliability_gaps,
                                                  required_logit_shift)
from flight_delay.analysis.stats import category_shares, total_variation_distance  # noqa: E402
from flight_delay.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

FINAL = PROJECT_ROOT / "reports" / "final"
FROZEN_FILES = ["frozen_manifest.json", "test_evaluation.json", "test_label_access_log.jsonl",
                "test_predictions.csv"]
OUT = PROJECT_ROOT / "reports" / "error_analysis" / "stage9_results.json"
FIG = PROJECT_ROOT / "reports" / "figures" / "stage9"
LR, XGB = "logistic_regression_sigmoid", "xgboost_isotonic"
PERIOD_COLORS = {"validation": "#2a78d6", "2013-11": "#eb6834", "2013-12": "#1baf7a"}
INK_2, GRID, NEUTRAL = "#52514e", "#e4e3df", "#9b9a95"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    before = {f: sha(FINAL / f) for f in FROZEN_FILES}
    cfg = load_config()
    saved = json.loads((FINAL / "test_evaluation.json").read_text())
    test = pd.read_csv(FINAL / "test_predictions.csv", parse_dates=["flight_date"])
    val = pd.read_csv(PROJECT_ROOT / "reports" / "models" / "validation_predictions.csv", parse_dates=["flight_date"])
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Label-free schedule columns only.
    assert_label_free_columns(LABEL_FREE_RAW_COLUMNS)
    raw_path = resolve_path(cfg["data"]["raw_path"])
    check("raw file matches frozen checksum", sha(raw_path) == cfg["data"]["raw_sha256"])
    sched = pd.read_csv(raw_path, usecols=list(LABEL_FREE_RAW_COLUMNS))
    check("only label-free raw columns were read", set(sched.columns) == set(LABEL_FREE_RAW_COLUMNS),
          sorted(sched.columns))
    sched["flight_date"] = pd.to_datetime(sched[["year", "month", "day"]])
    sched["dep_hour"] = sched["sched_dep_time"] // 100
    sched["day_of_week"] = sched["flight_date"].dt.day_of_week
    sched["route"] = sched["origin"] + "-" + sched["dest"]
    cols = ["id", "carrier", "origin", "dest", "route", "dep_hour", "day_of_week", "distance"]
    test = test.merge(sched[cols], on="id", how="left", validate="one_to_one")
    val = val.merge(sched[cols], on="id", how="left", validate="one_to_one")
    test["period"] = test["flight_date"].dt.strftime("%Y-%m")
    val["period"] = "validation"
    check("all prediction rows matched to schedule rows", test["carrier"].notna().all() and val["carrier"].notna().all())

    # Validation calibrated predictions via the frozen final calibrators (transform only).
    for name, fam, method in ((LR, "logistic_regression", "sigmoid"), (XGB, "xgboost", "isotonic")):
        cal = joblib.load(PROJECT_ROOT / "models" / f"stage8_{fam}_calibrator_{method}.joblib")
        val[name] = cal.transform(val[f"p_{fam}"].to_numpy())
    val["logistic_regression_raw"], val["xgboost_raw"] = val["p_logistic_regression"], val["p_xgboost"]

    # --- A. prediction distributions and anomalies (label-free) ---
    model_cols = [LR, XGB, "logistic_regression_raw", "xgboost_raw"]
    dist = {c: {"validation": prediction_summary(val[c]),
                **{m: prediction_summary(g[c]) for m, g in test.groupby("period")}} for c in model_cols}
    anomalies = {
        "test_rows": int(len(test)), "duplicate_ids": int(test["id"].duplicated().sum()),
        "non_finite_or_out_of_range": int(sum((~np.isfinite(test[c]) | (test[c] < 0) | (test[c] > 1)).sum()
                                              for c in model_cols)),
        "distinct_values": {c: int(test[c].nunique()) for c in model_cols},
        "prior_baseline_constant": float(test["prior_baseline"].iloc[0]) if test["prior_baseline"].nunique() == 1 else None,
        "test_prediction_columns_contain_labels": "y" in test.columns,
    }
    check("no prediction anomalies (finite, in [0,1], unique ids, no label column)",
          anomalies["duplicate_ids"] == 0 and anomalies["non_finite_or_out_of_range"] == 0
          and not anomalies["test_prediction_columns_contain_labels"])

    # --- B. saved aggregate outcomes ---
    by_month = {m: {name: {"prevalence": v[name]["metrics"]["prevalence"],
                           "mean_predicted": v[name]["metrics"]["mean_predicted_probability"],
                           "calibration_in_the_large": v[name]["calibration_in_the_large"],
                           "average_precision": v[name]["metrics"]["average_precision"],
                           "roc_auc": v[name]["metrics"]["roc_auc"],
                           "brier": v[name]["metrics"]["brier"], "log_loss": v[name]["metrics"]["log_loss"]}
                    for name in (LR, XGB, "prior_baseline", "logistic_regression_raw", "xgboost_raw")}
                for m, v in saved["by_month"].items()}
    rel = {name: reliability_gaps(saved["models"][name]["reliability"]) for name in (LR, XGB)}

    # --- C. prior-shift consistency check (descriptive; uses saved monthly prevalence only) ---
    pi_val = float(val["y"].mean())  # base rate on which the calibrators were fitted
    shift = {}
    for name in (LR, XGB):
        shift[name] = {}
        for m, g in test.groupby("period"):
            pi_m = by_month[m][name]["prevalence"]
            adj = prior_shift_adjust(g[name].to_numpy(), pi_val, pi_m)
            shift[name][m] = {
                "calibrator_base_rate_validation": pi_val, "saved_observed_prevalence": pi_m,
                "mean_predicted": float(g[name].mean()),
                "mean_after_pure_prior_shift_adjustment": float(adj.mean()),
                "prior_odds_shift_logit": prior_odds_shift(pi_val, pi_m),
                "constant_logit_shift_matching_saved_prevalence": required_logit_shift(g[name].to_numpy(), pi_m),
            }

    # --- D. label-free schedule-mix shift ---
    both = pd.concat([val[["period", "flight_date", *cols]], test[["period", "flight_date", *cols]]], ignore_index=True)
    mix = {}
    for col in ("carrier", "origin", "dest", "route", "dep_hour", "day_of_week"):
        sh = category_shares(both[col], both["period"])
        mix[col] = {"tvd_validation_vs_nov": total_variation_distance(sh["validation"], sh["2013-11"]),
                    "tvd_validation_vs_dec": total_variation_distance(sh["validation"], sh["2013-12"]),
                    "tvd_nov_vs_dec": total_variation_distance(sh["2013-11"], sh["2013-12"])}
    mix = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in mix.items()}
    mean_distance = both.groupby("period")["distance"].mean().round(1).to_dict()
    flights_per_day = both.groupby("period")["flight_date"].agg(lambda s: s.value_counts().mean()).round(1).to_dict()

    # --- E. group mean predictions, Nov vs Dec (label-free) ---
    group_preds = {}
    for col in ("carrier", "origin", "dep_hour"):
        t = group_mean_predictions(test, col, LR, "period", min_n=500)
        group_preds[col] = {str(k): {"nov_mean": round(float(r[("mean", "2013-11")]), 4),
                                     "dec_mean": round(float(r[("mean", "2013-12")]), 4),
                                     "nov_n": int(r[("size", "2013-11")]), "dec_n": int(r[("size", "2013-12")])}
                            for k, r in t.iterrows()}
    ratio_spread = {col: (min(v["dec_mean"] / v["nov_mean"] for v in g.values()),
                          max(v["dec_mean"] / v["nov_mean"] for v in g.values())) for col, g in group_preds.items()}

    # Daily mean predictions (label-free) and holiday window (label-free).
    daily = test.groupby("flight_date")[[LR, "logistic_regression_raw"]].mean()
    dec = test[test["period"] == "2013-12"]
    holiday = {"dec_01_19_mean_pred": float(dec[dec["flight_date"].dt.day <= 19][LR].mean()),
               "dec_20_31_mean_pred": float(dec[dec["flight_date"].dt.day >= 20][LR].mean())}

    # --- F. unseen categories (label-free) ---
    train_sched = sched[(sched["flight_date"] >= cfg["split"]["train"]["start"]) &
                        (sched["flight_date"] <= cfg["split"]["train"]["end"])]
    unseen = {}
    for col in ("dest", "route"):
        mask = ~test[col].isin(set(train_sched[col]))
        unseen[col] = {"rows": int(mask.sum()), "categories": sorted(test.loc[mask, col].unique().tolist()),
                       "mean_pred_unseen": float(test.loc[mask, LR].mean()) if mask.any() else None,
                       "mean_pred_seen": float(test.loc[~mask, LR].mean())}

    # --- G. frozen-model interpretation (no data) ---
    lr_model = joblib.load(PROJECT_ROOT / "models" / "stage8_logistic_regression_model.joblib")
    coefs = pd.Series(lr_model.estimator.coef_[0], index=lr_model.pre.feature_names_out_)
    top_lr = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).round(4).to_dict()
    xgb_model = joblib.load(PROJECT_ROOT / "models" / "stage8_xgboost_model.joblib")
    gain = pd.Series(xgb_model.estimator.get_booster().get_score(importance_type="total_gain"))
    gain.index = [xgb_model.pre.feature_names_out_[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
                  for k in gain.index]
    top_xgb = (gain / gain.sum()).sort_values(ascending=False).head(15).round(4).to_dict()

    after = {f: sha(FINAL / f) for f in FROZEN_FILES}
    check("frozen outputs unchanged (manifest, evaluation JSON, access log, predictions)", before == after, after)
    access = [l for l in (FINAL / "test_label_access_log.jsonl").read_text().splitlines() if l.strip()]
    check("still exactly one logged test-label release", len(access) == 1)

    # --- figures ---
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bins = np.linspace(0, 0.7, 71)
    for period, g in [("validation", val), *test.groupby("period")]:
        a1.hist(g[LR], bins=bins, density=True, histtype="step", lw=1.5, color=PERIOD_COLORS[period], label=period)
    a1.set(xlabel="Calibrated LR probability", ylabel="Density",
           title="(a) Prediction distribution by period (label-free)")
    a1.legend(frameon=False)
    months = list(shift[LR])
    x = np.arange(len(months)); w = 0.26
    for i, (key, lab, col) in enumerate([("mean_predicted", "mean predicted (frozen)", "#2a78d6"),
                                         ("mean_after_pure_prior_shift_adjustment", "after pure prior-shift adjustment", "#eb6834"),
                                         ("saved_observed_prevalence", "observed prevalence (saved)", NEUTRAL)]):
        a2.bar(x + (i - 1) * w, [shift[LR][m][key] for m in months], width=w, color=col, label=lab)
    a2.set_xticks(x, months); a2.set_ylabel("Probability / rate"); a2.set_ylim(0, 0.55)
    a2.set_title("(b) Prior-shift consistency check, LR (descriptive)", loc="left")
    a2.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "stage9_distribution_prior_shift.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    vd = val.groupby("flight_date").agg(pred=(LR, "mean"), obs=("y", "mean"))
    ax.plot(vd.index, vd["obs"], color=NEUTRAL, lw=1, label="validation observed daily rate")
    ax.plot(vd.index, vd["pred"], color="#2a78d6", lw=1.5, label="mean calibrated LR prediction")
    ax.plot(daily.index, daily[LR], color="#2a78d6", lw=1.5)
    for m in months:
        s = pd.Timestamp(f"{m}-01"); e = s + pd.offsets.MonthEnd(0)
        ax.hlines(by_month[m][LR]["prevalence"], s, e, color="#0b0b0b", lw=1.2, ls="--")
    ax.hlines([], 0, 0, color="#0b0b0b", ls="--", label="test monthly prevalence (saved aggregate)")
    ax.axvline(pd.Timestamp("2013-11-01"), color=INK_2, lw=0.8, ls=":")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Share delayed / mean prediction")
    ax.set_title("Daily mean prediction; test outcomes available only as monthly aggregates", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.savefig(FIG / "stage9_daily_predictions.png"); plt.close(fig)

    result = {
        "checks": checks, "all_passed": all(c["passed"] for c in checks),
        "data_access": {"test_labels": "not accessed; only saved aggregates from test_evaluation.json",
                        "raw_columns_read": list(LABEL_FREE_RAW_COLUMNS),
                        "excluded": "test-period historical-rate features (lagged summaries of test outcomes)"},
        "A_prediction_distributions": dist, "A_anomalies": anomalies,
        "B_saved_by_month": by_month, "B_reliability_gaps_overall": rel,
        "B_calibration_intercept_slope": {n: saved["models"][n]["calibration_intercept_slope"] for n in (LR, XGB)},
        "C_prior_shift_consistency": shift,
        "D_schedule_mix_tvd": mix, "D_mean_distance": mean_distance, "D_flights_per_day": flights_per_day,
        "E_group_mean_predictions_lr": group_preds, "E_dec_over_nov_mean_pred_ratio_range": ratio_spread,
        "E_holiday_window_lr": holiday,
        "F_unseen_categories": unseen,
        "G_lr_top_coefficients_standardised_inputs": top_lr, "G_xgb_top_total_gain_share": top_xgb,
        "not_possible_without_label_release": [
            "false positives / false negatives", "performance or calibration by carrier, route, airport, hour",
            "reliability curves per month", "day-level or holiday-window error", "errors for unseen categories",
            "permutation importance on test"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
