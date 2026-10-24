"""Generate LaTeX macros, tables and figure copies for the IEEE report from saved artifacts.

Reads (never writes) saved JSON results and the config. No model is trained, no prediction is
regenerated, and test labels are never accessed: test numbers come only from the saved output of
the single frozen evaluation (reports/final/test_evaluation.json), which is hash-checked before
and after.

Outputs (all regenerable; never edit by hand):
    report/generated/numbers.tex   \\newcommand macros for every number used in the text
    report/generated/tab_*.tex     tables
    report/figures/*.png           copies of result figures
Usage:
    .venv/bin/python scripts/make_report_tables.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from flight_delay.config import PROJECT_ROOT, load_config
from flight_delay.reporting.latex import escape, integer, macro, num, pct, signed, table, with_ci

R = PROJECT_ROOT / "reports"
OUT = PROJECT_ROOT / "report" / "generated"
FIG_OUT = PROJECT_ROOT / "report" / "figures"
SOURCES = {
    "audit": R / "audit" / "dataset_audit.json",
    "design": R / "experiment" / "design_check.json",
    "features": R / "features" / "feature_validation.json",
    "stage7": R / "models" / "baseline_selection.json",
    "stage8": R / "calibration" / "stage8_results.json",
    "test": R / "final" / "test_evaluation.json",
    "stage9": R / "error_analysis" / "stage9_results.json",
}
FROZEN = [R / "final" / f for f in ("frozen_manifest.json", "test_evaluation.json",
                                    "test_label_access_log.jsonl", "test_predictions.csv")]
FIGURES = {
    "eda_daily_prevalence.png": R / "figures" / "eda" / "eda_03_daily_prevalence.png",
    "eda_dep_hour.png": R / "figures" / "eda" / "eda_05_rate_by_dep_hour.png",
    "val_reliability.png": R / "figures" / "stage8" / "stage8_reliability.png",
    "val_pr_roc.png": R / "figures" / "stage8" / "stage8_pr_roc_validation.png",
    "test_pr_roc.png": R / "final" / "figures" / "test_pr_roc.png",
    "test_reliability.png": R / "final" / "figures" / "test_reliability.png",
    "stage9_prior_shift.png": R / "figures" / "stage9" / "stage9_distribution_prior_shift.png",
}
LR, XGB = "logistic_regression_sigmoid", "xgboost_isotonic"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def build(src: dict, cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (macros, tables) as LaTeX strings. Pure function of the loaded artifacts."""
    a, d, f, s7, s8, t, s9 = (src[k] for k in ("audit", "design", "features", "stage7", "stage8", "test", "stage9"))
    m: dict[str, str] = {}
    ug, tf = a["unlabelled_groups"], a["target_feasibility"]
    m |= {"DataRows": integer(a["schema"]["n_rows"]), "DataCols": integer(a["schema"]["n_cols"]),
          "DataStart": a["dates"]["min_date"], "DataEnd": a["dates"]["max_date"],
          "DataLabelled": integer(tf["labelled_rows"]), "DataUnlabelled": integer(ug["total_without_arr_delay"]),
          "DataNoDep": integer(ug["no_departure_time (cancellation proxy)"]),
          "DataDepNoArr": integer(ug["departed_but_no_arrival_time"]),
          "DataArrNoDelay": integer(ug["arrival_time_but_no_arr_delay (diversion proxy)"]),
          "DataPositives": integer(tf["positive"]), "DataPrevalence": pct(tf["prevalence"])}
    ps = d["partition_sizes_labelled"]
    tp = t["partition_summary"]
    m |= {"TrainRows": integer(ps["train"]["rows"]), "TrainPos": integer(ps["train"]["positives"]),
          "TrainPrev": num(ps["train"]["prevalence"]),
          "ValRows": integer(ps["validation"]["rows"]), "ValPos": integer(ps["validation"]["positives"]),
          "ValPrev": num(ps["validation"]["prevalence"]),
          "TestRows": integer(tp["labelled_rows"]), "TestPos": integer(tp["positives"]),
          "TestPrev": num(tp["prevalence"])}
    hc = f["history_config"]
    m |= {"NFeatures": integer(f["n_features"]), "NSchedFeatures": integer(len(f["features_by_group"]["schedule"])),
          "NHistFeatures": integer(len(f["features_by_group"]["history"])), "HistLag": integer(f["history_lag_days"]),
          "HistMinCount": integer(hc["min_count"]), "HistSmoothing": integer(hc["smoothing_m"]),
          "MinPrecision": pct(cfg["operating_point"]["min_precision"], 0),
          "TargetMinutes": integer(cfg["target"]["threshold_minutes"])}

    ci7 = s7["validation_bootstrap_ci"]
    pd7 = s7["paired_difference_xgboost_minus_logistic_regression"]["average_precision"]
    m |= {"ValAPLR": num(ci7["logistic_regression"]["average_precision"]["point"]),
          "ValAPLRci": with_ci(*(ci7["logistic_regression"]["average_precision"][k] for k in ("point", "ci95_low", "ci95_high"))),
          "ValROCLR": num(ci7["logistic_regression"]["roc_auc"]["point"]),
          "ValAPXGB": num(ci7["xgboost"]["average_precision"]["point"]),
          "ValROCXGB": num(ci7["xgboost"]["roc_auc"]["point"]),
          "ValDiffAPci": with_ci(pd7["point"], pd7["ci95_low"], pd7["ci95_high"], sign=True),
          "DummyAccuracy": pct(1 - ps["validation"]["prevalence"])}

    c_lr, c_xgb = s8["models"]["logistic_regression"], s8["models"]["xgboost"]
    m |= {"CalSlopeA": num(c_lr["final_calibrator"]["a"], 4), "CalInterceptB": num(c_lr["final_calibrator"]["b"], 4),
          "MaxPrecLR": num(c_lr["threshold_selection"]["max_attainable_precision"]),
          "MaxPrecXGB": num(c_xgb["threshold_selection"]["max_attainable_precision"]),
          "CalSelLR": escape(c_lr["calibration_selection"]["selected"]),
          "CalSelXGB": escape(c_xgb["calibration_selection"]["selected"])}

    tm, pdt = t["models"], t["paired_differences"]

    def tci(model, metric, sign=False):
        x = tm[model]["intervals_95"][metric]
        return with_ci(x["point"], x["ci95_low"], x["ci95_high"], sign=sign)

    def dci(key, metric, digits=3):
        x = pdt[key][metric]
        return with_ci(x["point"], x["ci95_low"], x["ci95_high"], digits, sign=True)

    m |= {"TestAPLR": num(tm[LR]["metrics"]["average_precision"]), "TestAPLRci": tci(LR, "average_precision"),
          "TestROCLR": num(tm[LR]["metrics"]["roc_auc"]), "TestROCLRci": tci(LR, "roc_auc"),
          "TestBrierLR": num(tm[LR]["metrics"]["brier"], 4), "TestLLLR": num(tm[LR]["metrics"]["log_loss"], 4),
          "TestCITLLR": num(tm[LR]["calibration_in_the_large"]),
          "TestSlopeLR": num(tm[LR]["calibration_intercept_slope"]["slope"]),
          "TestMeanPredLR": num(tm[LR]["metrics"]["mean_predicted_probability"]),
          "TestAPXGB": num(tm[XGB]["metrics"]["average_precision"]), "TestROCXGB": num(tm[XGB]["metrics"]["roc_auc"]),
          "TestBrierPrior": num(tm["prior_baseline"]["metrics"]["brier"], 4),
          "TestLLPrior": num(tm["prior_baseline"]["metrics"]["log_loss"], 4),
          "TestAPLift": num(tm[LR]["metrics"]["average_precision"] / tp["prevalence"], 2),
          "DiffAPLRXGB": dci(f"{LR}_minus_{XGB}", "average_precision"),
          "DiffROCLRXGB": dci(f"{LR}_minus_{XGB}", "roc_auc"),
          "DiffAPLRPrior": dci(f"{LR}_minus_prior_baseline", "average_precision"),
          "DiffBrierLRPrior": dci(f"{LR}_minus_prior_baseline", "brier", 4),
          "DiffLLLRPrior": dci(f"{LR}_minus_prior_baseline", "log_loss", 4)}
    bm = t["by_month"]
    m |= {"NovPrev": num(bm["2013-11"][LR]["metrics"]["prevalence"]), "DecPrev": num(bm["2013-12"][LR]["metrics"]["prevalence"]),
          "NovCITLLR": num(bm["2013-11"][LR]["calibration_in_the_large"]),
          "DecCITLLR": num(bm["2013-12"][LR]["calibration_in_the_large"]),
          "NovMeanPredLR": num(bm["2013-11"][LR]["metrics"]["mean_predicted_probability"]),
          "DecMeanPredLR": num(bm["2013-12"][LR]["metrics"]["mean_predicted_probability"])}
    dec = s9["C_prior_shift_consistency"][LR]["2013-12"]
    m |= {"DecPriorShiftAdj": num(dec["mean_after_pure_prior_shift_adjustment"]),
          "DecPriorShift": num(dec["prior_odds_shift_logit"], 2),
          "DecReqShift": num(dec["constant_logit_shift_matching_saved_prevalence"], 2)}

    # ---------------- tables ----------------
    tabs: dict[str, str] = {}
    tabs["tab_dataset"] = table(
        "Dataset summary (supplied file; NYC departures, 2013).", "tab:dataset",
        ["Quantity", "Value"],
        [["Rows $\\times$ columns", f"{m['DataRows']} $\\times$ {m['DataCols']}"],
         ["Date range", f"{m['DataStart']} to {m['DataEnd']}"],
         ["Labelled flights (arrival delay recorded)", m["DataLabelled"]],
         [f"Delayed ($\\geq${m['TargetMinutes']} min) / prevalence", f"{m['DataPositives']} / {m['DataPrevalence']}"],
         ["Excluded: never departed (cancellation proxy)", m["DataNoDep"]],
         ["Excluded: departed, no arrival time", m["DataDepNoArr"]],
         ["Excluded: arrived, no delay recorded (diversion proxy)", m["DataArrNoDelay"]],
         ["Features (schedule / historical)", f"{m['NFeatures']} ({m['NSchedFeatures']} / {m['NHistFeatures']})"]],
        "lr", source=f"{rel(SOURCES['audit'])}, {rel(SOURCES['features'])}")
    sp = cfg["split"]
    tabs["tab_protocol"] = table(
        "Experimental protocol and chronological partitions.", "tab:protocol",
        ["Partition", "Dates", "Labelled", "Delayed", "Prevalence"],
        [["Train", f"{sp['train']['start']} to {sp['train']['end']}", m["TrainRows"], m["TrainPos"], m["TrainPrev"]],
         ["Validation", f"{sp['validation']['start']} to {sp['validation']['end']}", m["ValRows"], m["ValPos"], m["ValPrev"]],
         ["Test (single evaluation)", f"{sp['test']['start']} to {sp['test']['end']}", m["TestRows"], m["TestPos"], m["TestPrev"]]],
        "lcrrr", wide=True,
        note=(f"Prediction at the end of day $D-1$; historical outcomes only from flight dates $\\leq D-{m['HistLag']}$. "
              f"Target: arrival delay $\\geq$ {m['TargetMinutes']} min. Threshold rule: maximise recall subject to "
              f"precision $\\geq$ {m['MinPrecision']} on validation (infeasible; see Sec.~\\ref{{sec:calibration}})."),
        source=f"{rel(SOURCES['design'])}, {rel(SOURCES['test'])}, configs/config.yaml")

    rows7 = []
    for name, label in (("dummy_most_frequent", "Most-frequent dummy"), ("prior_baseline", "Prior (train prevalence)"),
                        ("logistic_regression", "Logistic Regression"), ("xgboost", "XGBoost")):
        ci = ci7[name]
        met = (s7["references"][name]["validation"] if name in s7["references"] else s7["selected"][name]["validation"])
        ap = with_ci(ci["average_precision"]["point"], ci["average_precision"]["ci95_low"], ci["average_precision"]["ci95_high"])
        roc = (with_ci(ci["roc_auc"]["point"], ci["roc_auc"]["ci95_low"], ci["roc_auc"]["ci95_high"])
               if "ci95_low" in ci["roc_auc"] else num(ci["roc_auc"]["point"]))
        rows7.append([label, ap, roc, num(met["brier"], 4), num(met["log_loss"], 3)])
    tabs["tab_validation"] = table(
        "Validation comparison (development estimate; optimistic because validation was used for selection).",
        "tab:validation", ["Model", "AP [95\\% CI]", "ROC AUC [95\\% CI]", "Brier", "Log loss"], rows7, "lcccc",
        wide=True, note=(f"Day-block bootstrap, {integer(ci7['logistic_regression']['average_precision']['n_boot'])} "
                         "resamples. Raw (uncalibrated) probabilities."),
        source=rel(SOURCES["stage7"]))

    rows8 = []
    for fam, label in (("logistic_regression", "LR"), ("xgboost", "XGBoost")):
        cc = s8["models"][fam]["calibration_comparison"]
        sel = s8["models"][fam]["calibration_selection"]["selected"]
        for meth in ("none", "sigmoid", "isotonic"):
            mark = " (selected)" if meth == sel else ""
            rows8.append([label, escape(meth) + mark, num(cc[meth]["mean_heldout_log_loss"], 4),
                          num(cc[meth]["mean_heldout_brier"], 4)])
    tabs["tab_calibration"] = table(
        "Calibration choice by Sep$\\leftrightarrow$Oct cross-fitting (mean held-out values on validation).",
        "tab:calibration", ["Model", "Method", "Log loss", "Brier"], rows8, "llcc",
        note=(f"No threshold reached {m['MinPrecision']} precision on calibrator-unseen validation predictions "
              f"(maximum precision: LR {m['MaxPrecLR']}, XGBoost {m['MaxPrecXGB']})."),
        source=rel(SOURCES["stage8"]))

    rowst = []
    for name, label in ((LR, "LR + sigmoid (primary)"), (XGB, "XGBoost + isotonic"),
                        ("prior_baseline", "Prior baseline"), ("dummy_most_frequent", "Most-frequent dummy")):
        mm = tm[name]
        cis = mm["calibration_intercept_slope"]
        slope = num(cis["slope"]) if "slope" in cis else "undefined"
        rowst.append([label, tci(name, "average_precision"), tci(name, "roc_auc"), tci(name, "brier"),
                      tci(name, "log_loss"), num(mm["calibration_in_the_large"]), slope])
    n_boot = t["bootstrap"]["resamples"]
    n_undef = sum(iv["n_undefined"] for mm in tm.values() for iv in mm["intervals_95"].values())
    m |= {"TestBootResamples": integer(n_boot), "TestBootUndefined": integer(n_undef)}
    tabs["tab_test"] = table(
        "Single pre-registered test evaluation (Nov--Dec 2013).", "tab:test",
        ["Model", "AP", "ROC AUC", "Brier", "Log loss", "CITL", "Slope"], rowst, "lcccccc", wide=True,
        note=(f"95\\% day-block bootstrap intervals ({integer(n_boot)} resamples; undefined resamples across all "
              f"intervals: {integer(n_undef)}). CITL: mean predicted minus observed rate. Threshold-dependent "
              "metrics not applicable: no operating threshold was frozen (D32)."),
        source=rel(SOURCES["test"]))

    rowsp = []
    for key, label in ((f"{LR}_minus_{XGB}", "LR $-$ XGBoost"), (f"{LR}_minus_prior_baseline", "LR $-$ prior")):
        rowsp.append([label, dci(key, "average_precision"), dci(key, "roc_auc"), dci(key, "brier", 4), dci(key, "log_loss", 4)])
    tabs["tab_paired"] = table(
        "Paired test differences (same resampled days).", "tab:paired",
        ["Comparison", "$\\Delta$AP", "$\\Delta$ROC", "$\\Delta$Brier", "$\\Delta$Log loss"], rowsp, "lcccc", wide=True,
        note="AP/ROC $>0$ and Brier/log loss $<0$ favour the primary model.", source=rel(SOURCES["test"]))

    rowsm = []
    for mo in ("2013-11", "2013-12"):
        for name, label in ((LR, "LR + sigmoid"), ("prior_baseline", "Prior")):
            x = bm[mo][name]
            rowsm.append([mo, label, num(x["metrics"]["prevalence"]), num(x["metrics"]["mean_predicted_probability"]),
                          num(x["calibration_in_the_large"]), num(x["metrics"]["average_precision"]),
                          num(x["metrics"]["roc_auc"]), num(x["metrics"]["brier"], 4)])
    tabs["tab_monthly"] = table(
        "Test results by month (descriptive).", "tab:monthly",
        ["Month", "Model", "Prev.", "Mean pred.", "CITL", "AP", "ROC", "Brier"], rowsm, "llcccccc", wide=True,
        source=rel(SOURCES["test"]))

    rows9 = []
    for name, label in ((LR, "LR"), (XGB, "XGBoost")):
        for mo, x in s9["C_prior_shift_consistency"][name].items():
            rows9.append([label, mo, num(x["mean_predicted"]), num(x["mean_after_pure_prior_shift_adjustment"]),
                          num(x["saved_observed_prevalence"]), num(x["prior_odds_shift_logit"], 2),
                          num(x["constant_logit_shift_matching_saved_prevalence"], 2)])
    tabs["tab_priorshift"] = table(
        "Prior-shift consistency check (descriptive; never applied).", "tab:priorshift",
        ["Model", "Month", "Mean pred.", "Adjusted", "Observed", "$\\delta_{\\text{prior}}$", "$\\delta_{\\text{match}}$"],
        rows9, "llccccc", wide=True,
        note="Adjusted: pure prior-shift adjustment from the validation base rate; $\\delta$ in log-odds.",
        source=rel(SOURCES["stage9"]))
    return m, tabs


def main() -> int:
    before = {p: sha(p) for p in FROZEN}
    cfg = load_config()
    src = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    macros, tabs = build(src, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    header = ["% Generated by scripts/make_report_tables.py from saved artifacts. Do not edit by hand.",
              *[f"%   {k}: {rel(p)} (sha256 {sha(p)[:12]})" for k, p in SOURCES.items()]]
    (OUT / "numbers.tex").write_text("\n".join(header + [macro(k, v) for k, v in macros.items()]) + "\n")
    for name, tex in tabs.items():
        (OUT / f"{name}.tex").write_text(tex)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for dst, srcfig in FIGURES.items():
        shutil.copyfile(srcfig, FIG_OUT / dst)
    after = {p: sha(p) for p in FROZEN}
    if before != after:
        print("FAIL: frozen evaluation artifacts changed")
        return 1
    print(f"Wrote {len(macros)} macros, {len(tabs)} tables, {len(FIGURES)} figures; frozen artifacts unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
