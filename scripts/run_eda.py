"""Stage 3: descriptive exploratory data analysis.

Data-use policy (see docs/03_eda_findings.md):
  * Analyses relating the target to candidate features use the TRAINING split only
    (Jan-Aug), so no feature-design choice can be informed by validation/test outcomes.
  * Full-year views are limited to label-free quantities (volume, category shares, unseen
    categories, missingness) and the per-period delay prevalence already disclosed in Stage 2,
    shown for description of temporal shift only.
  * Nothing here selects features, thresholds, hyperparameters or models.

Outputs:
    reports/eda/eda_results.json
    reports/figures/eda/*.png

Usage:
    .venv/bin/python scripts/run_eda.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flight_delay.analysis.stats import (Z_95, autocorrelation, category_shares,  # noqa: E402
                                         rate_table, total_variation_distance, unseen_categories)
from flight_delay.config import PROJECT_ROOT, load_config  # noqa: E402
from flight_delay.data.load import load_flights  # noqa: E402
from flight_delay.data.split import SPLIT_ORDER  # noqa: E402

OUT_JSON = PROJECT_ROOT / "reports" / "eda" / "eda_results.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"

# Reference palette (dataviz skill, light mode); first three slots validated all-pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_N_STABILITY = 200   # min flights per half-period for entity-rate stability comparisons
SMALL_N = 30            # "small sample" flag for categories

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def records(t: pd.DataFrame, digits: int = 4) -> list[dict]:
    t = t.reset_index()
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(digits)
    return json.loads(t.to_json(orient="records", date_format="iso"))


def save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def rate_panel(ax, t: pd.DataFrame, labels, title: str, overall: float, color=BLUE) -> None:
    """Rate with Wilson 95% CI per category, plus the overall training rate for reference."""
    x = np.arange(len(t))
    ax.errorbar(x, t["rate"], yerr=[t["rate"] - t["ci_low"], t["ci_high"] - t["rate"]],
                fmt="o", ms=4, color=color, ecolor=color, elinewidth=1, capsize=2)
    ax.axhline(overall, color=NEUTRAL, lw=1, ls="--", label=f"Training overall ({overall:.3f})")
    ax.legend(loc="best", frameon=False)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share arriving ≥15 min late")
    ax.set_title(title, loc="left")


def count_panel(ax, t: pd.DataFrame, labels) -> None:
    ax.bar(np.arange(len(t)), t["n"], color=NEUTRAL, width=0.7)
    ax.set_xticks(np.arange(len(t)), labels)
    ax.set_ylabel("Labelled flights (n)")


def main() -> None:
    cfg = load_config()
    df = load_flights(cfg)
    df["dow"] = df["flight_date"].dt.day_of_week
    df["arr_hour"] = df["sched_arr_time"] // 100
    df["route"] = df["origin"] + "-" + df["dest"]
    train = df[df["split"] == "train"]
    lab_train = train[train["y"].notna()]
    p_train = float(lab_train["y"].mean())
    R: dict = {"policy": {
        "target_vs_feature_analyses": "training split only (2013-01-01..2013-08-31)",
        "full_year_analyses": "volume, missingness, category shares, unseen categories, "
                              "per-period prevalence (description of shift only)",
        "train_overall_rate": round(p_train, 4)}}
    figs: dict[str, str] = {}
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Q1 data quality: missingness by split, unlabelled share by month / carrier / origin
    R["q01_quality"] = {
        "missing_by_split": {s: {c: int(v) for c, v in df[df.split == s].isna().sum().items() if v}
                             for s in SPLIT_ORDER},
        "unlabelled_share_by_month": records(
            df.assign(u=df["y"].isna()).groupby("month")["u"].agg(["size", "sum", "mean"])),
        "unlabelled_share_by_carrier": records(
            df.assign(u=df["y"].isna()).groupby("carrier")["u"].agg(["size", "sum", "mean"])
              .sort_values("mean")),
        "unlabelled_share_by_origin": records(
            df.assign(u=df["y"].isna()).groupby("origin")["u"].agg(["size", "sum", "mean"])),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    um = df.assign(u=df["y"].isna()).groupby("month")["u"].mean()
    a1.bar(um.index, 100 * um.values, color=BLUE, width=0.7)
    a1.set_xticks(range(1, 13), [m[0] for m in MONTH_NAMES])
    a1.set_ylabel("Flights without arrival-delay label (%)")
    a1.set_title("(a) By scheduled month, all rows", loc="left")
    uc = df.assign(u=df["y"].isna()).groupby("carrier")["u"].agg(["mean", "size"]).sort_values("mean")
    a2.bar(range(len(uc)), 100 * uc["mean"], color=BLUE, width=0.7)
    a2.set_xticks(range(len(uc)), [f"{c} (n={n:,})" for c, n in zip(uc.index, uc["size"])], rotation=90)
    a2.set_ylabel("Flights without label (%)")
    a2.set_title("(b) By carrier, all rows", loc="left")
    figs["eda_01_unlabelled_share.png"] = save(fig, "eda_01_unlabelled_share.png")

    # Q2 class balance per split
    cb = (df.groupby("split", observed=True)
            .agg(rows=("y", "size"), labelled=("y", "count"),
                 positives=("y", lambda s: int((s == 1).sum())),
                 negatives=("y", lambda s: int((s == 0).sum()))))
    cb["prevalence"] = cb["positives"] / cb["labelled"]
    R["q02_class_balance"] = records(cb)

    # Q3 arrival-delay distribution (train labelled)
    ad = lab_train["arr_delay"]
    R["q03_arr_delay_train"] = {
        "n": int(ad.size), "mean": round(float(ad.mean()), 2), "median": float(ad.median()),
        "quantiles": {str(q): float(ad.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)},
        "share_early_lt0": round(float((ad < 0).mean()), 4),
        "share_ge15": round(float((ad >= 15).mean()), 4),
        "share_ge60": round(float((ad >= 60).mean()), 4),
        "share_ge180": round(float((ad >= 180).mean()), 4),
        "n_ge600": int((ad >= 600).sum()), "min": float(ad.min()), "max": float(ad.max()),
        "skew": round(float(ad.skew()), 2),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a1.hist(ad.clip(-90, 300), bins=np.arange(-90, 305, 5), color=BLUE, edgecolor="white", linewidth=0.3)
    a1.axvline(15, color=INK, lw=1, ls="--")
    a1.annotate("15-min threshold", (17, a1.get_ylim()[1] * 0.9), fontsize=7, color=INK)
    a1.set_xlabel("Arrival delay (minutes; values beyond [−90, 300] pooled in edge bins)")
    a1.set_ylabel("Flights")
    a1.set_title("(a) Training flights, bulk of distribution", loc="left")
    a2.hist(ad, bins=np.arange(-90, 1300, 20), color=BLUE, edgecolor="white", linewidth=0.3)
    a2.set_yscale("log")
    a2.set_xlabel("Arrival delay (minutes; full range)")
    a2.set_ylabel("Flights (log scale)")
    a2.set_title("(b) Full range, log count", loc="left")
    figs["eda_02_arr_delay_distribution.png"] = save(fig, "eda_02_arr_delay_distribution.png")

    # Q4 delay prevalence over time (full year; description of shift only)
    daily = rate_table(df, "flight_date")
    roll = daily["rate"].rolling(7, min_periods=7).mean()  # trailing 7-day mean
    monthly = rate_table(df, "month")
    R["q04_prevalence_over_time"] = {
        "monthly": records(monthly),
        "daily_rate_quantiles": {str(q): round(float(daily["rate"].quantile(q)), 4)
                                 for q in (0, 0.1, 0.5, 0.9, 1)},
        "worst_days_train": records(daily.loc[daily.index <= pd.Timestamp(cfg["split"]["train"]["end"])]
                                    .sort_values("rate", ascending=False).head(10)),
        "days_with_rate_ge_0_5": int((daily["rate"] >= 0.5).sum()),
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.scatter(daily.index, daily["rate"], s=5, color=NEUTRAL, label="Daily share")
    ax.plot(roll.index, roll, color=BLUE, lw=2, label="Trailing 7-day mean")
    for name in ("validation", "test"):
        ax.axvline(pd.Timestamp(cfg["split"][name]["start"]), color=INK, lw=0.8, ls="--")
    ymax = float(daily["rate"].max())
    for name in SPLIT_ORDER:
        s, e = pd.Timestamp(cfg["split"][name]["start"]), pd.Timestamp(cfg["split"][name]["end"])
        p = float(df.loc[(df.split == name), "y"].mean())
        ax.annotate(f"{name}\n{p:.3f}", (s + (e - s) / 2, ymax * 1.02),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_ylabel("Share of labelled flights ≥15 min late")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_title("Daily arrival-delay prevalence, 2013 (all partitions; descriptive only)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    figs["eda_03_daily_prevalence.png"] = save(fig, "eda_03_daily_prevalence.png")

    # Q10 flight volume over time (all scheduled rows, label-free)
    vol = df.groupby("flight_date").size()
    R["q10_volume"] = {
        "daily_by_split": {s: {"min": int(v.min()), "median": float(v.median()), "max": int(v.max())}
                           for s, v in vol.groupby(df.groupby("flight_date")["split"].first()
                                                   .astype(str))},
        "mean_daily_by_dow": {DOW_NAMES[k]: round(float(v), 1)
                              for k, v in vol.groupby(vol.index.day_of_week).mean().items()},
        "lowest_volume_days": {str(d.date()): int(v) for d, v in vol.nsmallest(5).items()},
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(vol.index, vol.values, color=BLUE, lw=1)
    for name in ("validation", "test"):
        ax.axvline(pd.Timestamp(cfg["split"][name]["start"]), color=INK, lw=0.8, ls="--")
    ax.set_ylabel("Scheduled departures per day")
    ax.set_title("Daily scheduled departures from EWR/JFK/LGA, 2013 (all rows)", loc="left")
    figs["eda_04_daily_volume.png"] = save(fig, "eda_04_daily_volume.png")

    # Q5 scheduled departure hour (train)
    by_hour = rate_table(lab_train, "hour")
    by_arr_hour = rate_table(lab_train, "arr_hour")
    R["q05_hour_train"] = {"dep_hour": records(by_hour), "arr_hour": records(by_arr_hour)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    rate_panel(a1, by_hour, by_hour.index, "Delay share by scheduled departure hour (training)", p_train)
    count_panel(a2, by_hour, by_hour.index)
    a2.set_xlabel("Scheduled departure hour (NYC local)")
    figs["eda_05_rate_by_dep_hour.png"] = save(fig, "eda_05_rate_by_dep_hour.png")

    # Q6 day of week and month (train)
    by_dow = rate_table(lab_train, "dow")
    by_month = rate_table(lab_train, "month")
    R["q06_calendar_train"] = {"dow": records(by_dow.rename(index=dict(enumerate(DOW_NAMES)))),
                               "month": records(by_month)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    rate_panel(a1, by_dow, [DOW_NAMES[i] for i in by_dow.index], "(a) Day of week (training)", p_train)
    rate_panel(a2, by_month, [MONTH_NAMES[i - 1] for i in by_month.index],
               "(b) Month (training covers Jan–Aug only)", p_train)
    a2.set_ylabel("")
    figs["eda_06_rate_by_calendar.png"] = save(fig, "eda_06_rate_by_calendar.png")

    # Q7 carrier (train)
    by_car = rate_table(lab_train, "carrier").sort_values("rate")
    R["q07_carrier_train"] = records(by_car)
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    y = np.arange(len(by_car))
    ax.errorbar(by_car["rate"], y, xerr=[by_car["rate"] - by_car["ci_low"], by_car["ci_high"] - by_car["rate"]],
                fmt="o", ms=4, color=BLUE, elinewidth=1, capsize=2)
    ax.axvline(p_train, color=NEUTRAL, lw=1, ls="--")
    ax.set_yticks(y, [f"{c} (n={n:,})" for c, n in zip(by_car.index, by_car["n"])])
    ax.set_xlabel("Share arriving ≥15 min late (95% Wilson CI)")
    ax.set_title("Delay share by carrier (training)", loc="left")
    ax.grid(axis="y", visible=False)
    figs["eda_07_rate_by_carrier.png"] = save(fig, "eda_07_rate_by_carrier.png")

    # Q8/Q9 origin, destination, route (train) with funnel plots
    by_origin = rate_table(lab_train, "origin")
    by_dest = rate_table(lab_train, "dest")
    by_route = rate_table(lab_train, "route")

    def funnel_stats(t: pd.DataFrame) -> dict:
        se = np.sqrt(p_train * (1 - p_train) / t["n"])
        outside = (t["rate"] - p_train).abs() > Z_95 * se
        big = t[t["n"] >= 1000]
        return {"n_groups": int(len(t)), "n_lt_30": int((t["n"] < SMALL_N).sum()),
                "n_lt_100": int((t["n"] < 100).sum()),
                "share_outside_95pct_funnel": round(float(outside.mean()), 3),
                "rate_range_groups_n_ge_1000": [round(float(big["rate"].min()), 4),
                                                round(float(big["rate"].max()), 4)] if len(big) else None,
                "n_groups_n_ge_1000": int(len(big))}

    R["q08_airports_train"] = {"origin": records(by_origin), "dest_summary": funnel_stats(by_dest),
                               "dest": records(by_dest.sort_values("n", ascending=False))}
    R["q09_routes_train"] = {"summary": funnel_stats(by_route),
                             "route_flight_count_quantiles": {str(q): float(by_route["n"].quantile(q))
                                                              for q in (0, 0.1, 0.25, 0.5, 0.75, 1)},
                             "smallest_routes": records(by_route.nsmallest(10, "n"))}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, t, label in ((axes[0], by_dest, "destination"), (axes[1], by_route, "route")):
        nn = np.logspace(0, np.log10(t["n"].max() * 1.2), 200)
        band = Z_95 * np.sqrt(p_train * (1 - p_train) / nn)
        ax.fill_between(nn, np.clip(p_train - band, 0, 1), np.clip(p_train + band, 0, 1),
                        color=GRID, label="95% range if every group had the overall rate")
        ax.scatter(t["n"], t["rate"], s=10, color=BLUE, alpha=0.8, linewidths=0, label=label.capitalize())
        ax.axhline(p_train, color=NEUTRAL, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(f"Labelled training flights per {label} (log scale)")
        ax.set_title(f"({'a' if label == 'destination' else 'b'}) Delay share by {label} (training)", loc="left")
    axes[0].set_ylabel("Share arriving ≥15 min late")
    axes[0].legend(loc="upper right", frameon=False, fontsize=6)
    figs["eda_08_funnel_dest_route.png"] = save(fig, "eda_08_funnel_dest_route.png")

    # Q11 numerical features: distributions and outliers (label-free, train vs all)
    R["q11_numeric"] = {
        "distance_quantiles_train": {str(q): float(train["distance"].quantile(q)) for q in (0, 0.01, 0.5, 0.99, 1)},
        "distance_extremes": records(df.groupby("route")["distance"].first().sort_values()
                                     .iloc[[0, 1, 2, -2, -1]].to_frame()),
        "dep_hour_counts_all": {int(k): int(v) for k, v in df["hour"].value_counts().sort_index().items()},
        "routes_HNL_ANC_rows": int(df["dest"].isin(["HNL", "ANC"]).sum()),
        "numeric_missing_in_permitted_inputs": {c: int(df[c].isna().sum())
                                                for c in ["sched_dep_time", "sched_arr_time", "distance"]},
    }

    # Q12 relationships among candidate inputs and with the target (train)
    lab_train = lab_train.assign(dist_decile=pd.qcut(lab_train["distance"], 10, duplicates="drop"))
    by_dist = rate_table(lab_train, "dist_decile")
    num = lab_train[["hour", "arr_hour", "distance", "dow", "month"]].astype(float)
    R["q12_relationships_train"] = {
        "rate_by_distance_decile": [{"bin": str(i), **r} for i, r in
                                    zip(by_dist.index, records(by_dist.reset_index(drop=True)))],
        "spearman_candidate_numeric": json.loads(num.corr(method="spearman").round(3).to_json()),
        "dest_served_by_n_carriers_quantiles": {
            str(q): float(train.groupby("dest")["carrier"].nunique().quantile(q)) for q in (0, 0.5, 1)},
        "carriers_per_origin": {k: int(v) for k, v in train.groupby("origin")["carrier"].nunique().items()},
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    labels = [f"{int(iv.left)}–{int(iv.right)}" for iv in by_dist.index]
    rate_panel(a1, by_dist, labels, "(a) Distance decile, miles (training)", p_train)
    a1.tick_params(axis="x", rotation=60)
    arr_full = by_arr_hour.astype(float).reindex(range(24))  # hours with no flights stay empty, not shifted
    rate_panel(a2, arr_full, [h if h % 3 == 0 else "" for h in arr_full.index],
               "(b) Scheduled arrival hour, dest-local (training)", p_train)
    a2.set_ylabel("")
    figs["eda_09_rate_by_distance_arrhour.png"] = save(fig, "eda_09_rate_by_distance_arrhour.png")

    # Q13 temporal distribution shift
    shift = {}
    for col in ["carrier", "origin", "dest", "hour", "dow", "route"]:
        sh = category_shares(df[col], df["split"].astype(str))
        shift[col] = {"tvd_train_vs_validation": round(total_variation_distance(sh["train"], sh["validation"]), 4),
                      "tvd_train_vs_test": round(total_variation_distance(sh["train"], sh["test"]), 4)}
    # Within-train stability of entity delay rates: Jan-Apr vs May-Aug
    h1 = lab_train[lab_train["month"] <= 4]
    h2 = lab_train[lab_train["month"] >= 5]
    stability = {}
    for col in ["carrier", "origin", "dest", "route"]:
        a, b = rate_table(h1, col), rate_table(h2, col)
        both = a.add_suffix("_h1").join(b.add_suffix("_h2"), how="inner")
        both = both[(both["n_h1"] >= MIN_N_STABILITY) & (both["n_h2"] >= MIN_N_STABILITY)]
        stability[col] = {"n_entities": int(len(both)),
                          "pearson_rate_h1_vs_h2": round(float(both["rate_h1"].corr(both["rate_h2"])), 3)
                          if len(both) > 2 else None,
                          "spearman_rate_h1_vs_h2": round(float(both["rate_h1"].rank().corr(both["rate_h2"].rank())), 3)
                          if len(both) > 2 else None}
        if col in ("carrier", "dest"):
            stability[col]["_points"] = both[["rate_h1", "rate_h2", "n_h1", "n_h2"]]
    daily_train = rate_table(lab_train, "flight_date")["rate"]
    full_days = pd.date_range(daily_train.index.min(), daily_train.index.max())
    assert len(full_days) == len(daily_train), "training days are not contiguous"
    R["q13_shift"] = {
        "category_share_tvd": shift,
        "entity_rate_stability_train_JanApr_vs_MayAug": {
            k: {kk: vv for kk, vv in v.items() if kk != "_points"} for k, v in stability.items()},
        "min_n_per_half": MIN_N_STABILITY,
        "daily_rate_autocorrelation_train": autocorrelation(daily_train, range(1, 15)),
        "note_autocorrelation": "lag 1 is not usable at prediction time (D10); lags >= 2 are.",
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for col, color, marker in (("carrier", BLUE, "o"), ("dest", ORANGE, "s")):
        pts = stability[col]["_points"]
        a1.scatter(pts["rate_h1"], pts["rate_h2"], s=12, color=color, marker=marker,
                   label=f"{col} (k={len(pts)})", linewidths=0)
    lim = [0, max(a1.get_xlim()[1], a1.get_ylim()[1])]
    a1.plot(lim, lim, color=NEUTRAL, lw=1, ls="--")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("Delay share, Jan–Apr")
    a1.set_ylabel("Delay share, May–Aug")
    a1.set_title(f"(a) Entity delay share, Jan–Apr vs May–Aug\n(entities with n≥{MIN_N_STABILITY} in each half)", loc="left")
    a1.legend(frameon=False, loc="upper left")
    ac = R["q13_shift"]["daily_rate_autocorrelation_train"]
    a2.bar(list(ac.keys()), list(ac.values()),
           color=[NEUTRAL if k == 1 else BLUE for k in ac.keys()], width=0.7)
    a2.axhline(0, color=INK_2, lw=0.8)
    a2.set_xlabel("Lag (days)")
    a2.set_ylabel("Autocorrelation of daily delay share")
    a2.set_title("(b) Autocorrelation of daily delay share\n(training days)", loc="left")
    a2.set_xticks(list(ac.keys()))
    a2.annotate("lag 1 unavailable\nat prediction time", (1, ac[1]), xytext=(4, ac[1]),
                fontsize=6, color=INK_2, va="center", arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.6})
    figs["eda_10_stability_autocorrelation.png"] = save(fig, "eda_10_stability_autocorrelation.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    sh_c = category_shares(df["carrier"], df["split"].astype(str))[list(SPLIT_ORDER)]
    sh_c = sh_c.sort_values("train", ascending=False)
    w = 0.27
    for i, s in enumerate(SPLIT_ORDER):
        a1.bar(np.arange(len(sh_c)) + (i - 1) * w, 100 * sh_c[s], width=w, color=SPLIT_COLORS[s], label=s)
    a1.set_xticks(np.arange(len(sh_c)), sh_c.index, rotation=90)
    a1.set_ylabel("Share of scheduled flights (%)")
    a1.set_title("(a) Carrier mix by partition (all rows)", loc="left")
    a1.legend(frameon=False)
    sh_h = (category_shares(df["hour"], df["split"].astype(str))[list(SPLIT_ORDER)]
            .reindex(range(24), fill_value=0.0))  # hours without flights are true zeros
    for s in SPLIT_ORDER:
        a2.plot(sh_h.index, 100 * sh_h[s], color=SPLIT_COLORS[s], lw=2, label=s)
    a2.set_xlabel("Scheduled departure hour")
    a2.set_ylabel("Share of scheduled flights (%)")
    a2.set_title("(b) Departure-hour mix (colours as in a)", loc="left")
    figs["eda_11_feature_shift.png"] = save(fig, "eda_11_feature_shift.png")

    # Q14 rare and unseen categories (label-free)
    unseen = {}
    for col in ["carrier", "origin", "dest", "route"]:
        tr = train[col]
        counts = tr.value_counts()
        rare = set(counts[counts < SMALL_N].index)
        unseen[col] = {}
        for s in ("validation", "test"):
            other = df.loc[df.split == s, col]
            unseen[col][s] = {**unseen_categories(tr, other),
                              "rows_in_train_rare_lt30": int(other.isin(rare).sum())}
    R["q14_unseen_rare"] = unseen

    R["figures"] = figs
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(R, indent=2, default=str))
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)} and {len(figs)} figures in "
          f"{FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
