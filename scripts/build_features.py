"""Stage 4: build the feature table from raw data and validate it on the real dataset.

Checks (all must pass; exit status 1 otherwise):
  1. Feature specs satisfy the prediction contract; built columns match the specs.
  2. Independent brute-force recomputation of every historical feature for a random sample
     of flights from all partitions (direct date filtering, no cumulative sums).
  3. Perturbing all outcomes on a date d leaves features unchanged for dates < d + lag.
  4. Features for rows up to the end of validation are identical whether or not test-period
     rows are present (no future/test data leaks backwards).
  5. Shuffling raw row order leaves features unchanged.
  6. Missing history occurs only on dates with no eligible past (first `lag` days).
  7. The raw file is unchanged.
Also writes descriptive summaries (coverage, fallbacks, unseen categories, redundancy).

The feature table itself is not saved: model stages rebuild it from raw data, so no cached
artifact is ever treated as evidence.

Outputs:
    reports/features/feature_validation.json
    docs/feature_dictionary.md
Usage:
    .venv/bin/python scripts/build_features.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from flight_delay.config import PROJECT_ROOT, load_config, require_history_lag, resolve_path
from flight_delay.data.load import file_sha256, load_raw, modelling_population, prepare_flights
from flight_delay.features.build import META_COLUMNS, build_feature_table, feature_specs
from flight_delay.features.specs import EXCLUDED_FEATURES

OUT_JSON = PROJECT_ROOT / "reports" / "features" / "feature_validation.json"
DICT_MD = PROJECT_ROOT / "docs" / "feature_dictionary.md"
N_BRUTE_FORCE = 300
PERTURB_DATES = ["2013-03-08", "2013-09-15", "2013-11-20"]  # one per partition
TOL = 1e-12


def brute_force_history(df: pd.DataFrame, row: pd.Series, cfg: dict) -> dict[str, float]:
    """Recompute one flight's historical features by direct filtering."""
    lag = require_history_lag(cfg)
    h = cfg["features"]["history"]
    end = row["flight_date"] - pd.Timedelta(days=lag)
    lab = df[df["y"].notna()]

    def window(frame: pd.DataFrame, w: int | None) -> pd.DataFrame:
        m = frame["flight_date"] <= end
        if w is not None:
            m &= frame["flight_date"] > end - pd.Timedelta(days=w)
        return frame[m]

    def rate(frame):
        return (float((frame["y"] == 1).sum()) / len(frame)) if len(frame) else np.nan

    out, prior = {}, {}
    for w in [None, *h["global_windows_days"], *h["entity_windows_days"]]:
        lab_w = "exp" if w is None else f"{w}d"
        g = window(lab, w)
        prior[lab_w] = rate(g)
        if w is None or w in h["global_windows_days"]:
            out[f"hist_all_rate_{lab_w}"], out[f"hist_all_n_{lab_w}"] = prior[lab_w], len(g)
    w0 = min(h["global_windows_days"])
    s = window(df, w0)
    out[f"hist_all_nodep_share_{w0}d"] = (float((s["label_status"] == "no_departure").sum()) / len(s)
                                         if len(s) else np.nan)
    for e in h["entities"]:
        mask = ((lab["origin"] == row["origin"]) & (lab["dest"] == row["dest"]) if e == "route"
                else lab[e] == row[e])
        for w in [None, *h["entity_windows_days"]]:
            lab_w = "exp" if w is None else f"{w}d"
            g = window(lab[mask], w)
            n, pos = len(g), float((g["y"] == 1).sum())
            p = prior[lab_w]
            out[f"hist_{e}_n_{lab_w}"] = n
            out[f"hist_{e}_rate_{lab_w}"] = p if n < h["min_count"] else (pos + h["smoothing_m"] * p) / (n + h["smoothing_m"])
    return out


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a, b)
        return True
    except AssertionError:
        return False


def main() -> int:
    cfg = load_config()
    lag = require_history_lag(cfg)
    raw_path = resolve_path(cfg["data"]["raw_path"])
    sha_before = file_sha256(raw_path)
    raw = load_raw(raw_path, cfg["data"]["raw_sha256"])
    df = prepare_flights(raw, cfg)
    specs = feature_specs(cfg)
    table = build_feature_table(df, cfg)
    feats = [s.name for s in specs]
    hist_feats = [s.name for s in specs if s.group == "history"]
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("specs satisfy contract and columns match specs",
          list(table.columns) == META_COLUMNS + feats, {"n_features": len(feats)})

    # 2. brute-force recomputation
    sample = df[df["y"].notna()].sample(N_BRUTE_FORCE, random_state=cfg["seed"])
    mismatches = []
    for idx, row in sample.iterrows():
        recomputed = brute_force_history(df, row, cfg)
        missing = set(hist_feats) - recomputed.keys()
        if missing:
            raise AssertionError(f"Brute force does not cover: {sorted(missing)}")
        for k in hist_feats:
            v = recomputed[k]
            got = table.at[idx, k]
            if not ((np.isnan(v) and pd.isna(got)) or abs(float(got) - v) <= TOL):
                mismatches.append({"id": int(row["id"]), "feature": k, "expected": v, "got": float(got)})
    check(f"brute-force recomputation matches for {N_BRUTE_FORCE} sampled flights",
          not mismatches, {"mismatches": mismatches[:10],
                           "sample_by_split": sample["split"].value_counts().to_dict()})

    # 3. perturbation of all outcomes on one date per partition
    by_id = table.set_index("id")
    perturb = {}
    for d in PERTURB_DATES:
        changed = raw.copy()
        m = pd.to_datetime(changed[["year", "month", "day"]]) == pd.Timestamp(d)
        changed.loc[m, ["arr_delay", "dep_delay"]] = 500.0
        t2 = build_feature_table(prepare_flights(changed, cfg), cfg).set_index("id")
        early = by_id["flight_date"] < pd.Timestamp(d) + pd.Timedelta(days=lag)
        unchanged = frames_equal(by_id.loc[early, feats], t2.loc[by_id.index[early], feats])
        changed_later = not frames_equal(by_id.loc[~early, feats], t2.loc[by_id.index[~early], feats])
        perturb[d] = {"rows_perturbed": int(m.sum()), "earlier_unchanged": unchanged,
                      "later_changed": changed_later}
    check("perturbing a date's outcomes changes nothing before date + lag",
          all(v["earlier_unchanged"] and v["later_changed"] for v in perturb.values()), perturb)

    # 4. no backward leakage from test-period rows
    val_end = pd.Timestamp(cfg["split"]["validation"]["end"])
    keep = pd.to_datetime(raw[["year", "month", "day"]]) <= val_end
    t_prefix = build_feature_table(prepare_flights(raw[keep], cfg), cfg).set_index("id")
    check("train/validation features identical with or without test-period rows",
          frames_equal(t_prefix[feats], by_id.loc[t_prefix.index, feats]),
          {"rows_compared": int(len(t_prefix))})

    # 5. row order
    t_shuf = build_feature_table(prepare_flights(raw.sample(frac=1.0, random_state=cfg["seed"]), cfg), cfg)
    check("features independent of raw row order",
          frames_equal(t_shuf.set_index("id").sort_index()[feats], by_id.sort_index()[feats]))

    # 6. missing history only where no eligible past exists
    first_date = df["flight_date"].min()
    no_past = table["flight_date"] < first_date + pd.Timedelta(days=lag)
    has_nan = table[hist_feats].isna().any(axis=1)
    check("missing history occurs exactly on dates with no eligible past",
          bool((has_nan == no_past).all()),
          {"rows_without_history": int(no_past.sum()),
           "dates": sorted(table.loc[no_past, "flight_date"].dt.date.astype(str).unique().tolist())})
    sched_feats = [s.name for s in specs if s.group == "schedule"]
    check("schedule features have no missing values", int(table[sched_feats].isna().sum().sum()) == 0)

    # Label-free guard against hidden time indices: no numeric feature may have its
    # validation/test values lie entirely outside its training range.
    numeric_feats = [s.name for s in specs if s.kind == "numeric"]
    tr = table.loc[table.split == "train", numeric_feats]
    disjoint = {}
    for s_name in ("validation", "test"):
        other = table.loc[table.split == s_name, numeric_feats]
        bad = [c for c in numeric_feats if other[c].min() > tr[c].max() or other[c].max() < tr[c].min()]
        if bad:
            disjoint[s_name] = bad
    check("no numeric feature's validation/test range is disjoint from training", not disjoint, disjoint)

    check("raw file unchanged", sha_before == file_sha256(raw_path) == cfg["data"]["raw_sha256"])

    # --- descriptive summaries (modelling population, train/validation only: test-period
    # feature statistics are withheld until the final evaluation, Stage 5) ---
    pop = modelling_population(table)
    pop = pop[pop["split"] != "test"]
    DEV = ("train", "validation")
    h = cfg["features"]["history"]
    fallback = {}
    for e in h["entities"]:
        for w in [f"{x}d" for x in h["entity_windows_days"]]:  # windows that expose counts
            n_col = f"hist_{e}_n_{w}"
            fallback[n_col] = {s: round(float((pop.loc[pop.split == s, n_col] < h["min_count"]).mean()), 4)
                               for s in DEV}
    train = pop[pop.split == "train"]
    unseen = {c: {s: {"categories": sorted(set(pop.loc[pop.split == s, c]) - set(train[c])),
                      "rows": int((~pop.loc[pop.split == s, c].isin(set(train[c]))).sum())}
                  for s in ("validation",)}
              for c in [s.name for s in specs if s.kind == "categorical"]}
    numeric = [s.name for s in specs if s.kind == "numeric"]
    summary = {s: pop.loc[pop.split == s, numeric].describe(percentiles=[0.5]).T[["min", "50%", "max"]]
                   .round(4).to_dict(orient="index") for s in DEV}
    rho = train[numeric].rank().corr()
    redundant = sorted({tuple(sorted((a, b))) + (round(float(rho.at[a, b]), 3),)
                        for a in numeric for b in numeric if a != b and abs(rho.at[a, b]) >= 0.9},
                       key=lambda t: -abs(t[2]))

    result = {
        "checks": checks, "all_passed": all(c["passed"] for c in checks),
        "history_lag_days": lag, "history_config": h,
        "n_features": len(feats),
        "features_by_group": {g: [s.name for s in specs if s.group == g] for g in ("schedule", "history")},
        "excluded_features": EXCLUDED_FEATURES,
        "modelling_population_rows": {s: int((pop.split == s).sum()) for s in DEV},
        "rows_without_history_in_population": {s: int(pop.loc[pop.split == s, hist_feats].isna().any(axis=1).sum())
                                               for s in DEV},
        "share_below_min_count_fallback": fallback,
        "unseen_categories_vs_train": unseen,
        "numeric_summary_by_split": summary,
        "train_spearman_pairs_abs_ge_0_9": [list(t) for t in redundant],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    lines = ["# Feature Dictionary", "",
             "Generated by `scripts/build_features.py` from `src/flight_delay/features/specs.py`. "
             f"Historical features use only flights with flight date ≤ D−{lag} (decision D10).", "",
             "| feature | kind | group | raw sources | description |", "|---|---|---|---|---|"]
    lines += [f"| `{s.name}` | {s.kind} | {s.group} | {', '.join(s.sources)} | {s.description} |"
              for s in specs]
    lines += ["", "## Excluded", ""] + [f"- `{k}`: {v}" for k, v in EXCLUDED_FEATURES.items()]
    DICT_MD.write_text("\n".join(lines) + "\n")

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
