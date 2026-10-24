"""Stage 1 dataset audit for the supplied flights.csv.

Read-only: loads the raw CSV, runs descriptive and consistency checks, and
writes the measured findings to reports/audit/dataset_audit.{json,md}.
No rows are dropped or modified; anomalies are counted, not cleaned.

Usage:
    .venv/bin/python scripts/audit_dataset.py [--data data/raw/flights.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "flights.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "audit"

EXPECTED_COLUMNS = [
    "id", "year", "month", "day", "dep_time", "sched_dep_time", "dep_delay",
    "arr_time", "sched_arr_time", "arr_delay", "carrier", "flight", "tailnum",
    "origin", "dest", "air_time", "distance", "hour", "minute", "time_hour", "name",
]
OUTCOME_COLS = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]
DELAY_THRESHOLD_MIN = 15  # proposed target threshold, audited here only


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert HHMM clock values (e.g. 1545, 2400) to minutes after midnight."""
    return (s // 100) * 60 + (s % 100)


def hhmm_validity(s: pd.Series) -> dict:
    v = s.dropna()
    hh, mm = v // 100, v % 100
    return {
        "non_null": int(v.size),
        "minute_part_ge_60": int((mm >= 60).sum()),
        "hour_part_gt_24": int((hh > 24).sum()),
        "value_2400": int((v == 2400).sum()),
        "value_gt_2400": int((v > 2400).sum()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def delay_consistency(actual: pd.Series, sched: pd.Series, delay: pd.Series) -> dict:
    """Check whether recorded delay equals actual - scheduled clock time,
    allowing for a +/- 1 day wrap (flights crossing midnight)."""
    mask = actual.notna() & delay.notna()
    raw = hhmm_to_minutes(actual[mask]) - hhmm_to_minutes(sched[mask])
    d = delay[mask]
    same_day = raw == d
    next_day = (raw + 1440) == d
    prev_day = (raw - 1440) == d
    any_match = same_day | next_day | prev_day
    return {
        "checked": int(mask.sum()),
        "match_same_day": int(same_day.sum()),
        "match_plus_1440": int(next_day.sum()),
        "match_minus_1440": int(prev_day.sum()),
        "no_match": int((~any_match).sum()),
    }


def quantiles(s: pd.Series) -> dict:
    qs = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]
    return {str(q): float(s.quantile(q)) for q in qs}


def audit(path: Path) -> dict:
    r: dict = {}
    r["file"] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

    df = pd.read_csv(path)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    r["schema"] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_expected_columns": missing_cols,
        "unexpected_columns": extra_cols,
        "memory_mb_deep": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }
    if missing_cols:
        raise SystemExit(f"Audit cannot continue, columns missing: {missing_cols}")

    r["missing"] = {
        c: {"count": int(n), "pct": round(100 * n / len(df), 3)}
        for c, n in df.isna().sum().items()
    }

    # --- identifiers and duplicates ---
    r["duplicates"] = {
        "full_row": int(df.duplicated().sum()),
        "all_columns_except_id": int(df.drop(columns="id").duplicated().sum()),
        "id_unique": bool(df["id"].is_unique),
        "id_equals_row_index": bool((df["id"].values == np.arange(len(df))).all()),
        "key_date_carrier_flight_origin": int(
            df.duplicated(["year", "month", "day", "carrier", "flight", "origin"]).sum()),
        "key_date_carrier_flight_origin_dest": int(
            df.duplicated(["year", "month", "day", "carrier", "flight", "origin", "dest"]).sum()),
        "key_date_tailnum_sched_dep_nonnull_tail": int(
            df[df.tailnum.notna()].duplicated(["year", "month", "day", "tailnum", "sched_dep_time"]).sum()),
    }

    # --- dates and chronology ---
    date = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    full_year = pd.date_range(date.min(), date.max(), freq="D")
    per_day = date.value_counts()
    r["dates"] = {
        "invalid_dates": int(date.isna().sum()),
        "years": sorted(df["year"].unique().tolist()),
        "min_date": str(date.min().date()),
        "max_date": str(date.max().date()),
        "n_unique_dates": int(date.nunique()),
        "n_calendar_days_in_range": int(len(full_year)),
        "calendar_days_with_no_flights": [str(d.date()) for d in full_year.difference(per_day.index)],
        "flights_per_day": {
            "min": int(per_day.min()), "median": float(per_day.median()), "max": int(per_day.max()),
            "min_day": str(per_day.idxmin().date()), "max_day": str(per_day.idxmax().date()),
        },
        "rows_per_month": {int(k): int(v) for k, v in df["month"].value_counts().sort_index().items()},
    }
    sched_ts = date + pd.to_timedelta(hhmm_to_minutes(df["sched_dep_time"]), unit="m")
    r["chronology"] = {
        "file_sorted_by_date": bool(date.is_monotonic_increasing),
        "file_sorted_by_sched_departure_ts": bool(sched_ts.is_monotonic_increasing),
        "rows_out_of_sched_departure_order": int((sched_ts.diff().dt.total_seconds() < 0).sum()),
        "month_sequence_in_file_order": [int(m) for m in
                                         df["month"][df["month"].ne(df["month"].shift())]],
    }

    # --- clock fields ---
    r["clock_validity"] = {c: hhmm_validity(df[c])
                           for c in ["dep_time", "sched_dep_time", "arr_time", "sched_arr_time"]}
    th = pd.to_datetime(df["time_hour"], errors="coerce")
    r["derived_field_consistency"] = {
        "hour_minute_vs_sched_dep_time_mismatch": int(
            (df["hour"] * 100 + df["minute"] != df["sched_dep_time"]).sum()),
        "time_hour_unparseable": int(th.isna().sum()),
        "time_hour_vs_date_plus_hour_mismatch": int(
            (th != date + pd.to_timedelta(df["hour"], unit="h")).sum()),
        "carriers": int(df["carrier"].nunique()),
        "carrier_names": int(df["name"].nunique()),
        "carriers_with_multiple_names": int((df.groupby("carrier")["name"].nunique() > 1).sum()),
        "routes_with_multiple_distances": int(
            (df.groupby(["origin", "dest"])["distance"].nunique() > 1).sum()),
    }

    # --- recorded delay vs clock-time differences ---
    r["delay_consistency"] = {
        "dep_delay": delay_consistency(df["dep_time"], df["sched_dep_time"], df["dep_delay"]),
        "arr_delay": delay_consistency(df["arr_time"], df["sched_arr_time"], df["arr_delay"]),
    }

    # --- missing-outcome patterns (cancellation / diversion proxies) ---
    pattern = df[OUTCOME_COLS].isna().apply(
        lambda row: "".join("N" if v else "." for v in row), axis=1)
    r["outcome_missing_patterns"] = {
        "legend": f"order={OUTCOME_COLS}; N=missing, .=present",
        "counts": {k: int(v) for k, v in pattern.value_counts().items()},
    }
    no_dep = df["dep_time"].isna()
    dep_no_arr = df["dep_time"].notna() & df["arr_time"].isna()
    arr_no_delay = df["arr_time"].notna() & df["arr_delay"].isna()
    r["unlabelled_groups"] = {
        "no_departure_time (cancellation proxy)": int(no_dep.sum()),
        "departed_but_no_arrival_time": int(dep_no_arr.sum()),
        "arrival_time_but_no_arr_delay (diversion proxy)": int(arr_no_delay.sum()),
        "total_without_arr_delay": int(df["arr_delay"].isna().sum()),
        "tailnum_missing_among_no_departure": int(df.loc[no_dep, "tailnum"].isna().sum()),
        "tailnum_missing_among_departed": int(df.loc[~no_dep, "tailnum"].isna().sum()),
        "no_departure_by_month": {int(k): int(v) for k, v in
                                  df.loc[no_dep, "month"].value_counts().sort_index().items()},
    }

    # --- delay distributions and extremes ---
    r["delay_distribution"] = {}
    for c in ["dep_delay", "arr_delay"]:
        s = df[c].dropna()
        r["delay_distribution"][c] = {
            "quantiles": quantiles(s),
            "lt_0": int((s < 0).sum()), "eq_0": int((s == 0).sum()),
            "ge_15": int((s >= 15).sum()), "ge_60": int((s >= 60).sum()),
            "ge_180": int((s >= 180).sum()), "ge_600": int((s >= 600).sum()),
            "le_minus_60": int((s <= -60).sum()),
        }

    # --- durations, speed, distance ---
    sched_block_local = (hhmm_to_minutes(df["sched_arr_time"])
                         - hhmm_to_minutes(df["sched_dep_time"]))
    speed = df["distance"] / (df["air_time"] / 60)
    r["durations"] = {
        "note": "sched_arr_time is destination-local; sched_dep_time is NYC-local. "
                "Their difference is NOT a true duration without time-zone data.",
        "sched_arr_minus_sched_dep_local_negative": int((sched_block_local < 0).sum()),
        "air_time_quantiles": quantiles(df["air_time"].dropna()),
        "implied_speed_mph_quantiles": quantiles(speed.dropna()),
        "implied_speed_lt_150mph": int((speed < 150).sum()),
        "implied_speed_gt_650mph": int((speed > 650).sum()),
        "distance_quantiles": quantiles(df["distance"]),
    }

    # --- categorical cardinality and sparsity ---
    route = df["origin"] + "-" + df["dest"]
    route_n = route.value_counts()
    dest_n = df["dest"].value_counts()
    route_date_n = df.groupby([date, route]).size()
    r["categoricals"] = {
        "origin_counts": {k: int(v) for k, v in df["origin"].value_counts().items()},
        "carrier_counts": {k: int(v) for k, v in df["carrier"].value_counts().items()},
        "n_dest": int(df["dest"].nunique()),
        "dest_with_lt_100_flights": int((dest_n < 100).sum()),
        "dest_with_lt_100_flights_list": {k: int(v) for k, v in dest_n[dest_n < 100].items()},
        "n_routes": int(route.nunique()),
        "routes_with_lt_30_flights": int((route_n < 30).sum()),
        "routes_with_lt_100_flights": int((route_n < 100).sum()),
        "route_flight_count_quantiles": quantiles(route_n),
        "route_dates_with_multiple_flights_pct": round(100 * (route_date_n > 1).mean(), 2),
        "max_flights_same_route_same_day": int(route_date_n.max()),
        "n_tailnum": int(df["tailnum"].nunique()),
        "n_flight_numbers": int(df["flight"].nunique()),
        "flight_number_reused_across_carriers": int(
            (df.groupby("flight")["carrier"].nunique() > 1).sum()),
    }

    # --- tail rotations (only NYC departures are present) ---
    tail_day = df[df.tailnum.notna()].groupby([date[df.tailnum.notna()], "tailnum"]).size()
    r["tail_rotation"] = {
        "tail_days": int(tail_day.size),
        "tail_days_with_multiple_nyc_departures_pct": round(100 * (tail_day > 1).mean(), 2),
        "note": "Only departures from EWR/JFK/LGA are present; an aircraft's inbound "
                "legs into NYC are not observed.",
    }

    # --- target feasibility (descriptive only) ---
    lab = df["arr_delay"].dropna()
    y = (lab >= DELAY_THRESHOLD_MIN).astype(int)
    month_lab = df.loc[lab.index, "month"]
    r["target_feasibility"] = {
        "threshold_minutes": DELAY_THRESHOLD_MIN,
        "labelled_rows": int(lab.size),
        "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "prevalence": round(float(y.mean()), 4),
        "arr_delay_exactly_15": int((lab == 15).sum()),
        "arr_delay_exactly_14": int((lab == 14).sum()),
        "arr_delay_is_integer": bool((lab == lab.round()).all()),
        "prevalence_by_month": {int(m): round(float(v), 4)
                                for m, v in y.groupby(month_lab).mean().items()},
        "labelled_by_month": {int(m): int(v) for m, v in month_lab.value_counts().sort_index().items()},
        "prevalence_by_origin": {k: round(float(v), 4) for k, v in
                                 y.groupby(df.loc[lab.index, "origin"]).mean().items()},
    }
    return r


def to_markdown(r: dict) -> str:
    """Render the JSON findings as a readable report; every value comes from `r`."""
    s, m, t = r["schema"], r["missing"], r["target_feasibility"]
    lines = [
        "# Dataset Audit (Stage 1) — measured findings",
        "",
        "Generated by `scripts/audit_dataset.py`. All numbers below are computed; nothing is hand-typed.",
        "",
        "## File",
        f"- Path: `{r['file']['path']}`  ",
        f"- Size: {r['file']['size_bytes']:,} bytes  ",
        f"- SHA-256: `{r['file']['sha256']}`",
        f"- Rows × columns: {s['n_rows']:,} × {s['n_cols']}  ",
        f"- In-memory size (pandas, deep): {s['memory_mb_deep']} MB",
        "",
        "## Columns, dtypes, missingness",
        "| column | dtype | missing | % |",
        "|---|---|---:|---:|",
    ]
    for c in s["columns"]:
        lines.append(f"| {c} | {s['dtypes'][c]} | {m[c]['count']:,} | {m[c]['pct']} |")
    sections = [
        ("Duplicates and identifiers", r["duplicates"]),
        ("Date coverage", r["dates"]),
        ("Chronological ordering of the file", r["chronology"]),
        ("Clock-field validity (HHMM)", r["clock_validity"]),
        ("Derived-field consistency", r["derived_field_consistency"]),
        ("Recorded delay vs. clock-time difference", r["delay_consistency"]),
        ("Missing-outcome patterns", r["outcome_missing_patterns"]),
        ("Unlabelled groups", r["unlabelled_groups"]),
        ("Delay distributions", r["delay_distribution"]),
        ("Durations, speed, distance", r["durations"]),
        ("Categorical cardinality", r["categoricals"]),
        ("Tail rotation", r["tail_rotation"]),
    ]
    for title, body in sections:
        lines += ["", f"## {title}", "```json", json.dumps(body, indent=2), "```"]
    lines += [
        "", f"## Target feasibility (arr_delay ≥ {t['threshold_minutes']} min, labelled rows only)",
        f"- Labelled rows: {t['labelled_rows']:,}  ",
        f"- Positive: {t['positive']:,}; negative: {t['negative']:,}; prevalence: {t['prevalence']}",
        "```json", json.dumps({k: t[k] for k in t if k.startswith(("prevalence_by", "labelled_by", "arr_delay"))}, indent=2), "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")
    results = audit(args.data.resolve())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_audit.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "dataset_audit.md").write_text(to_markdown(results))
    print(f"Wrote {OUT_DIR / 'dataset_audit.json'} and dataset_audit.md")


if __name__ == "__main__":
    main()
