"""Historical outcome features under the D-lag cutoff (decisions D3, D10, D16).

For a flight on date D, every aggregate uses only flights whose *flight date* is <= D - lag
(lag = 2: outcomes are verified to be complete before the end of D-1). Construction:

1. Aggregate outcomes per (entity, flight date): labelled count, positives, scheduled count,
   never-departed count. Rows with missing labels add to no label count (never negatives).
2. Lay the daily totals on a contiguous calendar and take cumulative sums.
3. The value for date D is cumsum(D - lag) for the expanding window, and
   cumsum(D - lag) - cumsum(D - lag - w) for a w-day window ending at D - lag.

Because the result depends only on dates and entity keys, it is independent of row order,
never contains the flight's own (or any same-day) outcome, and cannot change when records
for later dates are appended. The split column is never read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from flight_delay.features.schedule import route_key
from flight_delay.features.specs import count_feature, window_label

ALL = "__all__"


def _outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"]
    return pd.DataFrame({
        "n": y.notna().astype("int64"),
        "pos": (y == 1).fillna(False).astype("int64"),
        "sched": np.ones(len(df), dtype="int64"),
        "nodep": (df["label_status"] == "no_departure").astype("int64"),
    }, index=df.index)


def lagged_window_sums(dates: pd.Series, keys: pd.Series, values: pd.DataFrame,
                       lag: int, windows: list[int | None]) -> dict[tuple[str, str], pd.Series]:
    """Per-row sums of `values` over flights with the same key and flight date in the window
    ending at D - lag. Returns {(value_column, window_label): Series aligned to `dates`}."""
    if lag < 1:
        raise ValueError("lag must be >= 1 so a flight's own date is never included")
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    frame = values.assign(_date=dates.values, _key=keys.values)
    daily = frame.groupby(["_date", "_key"], observed=True)[list(values.columns)].sum()
    lookup = pd.MultiIndex.from_arrays([dates.values, keys.values])
    out = {}
    for col in values.columns:
        cum = (daily[col].unstack("_key").reindex(calendar).fillna(0).cumsum())
        end = cum.shift(lag, fill_value=0)  # row shift == day shift on a contiguous calendar
        for w in windows:
            total = end if w is None else end - cum.shift(lag + w, fill_value=0)
            stacked = total.stack()
            out[(col, window_label(w))] = pd.Series(
                stacked.reindex(lookup).to_numpy(), index=dates.index).fillna(0)
    return out


def raw_rate(pos: pd.Series, n: pd.Series) -> pd.Series:
    """pos / n, NaN when there is no history."""
    return (pos / n.where(n > 0)).astype("float64")


def shrunk_rate(pos: pd.Series, n: pd.Series, prior: pd.Series, min_count: int, m: float) -> pd.Series:
    """(pos + m * prior) / (n + m), replaced by the prior itself when n < min_count."""
    shrunk = (pos + m * prior) / (n + m)
    return shrunk.where(n >= min_count, prior).astype("float64")


def build_history_features(df: pd.DataFrame, lag: int, hist_cfg: dict[str, Any]) -> pd.DataFrame:
    dates = df["flight_date"]
    values = _outcome_columns(df)
    out = pd.DataFrame(index=df.index)

    g_windows = [None, *hist_cfg["global_windows_days"]]
    g = lagged_window_sums(dates, pd.Series(ALL, index=df.index), values, lag, g_windows)
    prior = {}
    for w in g_windows:
        lab = window_label(w)
        prior[lab] = raw_rate(g[("pos", lab)], g[("n", lab)])
        out[f"hist_all_rate_{lab}"] = prior[lab]
        if count_feature(w):
            out[f"hist_all_n_{lab}"] = g[("n", lab)].astype("int64")
    w0 = window_label(min(hist_cfg["global_windows_days"]))
    out[f"hist_all_nodep_share_{w0}"] = raw_rate(g[("nodep", w0)], g[("sched", w0)])

    e_windows = [None, *hist_cfg["entity_windows_days"]]
    missing_priors = {window_label(w) for w in e_windows} - prior.keys()
    if missing_priors:  # entity windows need a system-wide rate of the same length as prior
        extra = lagged_window_sums(dates, pd.Series(ALL, index=df.index), values, lag,
                                   [int(l[:-1]) for l in sorted(missing_priors)])
        for lab in missing_priors:
            prior[lab] = raw_rate(extra[("pos", lab)], extra[("n", lab)])

    for entity in hist_cfg["entities"]:
        keys = route_key(df) if entity == "route" else df[entity]
        e = lagged_window_sums(dates, keys, values[["n", "pos"]], lag, e_windows)
        for w in e_windows:
            lab = window_label(w)
            out[f"hist_{entity}_rate_{lab}"] = shrunk_rate(
                e[("pos", lab)], e[("n", lab)], prior[lab],
                hist_cfg["min_count"], hist_cfg["smoothing_m"])
            if count_feature(w):
                out[f"hist_{entity}_n_{lab}"] = e[("n", lab)].astype("int64")
    return out
