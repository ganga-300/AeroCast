"""Features describing the flight itself, built only from published-schedule fields."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flight_delay.data.timing import flight_date, hhmm_to_minutes

HOURS_PER_DAY = 24


def cyclic_hour(fractional_hour: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Encode a time of day on a 24-hour circle so 23:59 and 00:00 are neighbours."""
    angle = 2 * np.pi * fractional_hour / HOURS_PER_DAY
    return np.sin(angle), np.cos(angle)


def route_key(df: pd.DataFrame) -> pd.Series:
    return (df["origin"] + "-" + df["dest"]).rename("route")


def build_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reads only sched_dep_time, sched_arr_time, year/month/day, distance, carrier, origin, dest."""
    dep_frac = hhmm_to_minutes(df["sched_dep_time"]) / 60
    arr_frac = hhmm_to_minutes(df["sched_arr_time"]) / 60
    dow = flight_date(df).dt.day_of_week
    out = pd.DataFrame(index=df.index)
    out["dep_hour"] = (df["sched_dep_time"] // 100).astype("int64")
    out["dep_hour_sin"], out["dep_hour_cos"] = cyclic_hour(dep_frac)
    out["arr_hour"] = (df["sched_arr_time"] // 100).astype("int64") % HOURS_PER_DAY
    out["arr_hour_sin"], out["arr_hour_cos"] = cyclic_hour(arr_frac)
    out["day_of_week"] = dow.astype("int64")
    out["is_weekend"] = (dow >= 5).astype("int64")
    out["distance"] = df["distance"].astype("float64")
    for col in ("carrier", "origin", "dest"):
        out[col] = df[col].astype("string")
    out["route"] = route_key(df).astype("string")
    return out
