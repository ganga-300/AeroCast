"""Reconstruct dates and timestamps from the raw clock fields.

All timestamps are naive NYC wall-clock times. The data has no time-zone or DST information,
so wall-clock arithmetic can be off by one hour on the two DST-transition nights
(2013-03-10, 2013-11-03). Destination-local fields (`arr_time`, `sched_arr_time`) are never
combined with NYC-local fields to form durations (decision D12).
"""

from __future__ import annotations

import pandas as pd


def hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert HHMM clock values to minutes after midnight (2400 -> 1440). NaN stays NaN."""
    hours, minutes = s // 100, s % 100
    if ((minutes >= 60) | (hours > 24)).any():
        raise ValueError("Invalid HHMM clock value encountered")
    return hours * 60 + minutes


def flight_date(df: pd.DataFrame) -> pd.Series:
    """Scheduled flight date built from year/month/day; invalid dates raise."""
    return pd.to_datetime(df[["year", "month", "day"]], errors="raise").rename("flight_date")


def scheduled_departure_ts(df: pd.DataFrame) -> pd.Series:
    """Scheduled departure timestamp (NYC local) = flight date + sched_dep_time."""
    return (flight_date(df)
            + pd.to_timedelta(hhmm_to_minutes(df["sched_dep_time"]), unit="m")
            ).rename("sched_dep_ts")


def actual_departure_ts(df: pd.DataFrame) -> pd.Series:
    """Actual gate departure (NYC local) = scheduled departure + dep_delay.

    The Stage 1 audit verified that dep_delay reconciles exactly with dep_time, so this
    correctly places departures that slipped past midnight on the next date. NaT when the
    flight never departed.
    """
    return (scheduled_departure_ts(df)
            + pd.to_timedelta(df["dep_delay"], unit="m")).rename("actual_dep_ts")


def landing_lower_bound(df: pd.DataFrame) -> pd.Series:
    """Earliest possible landing time (NYC wall clock) = actual gate departure + air_time.

    It is a lower bound because taxi-out and taxi-in are non-negative and not recorded.
    Used to check when a flight's arrival outcome could first have been known.
    """
    return (actual_departure_ts(df)
            + pd.to_timedelta(df["air_time"], unit="m")).rename("landing_lower_bound")
