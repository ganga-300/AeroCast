"""Shared toy fixtures. Toy data verifies logic only; it is not a project result."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from flight_delay.config import load_config


def make_flights(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-schema frame; unspecified fields get valid, operated-flight defaults."""
    defaults = dict(year=2013, month=1, day=1, dep_time=900.0, sched_dep_time=900,
                    dep_delay=0.0, arr_time=1100.0, sched_arr_time=1100, arr_delay=0.0,
                    carrier="AA", flight=1, tailnum="N1", origin="JFK", dest="BOS",
                    air_time=60.0, distance=187, name="American Airlines Inc.")
    out = []
    for i, r in enumerate(rows):
        row = {"id": i, **defaults, **r}
        row["hour"], row["minute"] = divmod(row["sched_dep_time"], 100)
        row["time_hour"] = f"{row['year']}-{row['month']:02d}-{row['day']:02d} {row['hour']:02d}:00:00"
        out.append(row)
    df = pd.DataFrame(out)
    for c in ("carrier", "tailnum", "origin", "dest", "time_hour", "name"):
        df[c] = df[c].astype("string")
    for c in ("dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"):
        df[c] = df[c].astype(float)
    return df


@pytest.fixture
def cfg() -> dict:
    return copy.deepcopy(load_config())


NAN = np.nan
