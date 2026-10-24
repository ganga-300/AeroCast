"""Target construction (decisions D4, D6).

Missing labels are kept as <NA> and never coerced to the negative class.
"""

from __future__ import annotations

import pandas as pd

LABELLED = "labelled"
LABEL_STATUSES = (
    LABELLED,
    "no_departure",          # dep_time missing: cancellation proxy
    "departed_no_arrival",   # departed, no arr_time: diverted/returned, cannot tell
    "arrived_no_delay",      # arr_time present but arr_delay missing: diversion proxy
)


def label_status(df: pd.DataFrame) -> pd.Series:
    """Why a row does or does not have a usable arrival-delay label.

    Inferred from missingness patterns; the raw data has no cancellation or diversion flags.
    """
    status = pd.Series("arrived_no_delay", index=df.index, dtype="object")
    status[df["arr_time"].isna()] = "departed_no_arrival"
    status[df["dep_time"].isna()] = "no_departure"
    status[df["arr_delay"].notna()] = LABELLED
    return pd.Series(pd.Categorical(status, categories=LABEL_STATUSES),
                     index=df.index, name="label_status")


def build_target(arr_delay: pd.Series, threshold_minutes: float) -> pd.Series:
    """y = 1 if arr_delay >= threshold, 0 if below, <NA> if arr_delay is missing."""
    y = pd.Series(pd.NA, index=arr_delay.index, dtype="Int8", name="y")
    known = arr_delay.notna()
    y[known] = (arr_delay[known] >= threshold_minutes).astype("int8")
    return y
