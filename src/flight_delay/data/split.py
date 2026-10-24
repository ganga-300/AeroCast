"""Chronological train/validation/test split by verified flight date (decision D5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp  # inclusive
    end: pd.Timestamp    # inclusive

    def contains(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.start) & (dates <= self.end)


def parse_boundaries(split_cfg: dict[str, Any]) -> dict[str, DateRange]:
    """Parse and validate the configured date ranges: each well-formed, strictly ordered,
    non-overlapping, in the order train < validation < test."""
    missing = [s for s in SPLIT_ORDER if s not in split_cfg]
    if missing:
        raise ValueError(f"Split config missing partitions: {missing}")
    ranges = {}
    for name in SPLIT_ORDER:
        r = DateRange(pd.Timestamp(split_cfg[name]["start"]), pd.Timestamp(split_cfg[name]["end"]))
        if r.start > r.end:
            raise ValueError(f"Split '{name}' starts after it ends: {r.start.date()} > {r.end.date()}")
        ranges[name] = r
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        if not ranges[earlier].end < ranges[later].start:
            raise ValueError(
                f"Split '{earlier}' (ends {ranges[earlier].end.date()}) must end before "
                f"'{later}' starts ({ranges[later].start.date()})")
    return ranges


def assign_split(dates: pd.Series, boundaries: dict[str, DateRange]) -> pd.Series:
    """Map each flight date to its partition. Dates outside every range raise instead of
    being dropped silently. Depends only on each row's date, so row order is irrelevant."""
    split = pd.Series(pd.NA, index=dates.index, dtype="object", name="split")
    for name, r in boundaries.items():
        split[r.contains(dates)] = name
    outside = split.isna()
    if outside.any():
        raise ValueError(f"{int(outside.sum())} rows fall outside all split ranges, "
                         f"e.g. {dates[outside].min().date()}")
    split = pd.Series(pd.Categorical(split, categories=SPLIT_ORDER), index=dates.index, name="split")
    assert_chronological(dates, split)
    return split


def assert_chronological(dates: pd.Series, split: pd.Series) -> None:
    """Assert every date belongs to one partition and partitions are strictly ordered in time."""
    per_date = split.groupby(dates, observed=True).nunique()
    if (per_date > 1).any():
        raise AssertionError("A flight date is assigned to more than one partition")
    present = [s for s in SPLIT_ORDER if (split == s).any()]
    for earlier, later in zip(present, present[1:]):
        if not dates[split == earlier].max() < dates[split == later].min():
            raise AssertionError(f"Partition '{earlier}' overlaps or follows '{later}' in time")


def split_summary(dates: pd.Series, split: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Per-partition row counts and prevalence; prevalence uses labelled rows only."""
    rows = []
    for name in SPLIT_ORDER:
        m = split == name
        lab = y[m].dropna()
        rows.append({
            "split": name,
            "date_min": dates[m].min().date().isoformat(),
            "date_max": dates[m].max().date().isoformat(),
            "n_days": int(dates[m].nunique()),
            "rows_all": int(m.sum()),
            "rows_unlabelled": int(y[m].isna().sum()),
            "rows_labelled": int(lab.size),
            "positives": int((lab == 1).sum()),
            "negatives": int((lab == 0).sum()),
            "prevalence": round(float(lab.mean()), 4) if lab.size else None,
        })
    return pd.DataFrame(rows)
