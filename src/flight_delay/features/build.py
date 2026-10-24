"""Assemble the feature table: metadata + schedule features + historical features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flight_delay.config import require_history_lag
from flight_delay.features.history import build_history_features
from flight_delay.features.schedule import build_schedule_features
from flight_delay.features.specs import FeatureSpec, all_specs, validate_specs

META_COLUMNS = ["id", "flight_date", "sched_dep_ts", "split", "label_status", "y"]


def feature_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    specs = all_specs(cfg, require_history_lag(cfg))
    validate_specs(specs)
    return specs


def build_feature_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build features for every row of a prepared flight table (`prepare_flights` output).

    Unlabelled rows are kept (y = <NA>) so they can contribute disruption history and be
    counted; `modelling_population` removes them before fitting or evaluation.
    """
    specs = feature_specs(cfg)
    lag = require_history_lag(cfg)
    sched = build_schedule_features(df)
    hist = build_history_features(df, lag, cfg["features"]["history"])
    table = pd.concat([df[META_COLUMNS], sched, hist], axis=1)
    expected = META_COLUMNS + [s.name for s in specs]
    if list(table.columns) != expected:
        raise AssertionError(f"Feature columns differ from specs: "
                             f"{sorted(set(table.columns) ^ set(expected))}")
    return table


def feature_names(specs: list[FeatureSpec], kind: str | None = None) -> list[str]:
    return [s.name for s in specs if kind is None or s.kind == kind]
