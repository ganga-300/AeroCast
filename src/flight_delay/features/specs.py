"""Declarative list of model features and the raw columns each one reads.

Every feature declares its raw sources so the contract can be enforced mechanically:
schedule features may read only fields permitted for the flight being predicted, and
historical features may read outcome columns only of *other, earlier* flights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flight_delay.contract import assert_feature_sources_permitted, assert_history_sources_eligible

EXCLUDED_FEATURES = {
    "month": "Training covers Jan-Aug only; Sep-Dec values would be unseen (decision D15)",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str          # "numeric" | "categorical"
    group: str         # "schedule" | "history"
    sources: tuple[str, ...]
    description: str


DATE_SOURCES = ("year", "month", "day")

SCHEDULE_SPECS = [
    FeatureSpec("dep_hour", "numeric", "schedule", ("sched_dep_time",),
                "Scheduled departure hour, NYC local (0-23)"),
    FeatureSpec("dep_hour_sin", "numeric", "schedule", ("sched_dep_time",),
                "sin(2π·h/24), h = scheduled departure time in fractional hours"),
    FeatureSpec("dep_hour_cos", "numeric", "schedule", ("sched_dep_time",),
                "cos(2π·h/24), h = scheduled departure time in fractional hours"),
    FeatureSpec("arr_hour", "numeric", "schedule", ("sched_arr_time",),
                "Scheduled arrival hour, destination local (0-23); never combined with "
                "departure time into a duration (D12)"),
    FeatureSpec("arr_hour_sin", "numeric", "schedule", ("sched_arr_time",),
                "sin(2π·h/24), h = scheduled arrival time in fractional hours (dest local)"),
    FeatureSpec("arr_hour_cos", "numeric", "schedule", ("sched_arr_time",),
                "cos(2π·h/24), h = scheduled arrival time in fractional hours (dest local)"),
    FeatureSpec("day_of_week", "categorical", "schedule", DATE_SOURCES,
                "Day of week of the scheduled flight date (0 = Monday)"),
    FeatureSpec("is_weekend", "numeric", "schedule", DATE_SOURCES,
                "1 if the scheduled flight date is Saturday or Sunday"),
    FeatureSpec("distance", "numeric", "schedule", ("distance",),
                "Route distance in miles (proxy for scheduled duration, D12)"),
    FeatureSpec("carrier", "categorical", "schedule", ("carrier",), "Carrier code"),
    FeatureSpec("origin", "categorical", "schedule", ("origin",), "Origin airport"),
    FeatureSpec("dest", "categorical", "schedule", ("dest",), "Destination airport"),
    FeatureSpec("route", "categorical", "schedule", ("origin", "dest"), "Origin-destination pair"),
]

ENTITY_SOURCES = {"carrier": ("carrier",), "origin": ("origin",), "dest": ("dest",),
                  "route": ("origin", "dest")}


def window_label(w: int | None) -> str:
    return "exp" if w is None else f"{w}d"


def count_feature(window: int | None) -> bool:
    """Expanding-window counts grow monotonically with the date, so they act as a hidden time
    index whose validation/test values lie outside the training range (decision D17). Counts
    are therefore exposed only for fixed-length windows."""
    return window is not None


def history_specs(hist_cfg: dict[str, Any], lag: int) -> list[FeatureSpec]:
    """Historical feature specs generated from config (decisions D16, D17)."""
    cutoff = f"flight dates <= D-{lag}"
    specs: list[FeatureSpec] = []
    for w in [None, *hist_cfg["global_windows_days"]]:
        span = "all dates" if w is None else f"the {w} days ending at D-{lag}"
        lab = window_label(w)
        specs.append(FeatureSpec(f"hist_all_rate_{lab}", "numeric", "history", ("arr_delay",),
                                 f"System-wide share of labelled flights ≥15 min late over {span} "
                                 f"({cutoff}); NaN if no history"))
        if count_feature(w):
            specs.append(FeatureSpec(f"hist_all_n_{lab}", "numeric", "history", ("arr_delay",),
                                     f"Number of labelled flights behind hist_all_rate_{lab}"))
    w0 = min(hist_cfg["global_windows_days"])
    specs.append(FeatureSpec(f"hist_all_nodep_share_{w0}d", "numeric", "history", ("dep_time",),
                             f"Share of scheduled flights that never departed over the {w0} days "
                             f"ending at D-{lag}; NaN if no history"))
    for e in hist_cfg["entities"]:
        for w in [None, *hist_cfg["entity_windows_days"]]:
            span = "all dates" if w is None else f"the {w} days ending at D-{lag}"
            lab = window_label(w)
            specs.append(FeatureSpec(
                f"hist_{e}_rate_{lab}", "numeric", "history", ("arr_delay", *ENTITY_SOURCES[e]),
                f"Shrunk delay share for this {e} over {span}; equals the system-wide rate for "
                f"the same window when fewer than min_count outcomes exist"))
            if count_feature(w):
                specs.append(FeatureSpec(
                    f"hist_{e}_n_{lab}", "numeric", "history", ("arr_delay", *ENTITY_SOURCES[e]),
                    f"Number of labelled flights for this {e} over {span}"))
    return specs


def all_specs(cfg: dict[str, Any], lag: int) -> list[FeatureSpec]:
    return SCHEDULE_SPECS + history_specs(cfg["features"]["history"], lag)


def validate_specs(specs: list[FeatureSpec]) -> None:
    """Enforce the prediction contract on declared sources and basic spec hygiene."""
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names")
    banned = set(names) & EXCLUDED_FEATURES.keys()
    if banned:
        raise ValueError(f"Excluded features present: {banned}")
    for s in specs:
        if s.kind not in {"numeric", "categorical"} or s.group not in {"schedule", "history"}:
            raise ValueError(f"Bad kind/group for {s.name}")
        if s.group == "schedule":
            assert_feature_sources_permitted(s.sources)
        else:
            assert_history_sources_eligible(s.sources)
