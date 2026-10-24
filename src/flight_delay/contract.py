"""Feature-availability registry: the code form of docs/02_prediction_contract.md.

Every raw column is classified by whether its value is known at the prediction timestamp
(end of the day before the flight, decision D3). Feature-building code must declare which
raw columns it reads; `assert_feature_sources_permitted` rejects anything not permitted.

Outcome columns marked `history_eligible` may feed historical aggregates of *earlier*
flights only, subject to the historical-outcome cutoff (decision D10); they are never inputs
describing the flight being predicted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    meaning: str
    known_at: str
    available_at_prediction: str  # "yes" | "no" | "unverifiable"
    permitted_as_input: bool
    history_eligible: bool
    reason: str
    transformation: str = ""


SCHEDULE = "published schedule (before D-1)"
POST_EVENT = "during/after the flight on date D"

FIELD_REGISTRY: dict[str, FieldSpec] = {
    "id": FieldSpec("row identifier added to the Kaggle export", "n/a", "yes", False, False,
                    "Export artefact; encodes file order (not time); no predictive meaning"),
    "year": FieldSpec("scheduled year", SCHEDULE, "yes", True, False,
                      "Schedule field", "Combined with month/day into flight_date (constant 2013)"),
    "month": FieldSpec("scheduled month", SCHEDULE, "yes", True, False,
                       "Schedule field; note test months (11-12) are absent from training",
                       "Calendar features from flight_date; use as a feature decided in Stage 4"),
    "day": FieldSpec("scheduled day of month", SCHEDULE, "yes", True, False,
                     "Schedule field", "Combined into flight_date -> day-of-week, weekend"),
    "dep_time": FieldSpec("actual gate departure, NYC local HHMM", POST_EVENT, "no", False, True,
                          "Post-event outcome; also defines the cancellation proxy"),
    "sched_dep_time": FieldSpec("scheduled departure, NYC local HHMM", SCHEDULE, "yes", True, False,
                                "Schedule field", "Departure hour/minute, cyclic hour encoding"),
    "dep_delay": FieldSpec("departure delay, minutes", POST_EVENT, "no", False, True,
                           "Post-event outcome; strongly coupled to the target"),
    "arr_time": FieldSpec("actual arrival, destination-local HHMM", POST_EVENT, "no", False, False,
                          "Post-event outcome; defines the target"),
    "sched_arr_time": FieldSpec("scheduled arrival, destination-local HHMM", SCHEDULE, "yes", True, False,
                                "Schedule field; destination-local, no time zone",
                                "Scheduled arrival local hour only; never subtracted from "
                                "sched_dep_time (decision D12)"),
    "arr_delay": FieldSpec("arrival delay, minutes (target source)", POST_EVENT, "no", False, True,
                           "Defines the target; prohibited as an input for the flight itself"),
    "carrier": FieldSpec("two-letter carrier code", SCHEDULE, "yes", True, False,
                         "Schedule field", "Categorical encoding fitted on training data"),
    "flight": FieldSpec("flight number", SCHEDULE, "yes", False, False,
                        "Available, but a high-cardinality identifier (3,844 values; 1,365 "
                        "reused across carriers) with no defensible generalisation"),
    "tailnum": FieldSpec("aircraft tail number", "assignment may change on D (swaps observed)",
                         "unverifiable", False, False,
                         "Assignment at end of D-1 cannot be verified; 31 same-day swaps "
                         "observed (decision D13)"),
    "origin": FieldSpec("origin airport (EWR/JFK/LGA)", SCHEDULE, "yes", True, False,
                        "Schedule field", "Categorical encoding fitted on training data"),
    "dest": FieldSpec("destination airport", SCHEDULE, "yes", True, False,
                      "Schedule field", "Categorical encoding with explicit unseen/rare handling"),
    "air_time": FieldSpec("actual airborne minutes", POST_EVENT, "no", False, False,
                          "Post-event outcome"),
    "distance": FieldSpec("great-circle route distance, miles", SCHEDULE, "yes", True, False,
                          "Fixed property of the route", "Numeric; proxy for scheduled duration"),
    "hour": FieldSpec("scheduled departure hour", SCHEDULE, "yes", True, False,
                      "Derived from sched_dep_time (verified 0 mismatches)", "Same as sched_dep_time"),
    "minute": FieldSpec("scheduled departure minute", SCHEDULE, "yes", True, False,
                        "Derived from sched_dep_time (verified 0 mismatches)", "Same as sched_dep_time"),
    "time_hour": FieldSpec("scheduled date + hour string", SCHEDULE, "yes", False, False,
                           "Redundant with flight_date + hour (verified 0 mismatches)"),
    "name": FieldSpec("carrier full name", SCHEDULE, "yes", False, False,
                      "Redundant: 1:1 with carrier (verified)"),
}

PERMITTED_INPUTS = frozenset(k for k, v in FIELD_REGISTRY.items() if v.permitted_as_input)
HISTORY_ELIGIBLE = frozenset(k for k, v in FIELD_REGISTRY.items() if v.history_eligible)
PROHIBITED_INPUTS = frozenset(k for k, v in FIELD_REGISTRY.items()
                              if v.available_at_prediction != "yes")


def assert_registry_covers(columns: Iterable[str]) -> None:
    """Every raw column must be classified, and the registry must not describe absent columns."""
    columns = set(columns)
    unclassified = columns - FIELD_REGISTRY.keys()
    stale = FIELD_REGISTRY.keys() - columns
    if unclassified or stale:
        raise ValueError(f"Registry/schema mismatch. Unclassified: {sorted(unclassified)}; "
                         f"not in data: {sorted(stale)}")


def assert_feature_sources_permitted(sources: Iterable[str]) -> None:
    """Reject any feature that reads a raw column not permitted as a current-flight input."""
    bad = {s: FIELD_REGISTRY[s].reason if s in FIELD_REGISTRY else "unknown column"
           for s in sources if s not in PERMITTED_INPUTS}
    if bad:
        raise ValueError(f"Feature sources not permitted at the prediction timestamp: {bad}")


def assert_history_sources_eligible(sources: Iterable[str]) -> None:
    """Reject historical aggregates built from columns not eligible as past outcomes."""
    bad = sorted(s for s in sources if s not in HISTORY_ELIGIBLE and s not in PERMITTED_INPUTS)
    if bad:
        raise ValueError(f"Columns not eligible for historical aggregation: {bad}")


def availability_table() -> pd.DataFrame:
    return pd.DataFrame([{"raw_field": k, **asdict(v)} for k, v in FIELD_REGISTRY.items()])
