import pytest

from flight_delay.config import require_history_lag, validate_config
from flight_delay.contract import (FIELD_REGISTRY, PERMITTED_INPUTS, assert_feature_sources_permitted,
                                   assert_history_sources_eligible, assert_registry_covers)
from flight_delay.data.load import EXPECTED_COLUMNS

POST_EVENT = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]


def test_registry_matches_expected_schema():
    assert_registry_covers(EXPECTED_COLUMNS)
    with pytest.raises(ValueError):
        assert_registry_covers(list(EXPECTED_COLUMNS) + ["cancelled"])
    with pytest.raises(ValueError):
        assert_registry_covers([c for c in EXPECTED_COLUMNS if c != "distance"])


@pytest.mark.parametrize("col", POST_EVENT + ["tailnum", "id", "unknown_col"])
def test_post_event_and_unverifiable_inputs_rejected(col):
    with pytest.raises(ValueError):
        assert_feature_sources_permitted([col])


def test_schedule_fields_permitted():
    assert_feature_sources_permitted(["sched_dep_time", "carrier", "origin", "dest", "distance"])


def test_no_permitted_field_is_unavailable():
    for name in PERMITTED_INPUTS:
        assert FIELD_REGISTRY[name].available_at_prediction == "yes"


def test_history_eligibility():
    assert_history_sources_eligible(["arr_delay", "dep_delay", "dep_time"])
    with pytest.raises(ValueError):
        assert_history_sources_eligible(["air_time"])


def test_open_history_lag_blocks_historical_features(cfg):
    cfg["prediction"]["history_lag_days"] = None
    with pytest.raises(ValueError, match="D10"):
        require_history_lag(cfg)
    cfg["prediction"]["history_lag_days"] = 2
    assert require_history_lag(cfg) == 2


@pytest.mark.parametrize("section,key", [("target", "threshold_minutes"), ("split", "test"),
                                         ("operating_point", "min_precision")])
def test_missing_config_field_raises(cfg, section, key):
    del cfg[section][key]
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_confirmed_values_frozen_in_config(cfg):
    assert cfg["target"]["threshold_minutes"] == 15
    assert cfg["operating_point"]["min_precision"] == 0.70
    assert cfg["prediction"]["timestamp_rule"] == "end_of_previous_day"
    assert cfg["split"]["validation"] == {"start": "2013-09-01", "end": "2013-10-31"}
