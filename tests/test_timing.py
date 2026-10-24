import pandas as pd
import pytest

from conftest import NAN, make_flights
from flight_delay.data.timing import (actual_departure_ts, hhmm_to_minutes, landing_lower_bound,
                                      scheduled_departure_ts)


def test_hhmm_to_minutes_including_midnight_and_nan():
    out = hhmm_to_minutes(pd.Series([0, 1, 59, 100, 1545, 2359, 2400, NAN]))
    assert out.iloc[:7].tolist() == [0, 1, 59, 60, 945, 1439, 1440]
    assert pd.isna(out.iloc[7])


def test_hhmm_rejects_invalid_minutes():
    with pytest.raises(ValueError):
        hhmm_to_minutes(pd.Series([1260]))


def test_departure_after_midnight_lands_on_next_date():
    # Scheduled 23:30 on Jan 1, departed 75 min late -> 00:45 on Jan 2.
    df = make_flights([dict(sched_dep_time=2330, dep_time=45.0, dep_delay=75.0, air_time=100.0)])
    assert scheduled_departure_ts(df).iloc[0] == pd.Timestamp("2013-01-01 23:30")
    assert actual_departure_ts(df).iloc[0] == pd.Timestamp("2013-01-02 00:45")
    assert landing_lower_bound(df).iloc[0] == pd.Timestamp("2013-01-02 02:25")


def test_landing_bound_is_nat_for_cancelled():
    df = make_flights([dict(dep_time=NAN, dep_delay=NAN, arr_time=NAN, arr_delay=NAN, air_time=NAN)])
    assert pd.isna(landing_lower_bound(df).iloc[0])
