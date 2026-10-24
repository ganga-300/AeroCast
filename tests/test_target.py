import pandas as pd

from conftest import NAN, make_flights
from flight_delay.data.target import build_target, label_status


def test_threshold_boundaries_and_early_arrivals():
    arr = pd.Series([15.0, 14.0, 16.0, 0.0, -30.0, 1272.0])
    assert build_target(arr, 15).tolist() == [1, 0, 1, 0, 0, 1]


def test_missing_label_is_na_not_negative():
    y = build_target(pd.Series([NAN, 3.0, NAN, 20.0]), 15)
    assert str(y.dtype) == "Int8"
    assert y.isna().tolist() == [True, False, True, False]
    assert int((y == 0).sum()) == 1  # only the real negative


def test_label_status_patterns():
    df = make_flights([
        {},                                                        # operated, labelled
        dict(dep_time=NAN, dep_delay=NAN, arr_time=NAN, arr_delay=NAN, air_time=NAN, tailnum=None),
        dict(arr_time=NAN, arr_delay=NAN, air_time=NAN),           # departed, no arrival
        dict(arr_delay=NAN, air_time=NAN),                         # arrived, no delay
    ])
    assert label_status(df).tolist() == [
        "labelled", "no_departure", "departed_no_arrival", "arrived_no_delay"]


def test_target_does_not_modify_input():
    arr = pd.Series([10.0, NAN, 30.0])
    before = arr.copy()
    build_target(arr, 15)
    pd.testing.assert_series_equal(arr, before)
