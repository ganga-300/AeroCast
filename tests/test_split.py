import pandas as pd
import pytest

from conftest import NAN, make_flights
from flight_delay.data.load import modelling_population, prepare_flights
from flight_delay.data.split import assert_chronological, assign_split, parse_boundaries


def test_confirmed_boundaries_assign_edge_dates(cfg):
    dates = pd.Series(pd.to_datetime(
        ["2013-01-01", "2013-08-31", "2013-09-01", "2013-10-31", "2013-11-01", "2013-12-31"]))
    split = assign_split(dates, parse_boundaries(cfg["split"]))
    assert split.tolist() == ["train", "train", "validation", "validation", "test", "test"]


@pytest.mark.parametrize("bad", [
    # overlap: validation starts on the last train day
    {"train": {"start": "2013-01-01", "end": "2013-09-01"},
     "validation": {"start": "2013-09-01", "end": "2013-10-31"},
     "test": {"start": "2013-11-01", "end": "2013-12-31"}},
    # wrong order: test before validation
    {"train": {"start": "2013-01-01", "end": "2013-08-31"},
     "validation": {"start": "2013-11-01", "end": "2013-12-31"},
     "test": {"start": "2013-09-01", "end": "2013-10-31"}},
    # start after end
    {"train": {"start": "2013-08-31", "end": "2013-01-01"},
     "validation": {"start": "2013-09-01", "end": "2013-10-31"},
     "test": {"start": "2013-11-01", "end": "2013-12-31"}},
])
def test_invalid_boundaries_raise(bad):
    with pytest.raises(ValueError):
        parse_boundaries(bad)


def test_dates_outside_ranges_raise(cfg):
    with pytest.raises(ValueError, match="outside"):
        assign_split(pd.Series(pd.to_datetime(["2014-01-01"])), parse_boundaries(cfg["split"]))


def test_assert_chronological_detects_interleaving():
    dates = pd.Series(pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]))
    split = pd.Series(["train", "validation", "train"])
    with pytest.raises(AssertionError):
        assert_chronological(dates, split)


def test_prepare_is_row_order_independent_and_keeps_raw_intact(cfg):
    raw = make_flights([
        dict(month=12, day=31, sched_dep_time=2359, arr_delay=40.0),
        dict(month=1, day=1, sched_dep_time=600, arr_delay=-5.0),
        dict(month=9, day=1, sched_dep_time=700, arr_delay=NAN, air_time=NAN),
        dict(month=1, day=1, sched_dep_time=600, arr_delay=15.0, carrier="UA"),
    ])
    snapshot = raw.copy(deep=True)
    a = prepare_flights(raw, cfg)
    b = prepare_flights(raw.iloc[::-1], cfg)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(raw, snapshot)
    assert a["id"].tolist() == [1, 3, 2, 0]  # sorted by time, ties broken by id
    assert a["split"].tolist() == ["train", "train", "validation", "test"]


def test_modelling_population_excludes_only_unlabelled(cfg):
    raw = make_flights([dict(arr_delay=NAN, air_time=NAN), dict(arr_delay=3.0), dict(arr_delay=99.0)])
    pop = modelling_population(prepare_flights(raw, cfg))
    assert sorted(pop["id"]) == [1, 2]
    assert pop["y"].tolist() == [0, 1]
