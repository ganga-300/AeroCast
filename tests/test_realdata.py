"""Checks on the actual supplied file; skipped when data/raw/flights.csv is absent."""

import pytest

from flight_delay.config import load_config, resolve_path
from flight_delay.data.load import file_sha256, load_flights

CFG = load_config()
RAW = resolve_path(CFG["data"]["raw_path"])
pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not RAW.exists(), reason="raw data not present")]


def test_real_split_is_chronological_and_raw_unchanged():
    df = load_flights(CFG)
    assert file_sha256(RAW) == CFG["data"]["raw_sha256"]
    tr, va, te = (df.loc[df.split == s, "flight_date"] for s in ("train", "validation", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert str(tr.min().date()) == "2013-01-01" and str(te.max().date()) == "2013-12-31"
    assert df["y"].isna().sum() == (df["label_status"] != "labelled").sum()
