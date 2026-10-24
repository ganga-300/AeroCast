import numpy as np
import pandas as pd
import pytest

from flight_delay.analysis.stats import (autocorrelation, category_shares, rate_table,
                                         total_variation_distance, unseen_categories,
                                         wilson_interval)


def test_wilson_matches_hand_calculation():
    # p = 0.5, n = 10: centre 0.5, half-width 1.96*sqrt(0.025 + 3.8415/400)/1.38415 = 0.2634
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_edge_cases():
    lo, hi = wilson_interval([0, 3, 0], [5, 3, 0])
    assert lo[0] == pytest.approx(0.0) and 0 < hi[0] < 1   # zero successes: lower bound 0
    assert hi[1] == pytest.approx(1.0) and lo[1] < 1        # all successes: upper bound 1
    assert np.isnan(lo[2]) and np.isnan(hi[2])              # empty group is undefined, not 0


def test_rate_table_ignores_missing_labels():
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b"],
                       "y": pd.array([1, 0, pd.NA, pd.NA, 1], dtype="Int8")})
    t = rate_table(df, "g")
    assert t.loc["a", "n"] == 2 and t.loc["a", "rate"] == 0.5
    assert t.loc["b", "n"] == 1 and t.loc["b", "rate"] == 1.0


def test_tvd_and_shares():
    shares = category_shares(pd.Series(["x", "x", "y", "y"]), pd.Series(["A", "A", "B", "B"]))
    assert total_variation_distance(shares["A"], shares["B"]) == 1.0
    assert total_variation_distance(shares["A"], shares["A"]) == 0.0


def test_unseen_categories():
    r = unseen_categories(pd.Series(["a", "b"]), pd.Series(["a", "c", "c", "d"]))
    assert r["categories"] == ["c", "d"] and r["rows"] == 3 and r["row_share"] == 0.75


def test_autocorrelation_of_alternating_series():
    s = pd.Series([1.0, -1.0] * 10)
    ac = autocorrelation(s, [1, 2])
    assert ac[1] == pytest.approx(-1.0) and ac[2] == pytest.approx(1.0)
