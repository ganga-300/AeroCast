"""Leakage and correctness tests for feature construction (toy data only)."""

import numpy as np
import pandas as pd
import pytest

from conftest import NAN, make_flights
from flight_delay.data.load import prepare_flights
from flight_delay.features.build import build_feature_table, feature_specs
from flight_delay.features.schedule import cyclic_hour
from flight_delay.features.specs import FeatureSpec, validate_specs

CANCELLED = dict(dep_time=NAN, dep_delay=NAN, arr_time=NAN, arr_delay=NAN, air_time=NAN)


@pytest.fixture
def toy_cfg(cfg):
    cfg["features"]["history"] = {"entities": ["carrier", "route"], "entity_windows_days": [1],
                                  "global_windows_days": [1, 2], "min_count": 2, "smoothing_m": 1}
    return cfg


def build(raw, cfg):
    return build_feature_table(prepare_flights(raw, cfg), cfg).set_index("id")


def random_flights(n=400, days=20, seed=0, start_day=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        delay = float(rng.normal(5, 30))
        row = dict(day=int(rng.integers(start_day, start_day + days)),
                   sched_dep_time=int(rng.integers(5, 23)) * 100 + int(rng.integers(0, 60)),
                   carrier=str(rng.choice(["AA", "UA", "B6"])),
                   dest=str(rng.choice(["BOS", "ORD", "MIA", "SFO"])),
                   arr_delay=round(delay), dep_delay=round(delay))
        if rng.random() < 0.05:
            row.update(CANCELLED)
        rows.append(row)
    return make_flights(rows)


HAND = [
    dict(day=1, carrier="A", arr_delay=30.0),
    dict(day=1, carrier="A", arr_delay=40.0),
    dict(day=1, carrier="B", arr_delay=0.0),
    dict(day=1, carrier="B", **CANCELLED),
    dict(day=2, carrier="A", arr_delay=20.0),
    dict(day=2, carrier="B", arr_delay=0.0),
    dict(day=3, carrier="A", arr_delay=0.0),
    dict(day=3, carrier="B", arr_delay=0.0),
    dict(day=4, carrier="A", arr_delay=5.0),
    dict(day=4, carrier="B", arr_delay=5.0),
]


def test_hand_calculated_history(toy_cfg):
    f = build(make_flights(HAND), toy_cfg)
    # Day 3 sees only day 1: labelled n=3 (cancelled row excluded), positives=2.
    d3a, d3b = f.loc[6], f.loc[7]
    assert d3a["hist_all_n_2d"] == 3 and d3a["hist_all_rate_exp"] == pytest.approx(2 / 3)
    assert d3a["hist_all_nodep_share_1d"] == pytest.approx(1 / 4)
    # Carrier A: n=2 >= min_count, shrunk (2 + 1*2/3) / (2 + 1)
    assert d3a["hist_carrier_rate_exp"] == pytest.approx((2 + 2 / 3) / 3)
    # Carrier B: n=1 < min_count -> falls back to the system-wide rate
    assert d3b["hist_carrier_n_1d"] == 1
    assert d3b["hist_carrier_rate_exp"] == pytest.approx(2 / 3)
    # Day 4 sees days 1-2: n=5, positives=3; 1-day window = day 2 only (n=2, pos=1)
    d4a, d4b = f.loc[8], f.loc[9]
    assert d4a["hist_all_rate_exp"] == pytest.approx(3 / 5)
    assert d4a["hist_all_rate_1d"] == pytest.approx(1 / 2) and d4a["hist_all_n_1d"] == 2
    assert d4a["hist_all_rate_2d"] == pytest.approx(3 / 5)
    assert d4a["hist_carrier_rate_exp"] == pytest.approx((3 + 0.6) / 4)
    assert d4b["hist_carrier_rate_exp"] == pytest.approx((0 + 0.6) / 3)
    # Carrier A 1-day window on day 4: only day 2 (n=1 < 2) -> 1-day system-wide prior 0.5
    assert d4a["hist_carrier_n_1d"] == 1 and d4a["hist_carrier_rate_1d"] == pytest.approx(0.5)


def test_no_history_gets_documented_fallback(toy_cfg):
    f = build(make_flights(HAND), toy_cfg)
    early = f[f["flight_date"] <= "2013-01-02"]
    hist_rates = [c for c in f.columns if c.startswith("hist_") and "_rate_" in c]
    assert early[hist_rates].isna().all().all()            # no history -> NaN (imputed in Stage 6)
    assert (early.filter(like="_n_") == 0).all().all()     # counts are zero, not missing


def test_unseen_entity_falls_back_to_prior(toy_cfg):
    raw = make_flights(HAND + [dict(day=4, carrier="NEW", arr_delay=0.0)])
    row = build(raw, toy_cfg).loc[10]
    assert row["hist_carrier_n_1d"] == 0
    assert row["hist_carrier_rate_exp"] == pytest.approx(row["hist_all_rate_exp"])


def test_appending_future_records_does_not_change_earlier_features(cfg):
    base = random_flights(seed=1)
    future = random_flights(n=200, days=5, seed=2, start_day=21)
    future["id"] += len(base)
    before = build(base, cfg)
    after = build(pd.concat([base, future], ignore_index=True), cfg)
    pd.testing.assert_frame_equal(before, after.loc[before.index])


@pytest.mark.parametrize("perturb_day", [5, 10])
def test_own_same_day_and_previous_day_outcomes_cannot_leak(cfg, perturb_day):
    raw = random_flights(seed=3)
    changed = raw.copy()
    m = changed["day"] == perturb_day
    # Flip every post-event field of every flight on that date.
    changed.loc[m, ["arr_delay", "dep_delay"]] = 500.0
    changed.loc[m, "air_time"] = 10.0
    a, b = build(raw, cfg), build(changed, cfg)
    feats = [c for c in a.columns if c not in ("y", "label_status")]
    lag = cfg["prediction"]["history_lag_days"]
    cutoff = pd.Timestamp(2013, 1, perturb_day) + pd.Timedelta(days=lag)
    early = a["flight_date"] < cutoff
    pd.testing.assert_frame_equal(a.loc[early, feats], b.loc[early, feats])
    # Sensitivity check: the perturbation is visible from D + lag onwards.
    assert not a.loc[~early, feats].equals(b.loc[~early, feats])


def test_row_order_does_not_change_features(cfg):
    raw = random_flights(seed=4)
    a = build(raw, cfg)
    b = build(raw.sample(frac=1.0, random_state=7), cfg)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_split_labels_are_not_used_by_feature_builder(cfg):
    df = prepare_flights(random_flights(seed=5), cfg)
    shuffled = df.copy()
    shuffled["split"] = shuffled["split"].sample(frac=1.0, random_state=0).to_numpy()
    a, b = build_feature_table(df, cfg), build_feature_table(shuffled, cfg)
    pd.testing.assert_frame_equal(a.drop(columns="split"), b.drop(columns="split"))


def test_partitions_follow_identical_rules(cfg):
    """A flight's features are the same whether built from the full year or only from the
    rows up to and including its own date (one pass, no split-specific logic)."""
    rows = [dict(month=m, day=d, carrier=c, arr_delay=float((m * 7 + d * 3) % 40 - 10))
            for m in (8, 9, 10, 11) for d in (28, 29, 30) for c in ("AA", "UA")]
    raw = make_flights(rows)
    full = build(raw, cfg)
    prefix = build(raw[raw["month"] <= 9], cfg)
    pd.testing.assert_frame_equal(prefix, full.loc[prefix.index])


def test_cyclic_hour_encoding():
    s, c = cyclic_hour(pd.Series([0.0, 6.0, 12.0, 24.0, 23.99]))
    assert s[0] == pytest.approx(s[3], abs=1e-12) and c[0] == pytest.approx(c[3])
    assert (s[1], c[1]) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert c[2] == pytest.approx(-1.0)
    assert np.hypot(s[4] - s[0], c[4] - c[0]) < 0.01      # 23:59 is next to 00:00


def test_schedule_features_values(cfg):
    f = build(make_flights([dict(day=5, sched_dep_time=2330, sched_arr_time=45,
                                 origin="JFK", dest="LAX", distance=2475)]), cfg).iloc[0]
    assert f["dep_hour"] == 23 and f["arr_hour"] == 0
    assert f["day_of_week"] == 5 and f["is_weekend"] == 1     # 2013-01-05 was a Saturday
    assert f["route"] == "JFK-LAX" and f["distance"] == 2475.0


def test_specs_enforce_contract(cfg):
    validate_specs(feature_specs(cfg))
    bad = FeatureSpec("leak", "numeric", "schedule", ("arr_delay",), "")
    with pytest.raises(ValueError):
        validate_specs([bad])
    with pytest.raises(ValueError):
        validate_specs([FeatureSpec("month", "numeric", "schedule", ("month",), "")])
    with pytest.raises(ValueError):
        validate_specs([FeatureSpec("h", "numeric", "history", ("air_time",), "")])


def test_open_lag_blocks_feature_building(cfg):
    cfg["prediction"]["history_lag_days"] = None
    with pytest.raises(ValueError, match="D10"):
        feature_specs(cfg)


def test_no_expanding_count_features(cfg):
    """Expanding counts are monotone in time (a hidden date index), so they are not inputs."""
    names = [s.name for s in feature_specs(cfg)]
    assert not any(n.endswith("_n_exp") for n in names)
    assert "hist_carrier_n_28d" in names and "hist_carrier_rate_exp" in names
