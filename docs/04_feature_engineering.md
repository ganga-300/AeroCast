# Stage 4 — Leakage-Safe Feature Engineering

Code: `src/flight_delay/features/` (`specs.py`, `schedule.py`, `history.py`, `build.py`).
Real-data validation: `.venv/bin/python scripts/build_features.py` →
`reports/features/feature_validation.json`. Generated dictionary of all features:
[`docs/feature_dictionary.md`](feature_dictionary.md). Tests: `tests/test_features.py`.

No model has been trained. Encoding, scaling and imputation belong to Stage 6 (fitted on
training data only).

## 1. Final feature set (31 features)
Every feature declares its raw source columns. `validate_specs` refuses a schedule feature that
reads a non-permitted column, and a historical feature that aggregates a non-eligible one.

### 1.1 Schedule features (13): describe the flight itself, from the published schedule
| feature | kind | definition |
|---|---|---|
| `dep_hour` | numeric | scheduled departure hour (NYC local) |
| `dep_hour_sin`, `dep_hour_cos` | numeric | sin/cos(2π·h/24), h = scheduled departure in fractional hours |
| `arr_hour` | numeric | scheduled arrival hour (destination local) |
| `arr_hour_sin`, `arr_hour_cos` | numeric | same cyclic encoding of scheduled arrival |
| `day_of_week` | categorical | 0 = Monday … 6 = Sunday, from the scheduled flight date |
| `is_weekend` | numeric | Saturday/Sunday indicator |
| `distance` | numeric | route distance (miles), the permitted proxy for scheduled duration (D12) |
| `carrier`, `origin`, `dest`, `route` | categorical | identifiers; `route` = `origin-dest` |

No scheduled duration is computed: arrival and departure clocks are in different, unknown time
zones (D12). `arr_hour` is used only as a time of day.

### 1.2 Historical features (18): outcomes of other, earlier flights
For a flight on date D, every aggregate uses only flights with **flight date ≤ D−2** (D10).

| family | features | window |
|---|---|---|
| System-wide delay rate | `hist_all_rate_{exp,7d,28d}` | expanding; 7 and 28 days ending at D−2 |
| System-wide volume | `hist_all_n_{7d,28d}` | labelled flights in the window |
| System-wide disruption | `hist_all_nodep_share_7d` | share of scheduled flights that never departed |
| Entity delay rate | `hist_{carrier,origin,dest,route}_rate_{exp,28d}` | expanding; 28 days |
| Entity volume | `hist_{carrier,origin,dest,route}_n_28d` | 28 days |

Documentation for each historical feature, as the brief requires:

| item | rule |
|---|---|
| Entity | system-wide, carrier, origin, destination, route |
| Outcome aggregated | `y` (arrival delay ≥ 15) over labelled flights; never-departed share over all scheduled flights |
| Lookback window | expanding (from 2013-01-01), 28 days, 7 days (system-wide only) |
| Time cutoff | flight date ≤ D−2; verified that every such outcome was complete before 00:00 on D (Stage 2) |
| Unlabelled flights | contribute to no rate denominator (never counted as negatives) |
| Minimum sample | `min_count = 20` labelled outcomes in the window |
| Fallback | fewer than 20 → the system-wide rate for the same window; no history at all → NaN |
| Shrinkage | n ≥ 20 → (positives + m·prior) / (n + m), m = 20, prior = system-wide rate of the same window |
| Available at prediction time? | yes, by construction and verified on real data (§3) |

The entity rate is an empirical-Bayes-style shrunk estimate. Small entities are pulled towards the
system-wide rate, so a route with a handful of flights is not treated as a stable risk (EDA Q8/Q9).
`min_count` and `m` were fixed a priori. They were not tuned on validation or test.

**History versus fitting data (contract §4.5).** A validation or test flight's history includes
outcomes from earlier dates, which may themselves be in validation or test (e.g. a test flight on
2013-12-10 uses outcomes up to 2013-12-08). Those outcomes would genuinely be known when
predicting. This does not train the model on validation/test labels; it only mirrors the
information a deployed system would hold.

### 1.3 Excluded
| candidate | reason |
|---|---|
| `month` | Training covers Jan–Aug, so Sep–Dec values are unseen (D15) |
| expanding-window counts (`*_n_exp`) | Monotone in time: in a first build, `hist_all_n_exp` ranged 0–216,101 in training but 217,055–271,470 in validation and 272,439–325,634 in test, a hidden date index with no overlap (D17) |
| `tailnum`, `flight`, post-event fields | Prohibited or unused by the contract (Stage 2) |

## 2. Construction method (why it cannot leak)
`history.py` aggregates outcomes per (entity, flight date), lays daily totals on a contiguous
calendar, and takes cumulative sums. The value for date D is `cumsum(D−2)` (expanding) or
`cumsum(D−2) − cumsum(D−2−w)` (w-day window). Consequences:
- The flight's own outcome and every outcome from D and D−1 are excluded by construction.
- The result depends only on dates and keys, so CSV row order cannot matter.
- Appending later dates cannot change earlier values (earlier cumulative sums are untouched).
- The builder never reads the `split` column, so train, validation and test follow one rule.

## 3. Verification

### 3.1 Unit tests (toy data; `tests/test_features.py`, 14 tests)
| requirement (brief §4.3) | test |
|---|---|
| 1 Future records cannot influence earlier features | `test_appending_future_records_does_not_change_earlier_features` |
| 2 A flight's own label cannot influence its features | `test_own_same_day_and_previous_day_outcomes_cannot_leak` (all post-event fields on a date changed; features before D+2 identical, and changes do appear from D+2 onwards, proving the test is sensitive) |
| 3 Row reordering leaves features unchanged | `test_row_order_does_not_change_features` |
| 4 No history → documented fallback | `test_no_history_gets_documented_fallback`, `test_unseen_entity_falls_back_to_prior` |
| 5 Minimum sample requirement | `test_hand_calculated_history` (carrier with n=1 < 2 gets the prior exactly) |
| 6 Appending future data | same as 1 |
| 7 Same rules across partitions | `test_partitions_follow_identical_rules`, `test_split_labels_are_not_used_by_feature_builder` |
| Hand-calculated toy example | `test_hand_calculated_history` (rates, windows, shrinkage, cancelled flight excluded from denominators, computed by hand in the test comments) |
| Hidden time index | `test_no_expanding_count_features` |

**Mutation check.** I deliberately planted three bugs and confirmed the tests catch each, then
restored the code:
- cutoff off by one day (`shift(lag−1)`): 4 tests failed;
- flight's own date included (`shift(0)`): 5 failed;
- missing labels counted as negatives: 1 failed.

### 3.2 Real-data checks (`scripts/build_features.py`, all PASS)
| check | result |
|---|---|
| Independent brute-force recomputation (direct date filtering) of all 18 historical features for 300 random labelled flights | 0 mismatches (tolerance 1e-12); sample 199 train / 53 validation / 48 test |
| All outcomes on 2013-03-08, 2013-09-15, 2013-11-20 set to 500 min | features before date + 2 unchanged; later features change (979, 900, 977 rows perturbed) |
| Train+validation features rebuilt without any test-period rows | identical (281,373 rows compared) |
| Raw rows shuffled | identical |
| Missing history | exactly the 1,785 rows on 2013-01-01/02 (1,759 of them labelled, all in training) |
| Schedule features missing | 0 |
| Any numeric feature with validation/test range disjoint from training | none |
| Raw file checksum | unchanged |

### 3.3 Descriptive summaries (modelling population)
- Share of rows where the 28-day entity count is below `min_count`, so the fallback is used
  (train / validation / test): carrier 0.86% / 0.03% / 0.01%; origin 0.81% / 0% / 0%;
  destination 1.63% / 0.24% / 0.40%; route 2.87% / 0.95% / 1.06%. Training is higher because
  of the January warm-up, when windows are not yet filled.
- Categories unseen in training: destination ILM (validation 51 rows; test with LEX, 57 rows) and
  routes EWR-SBN, LGA-ILM (validation, 54 rows), plus JFK-JAC, JFK-STL, LGA-LEX (test, 61 rows in
  total). Their historical features still work: they fall back to system-wide rates and then use
  their own history once it accumulates.
- No pair of numeric features has |Spearman| ≥ 0.9 on training (after D17).

## 4. Open items handed to Stage 6 (proposed, to be decided on training/validation only)
1. **Cold start (D18).** The 1,759 labelled training flights on Jan 1–2 have no history (NaN rates).
   Proposal: keep them; impute with training statistics in the Stage 6 pipeline (Logistic
   Regression) and use XGBoost's native missing-value handling. Dropping them would be the
   alternative.
2. **Categorical encoding (D19).** Logistic Regression: one-hot for `carrier`, `origin`, `dest`,
   `day_of_week`, with categories below a minimum training frequency grouped as "other" and
   unknown categories mapped to "other". `route` (218 levels, many small) is represented mainly by
   its historical rates. Final handling for XGBoost is decided in Stage 6.
3. **Redundant encodings.** `dep_hour` and its sin/cos versions are deliberately all present. The
   preprocessing for each model will select a consistent subset (e.g. cyclic for Logistic
   Regression); any choice beyond that is made on validation.
