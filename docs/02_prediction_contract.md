# Prediction Contract and Leakage Policy (Stage 2)

This contract governs EDA, feature engineering, training, evaluation and the report. Its
machine-enforced form is `configs/config.yaml` + `src/flight_delay/contract.py`; consistency with
the real data is verified by `scripts/check_contract.py` (output:
`reports/contract/contract_check.json`). Decision IDs refer to `docs/decision_register.md`.

## 1. Prediction task
**Question answered:** given the published schedule and legitimately available history, will this
flight, *if it operates and its arrival delay is recorded*, arrive at least 15 minutes late?

| Element | Definition | Status |
|---|---|---|
| Prediction timestamp | All flights on scheduled flight date D are scored at the **end of D−1** (00:00 NYC local time on D). No same-day information is used. | CONFIRMED (D3) |
| Unit of prediction | One row = one scheduled departure from EWR/JFK/LGA | FACT |
| Target | `y = 1` if `arr_delay ≥ 15`; `y = 0` if `arr_delay < 15` (early arrivals included); `y = <NA>` if `arr_delay` is missing | CONFIRMED (D6) |
| Eligible population | Rows with a recorded `arr_delay` (`label_status == "labelled"`) | CONFIRMED (D4) |
| Operating requirement | Minimum precision 0.70; threshold maximises recall subject to it on validation; ties → highest threshold; frozen before test; infeasibility reported | CONFIRMED (D7) |
| External data | None (no weather, TZ database, holiday calendar or other dataset) | CONFIRMED (D8) |

## 2. Population and exclusions (measured by `check_contract.py`)
Missing labels are represented as `<NA>` with an explicit `label_status`; they are **never**
counted as negatives (verified: negatives = 247,246 = audit count; missing = 9,430).

| label_status | Meaning (inferred; no flags in the data) | train | validation | test | total |
|---|---|---:|---:|---:|---:|
| `labelled` | arrival delay recorded → in population | 217,727 | 55,628 | 53,991 | 327,346 |
| `no_departure` | never departed (cancellation proxy) | 6,309 | 688 | 1,258 | 8,255 |
| `departed_no_arrival` | departed, no arrival time | 341 | 63 | 54 | 458 |
| `arrived_no_delay` | arrived, no delay recorded (diversion proxy) | 533 | 84 | 100 | 717 |

**Limitation (non-random missingness):** the unlabelled share ranges from 0% (HA) to 9.48% (YV) by
carrier and varies by month. Excluded flights are disproportionately disrupted operations, so the
model describes delay *conditional on the flight operating and being recorded*, and its error
rates do not apply to cancellations or diversions.

## 3. Permitted and prohibited inputs
Full table (generated from code): `docs/feature_availability.md`.

**Permitted raw sources for the flight being predicted** (known from the published schedule):
`year, month, day` (→ flight date, day of week, weekend), `sched_dep_time`/`hour`/`minute`,
`sched_arr_time` (destination-local hour only), `carrier`, `origin`, `dest`, `distance`.

**Prohibited as inputs for the flight being predicted:**

| Field | Reason |
|---|---|
| `dep_time`, `dep_delay` | Actual departure: happens on D, after the prediction time |
| `arr_time`, `arr_delay` | Actual arrival / target source |
| `air_time` | Actual airborne time: post-event |
| `tailnum` | Aircraft assignment at end of D−1 not verifiable; same-day swaps observed (D13) |
| `id` | Export artefact encoding file order, not time |

**Available but not used:** `flight` (high-cardinality identifier, no defensible generalisation),
`time_hour` (redundant with date + hour), `name` (1:1 with `carrier`).

**Scheduled duration:** never computed as `sched_arr_time − sched_dep_time`. The two are in
different local time zones and no time-zone data is permitted (D12). `distance` is the proxy.

Enforcement: feature code must declare its raw sources; `assert_feature_sources_permitted` raises
on any non-permitted column, and `assert_registry_covers` fails if the schema gains or loses a column.

## 4. Historical aggregation rules
Historical features summarise outcomes of *other, earlier* flights (e.g. delay rate by carrier,
origin, destination, route). The rules are:

1. **Eligible outcome columns:** `arr_delay`/`y`, `dep_delay`, and the cancellation proxy derived
   from `dep_time` (`history_eligible` in the registry). `air_time` and `arr_time` are not used.
2. **Time cutoff — CONFIRMED (D10): a flight on date D may use outcomes only of flights with
   flight date ≤ D−2** (`history_lag_days: 2`). Principle: a historical outcome may be used for a
   prediction at time *t* only if it was available before *t*. Measured evidence:
   - With a cutoff of "flight date < D" (lag 1), **19,303 labelled flights (5.90%)** have a
     landing lower bound (actual departure + `air_time`) at or after 00:00 on D, so their outcome
     was *not* known at the prediction time. 53.8% of them are positives (all partitions, as
     computed in Stage 2; 54.4% when test rows are excluded), so this leakage is correlated with
     the target.
   - With "flight date ≤ D−2" (lag 2), **0** flights violate the rule, even with a 180-minute
     allowance for unrecorded taxi-in (latest landing lower bound: 41.55 h after the start of the
     flight date).
   - `scripts/check_contract.py` re-verifies on every run that the configured lag has zero
     violations; `require_history_lag()` refuses to build historical features if the value is unset.
   - Consequence: the cancellation proxy of a D−2 flight is also known by then (it never departed),
     so the same cutoff applies to all history-eligible outcomes.
3. **Never** use the current flight's own outcome, same-day outcomes, future records,
   full-dataset aggregates or row position.
4. **Ordering:** aggregation is computed by date, never by CSV row order. `prepare_flights` sorts by
   (`sched_dep_ts`, `id`); verified that shuffling raw rows yields identical output.
5. **History vs. fitting data:** a validation or test flight's history may include earlier
   training-period outcomes (they were available then). This never permits fitting the model on
   validation/test labels.
6. **Small samples:** every rate uses `min_count = 20` with a fallback to the system-wide rate,
   plus shrinkage (m = 20); see `docs/04_feature_engineering.md`.
7. **Required tests:** implemented in `tests/test_features.py` and `scripts/build_features.py`
   (future-append, own/same-day label, row order, fallback, partition consistency, brute force).

## 5. Chronological split (CONFIRMED, D5)
Inclusive dates on the verified flight date; `parse_boundaries` rejects overlap or misordering;
`assign_split` raises for dates outside all ranges; `assert_chronological` checks that every date
maps to one partition and partitions are strictly ordered.

| split | dates | days | all rows | unlabelled | labelled | positives | negatives | prevalence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2013-01-01 – 2013-08-31 | 243 | 224,910 | 7,183 | 217,727 | 57,423 | 160,304 | 0.2637 |
| validation | 2013-09-01 – 2013-10-31 | 61 | 56,463 | 835 | 55,628 | 8,731 | 46,897 | 0.1570 |
| test | 2013-11-01 – 2013-12-31 | 61 | 55,403 | 1,412 | 53,991 | 13,946 | 40,045 | 0.2583 |

Validation is used for hyperparameters, feature choices, threshold and calibration selection. Test
is evaluated once with frozen choices. This is a **retrospective chronological holdout**, not a
prospective deployment test.

## 6. Known limitations
- Single year (2013), NYC departures only; no year-over-year seasonality can be learned. Test months
  (Nov–Dec) never appear in training, so `month` cannot be learned as a category for them.
- Validation prevalence (0.157) is much lower than test (0.258). Thresholds and calibration chosen
  on validation may not transfer; this will be measured and reported, not tuned away.
- The day-before horizon excludes same-day weather and congestion, the strongest delay drivers;
  modest discrimination is expected.
- No weather, time zone, holiday or aircraft-rotation information (D8, D13).
- Cancellation/diversion status is inferred from missingness, not recorded.
- Timestamps are naive NYC wall-clock times; the two DST nights can shift times by up to 1 h.
- Results are associations; nothing here supports causal claims about delay drivers.
