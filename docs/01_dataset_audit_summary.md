# Stage 1 — Dataset Audit: Interpreted Summary

Source of every number: `reports/audit/dataset_audit.json`, produced by
`.venv/bin/python scripts/audit_dataset.py`. Follow-up checks noted as *(ad-hoc check)* were run
interactively during the audit and will be folded into scripted EDA in Stage 3.

## 1. What the file actually is
- 336,776 rows × 21 columns; one calendar year, **2013-01-01 to 2013-12-31**, all 365 days present
  (634–1,014 flights/day; minimum on 2013-11-28, i.e. US Thanksgiving).
- **Origins are only EWR (120,835), JFK (111,279) and LGA (104,662).** This is a dataset of flights
  *departing the three New York City airports*, not all US domestic flights. The schema and values
  match the public `nycflights13` dataset (an `id` column has been added). The project brief's
  "US flight records from 2013" is therefore accurate only in this narrower sense; all conclusions
  must be scoped to NYC departures.
- Columns: `id, year, month, day, dep_time, sched_dep_time, dep_delay, arr_time, sched_arr_time,
  arr_delay, carrier, flight, tailnum, origin, dest, air_time, distance, hour, minute, time_hour, name`.
- **Not present:** cancellation flag, diversion flag, delay-cause columns, taxi times, wheels-off/on,
  scheduled elapsed time, time-zone information, weather.

## 2. Data quality (measured)
| Check | Result |
|---|---|
| Full-row duplicates / duplicates ignoring `id` | 0 / 0 |
| `id` unique and equal to row index | yes / yes |
| Duplicate (date, carrier, flight, origin) | 0 |
| Duplicate (date, tailnum, sched_dep_time) | 31 — *(ad-hoc check)* each pair is one operated flight plus one non-departed flight on the same tail, consistent with an aircraft swap; not true duplicates |
| Invalid dates | 0 |
| HHMM clock fields with minutes ≥ 60 or hour > 24 | 0 in all four clock fields |
| Clock value `2400` | `dep_time` 29, `arr_time` 150 (midnight encoded as 2400) |
| `hour*100+minute == sched_dep_time` | 0 mismatches |
| `time_hour == date + hour` | 0 mismatches |
| `dep_delay` = `dep_time − sched_dep_time` (allowing ±1 day wrap) | 0 unexplained; 1,207 need a +1 day wrap (departed after midnight) |
| `arr_delay` = `arr_time − sched_arr_time` (allowing ±1 day wrap) | 0 unexplained; 6,518 need +1 day, 777 need −1 day |
| Carrier ↔ name mapping | 16 carriers, 16 names, 1:1 |
| Routes with >1 distance | 2 (EWR–EGE 1725/1726, JFK–EGE 1746/1747; 1-mile discrepancies) |
| **File row order** | **Not chronological.** Month blocks appear in order `1,10,11,12,2,3,…,9` (lexicographic). Row position must never be used as time. |

Because the recorded delays reconcile exactly with clock-time differences, the delay columns are
internally consistent; extreme delays (max `dep_delay` 1,301 min, max `arr_delay` 1,272 min) are
plausible long delays rather than encoding errors. 40 flights have `dep_delay ≥ 600` min.

### Anomalies flagged (not removed)
- `EWR→LGA`, 1 row (2013-07-27), distance 17 mi, never departed.
- `LGA→LEX`, 1 row; 12 destinations have < 100 flights in the year.
- 352 operated flights have implied airborne speed < 150 mph (mostly very short routes, median 96 mi);
  2 exceed 650 mph. `air_time` is a post-flight outcome and will not be a feature, so this only
  matters for EDA.
- `sched_arr_time − sched_dep_time` is negative for 6,078 rows. Scheduled arrival is in
  destination-local time and departure in NYC-local time, and **no time-zone data is supplied**, so
  a true scheduled duration cannot be computed from this file alone.

## 3. Missing outcomes (no explicit cancellation/diversion flags)
Pattern over `[dep_time, dep_delay, arr_time, arr_delay, air_time]`:

| Pattern | Rows | Interpretation (inference, not a flag in the data) |
|---|---:|---|
| all present | 327,346 | operated, arrival delay known |
| all missing | 8,255 | never departed → **cancellation proxy** |
| departed, no `arr_time` | 458 | departed but no recorded arrival (diverted/returned — cannot tell) |
| `arr_time` present, no `arr_delay`/`air_time` | 717 | **diversion proxy** (arrived, but delay vs schedule not recorded) |

- Total without a label: **9,430 (2.80%)**.
- All 2,512 missing `tailnum` values are among the never-departed rows.
- Missingness is **not random**: unlabelled share by carrier ranges from 0% (HA) to 9.48% (YV)
  and 6.32% (9E); never-departed count by month ranges from 233 (Nov) to 1,261 (Feb).
  Excluding these rows therefore changes the population toward carriers/months with fewer disruptions.

## 4. Target feasibility
`arr_delay` is integer-valued, internally consistent and present for 327,346 rows, so a binary
target `arr_delay ≥ 15` is constructible for those rows.

| Quantity | Value |
|---|---|
| Labelled rows | 327,346 |
| Positive (`arr_delay ≥ 15`) | 80,100 |
| Negative | 247,246 |
| Prevalence | 0.2447 |
| Rows exactly at 15 / at 14 | 2,470 / 2,642 (boundary convention matters; `≥ 15` is used) |
| Early arrivals (`arr_delay < 0`) | 188,933 (negative class) |

Prevalence by month varies strongly: Sep 0.137, Oct 0.176, Nov 0.181 vs Jun 0.315, Jul 0.322,
Dec 0.336. **Strong seasonality** means any chronological split will have prevalence shift between
partitions (see decision D5).

## 5. Implications for later stages (design notes, not yet decisions)
- Must sort by a reconstructed scheduled-departure timestamp before any historical feature.
- `month` cannot be a meaningful learned feature if train covers only part of the year (test months
  would be unseen); calendar effects must be handled carefully.
- Tail-number history is only partial: an aircraft's inbound flights to NYC are not in the data.
- Scheduled duration needs time-zone data (not supplied); `distance` is available as a proxy.
- `time_hour` is a local wall-clock string with no offset; DST handling must be documented if
  sub-day timing is used (no scheduled departures exist between 00:00–01:00 local).
