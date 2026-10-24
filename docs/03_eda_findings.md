# Stage 3 — Exploratory Data Analysis: Findings

Every number below comes from `reports/eda/eda_results.json` (key shown in brackets), which is
produced by `.venv/bin/python scripts/run_eda.py`. Figures are in `reports/figures/eda/`.

## 0. Data-use policy (what each analysis was allowed to see)
- **Target vs. candidate feature** analyses (hour, weekday, month, carrier, airport, route,
  distance, entity-rate stability, autocorrelation) use the **training split only**
  (2013-01-01..08-31; 217,727 labelled flights, prevalence 0.2637) [`policy`].
- **Full-year** views are limited to label-free quantities (volume, missingness, category mix,
  unseen categories) and to per-period delay prevalence, shown only to describe temporal shift.
- **Disclosure:** the whole-year monthly prevalence was already seen in the Stage 1 audit, before the
  split was confirmed. The split uses calendar-month boundaries chosen for explainability; it was not
  tuned on these values. No validation or test outcome informed any statement in §2.
- Nothing in this stage selects features, thresholds, hyperparameters or models. §4 lists
  *hypotheses* for Stage 4, to be decided on training/validation data only.

## 1. Data quality, class balance and the target

**Q1 Missingness.** Permitted schedule inputs (`sched_dep_time`, `sched_arr_time`, `distance`,
`carrier`, `origin`, `dest`) have **0** missing values [`q11_numeric`]. Missingness exists only in
outcome columns and defines the 9,430 unlabelled rows (Stage 2). The unlabelled share varies by
month from 0.94% (Oct) to 5.37% (Feb), by origin from 1.98% (JFK) to 3.37% (LGA), and by carrier
from 0% (HA) to 9.48% (YV) [`q01_quality`].

*Figure `eda_01_unlabelled_share.png`. Shows that excluded (unlabelled) flights are not spread evenly
across months or carriers. Does not show why they are missing: cancellation and diversion are
inferred from missing fields, not recorded.*

**Q2 Class balance** [`q02_class_balance`]:

| split | rows | labelled | positives | negatives | prevalence |
|---|---:|---:|---:|---:|---:|
| train | 224,910 | 217,727 | 57,423 | 160,304 | 0.2637 |
| validation | 56,463 | 55,628 | 8,731 | 46,897 | 0.1570 |
| test | 55,403 | 53,991 | 13,946 | 40,045 | 0.2583 |

A classifier that always predicts "not delayed" would be correct for 73.6% of training flights while
detecting no delays. Accuracy is therefore not a useful headline metric here.

**Q3 Arrival-delay distribution (training)** [`q03_arr_delay_train`]. Median −4 min, mean 8.98 min,
skewness 3.53. 56.0% of flights arrive early; 26.4% are ≥15 min late, 9.9% ≥60 min, 1.4% ≥180 min.
28 flights are ≥600 min late (max 1,272). The Stage 1 audit showed these extremes reconcile exactly
with clock times, so they are kept: they are positives regardless of size.

*Figure `eda_02_arr_delay_distribution.png`. (a) Bulk of the distribution with the 15-min threshold;
values beyond [−90, 300] are pooled in the edge bins, which causes the spike at 300. (b) Full range on
a log count scale, showing the long right tail. Does not show delay causes; none are recorded.*

## 2. Patterns relevant to candidate features (training split only)

**Q5 Scheduled departure hour** [`q05_hour_train.dep_hour`]. The delay share rises almost
monotonically through the day: 0.108 at 05:00 (n=1,295), 0.124 at 06:00 (n=17,434), 0.260 at 13:00,
0.362 at 17:00, peaking at 0.406 at 21:00 (n=7,125). Hours 22–23 have small n (1,731; 751) and wider
intervals. This is the clearest single pattern in the data. It is consistent with delays building up
during the day, but this analysis cannot establish that mechanism.

*Figure `eda_05_rate_by_dep_hour.png`. Rate with Wilson 95% CIs (top) and the sample size behind each
point (bottom). Does not control for carrier or route mix, which also varies by hour.*

**Q12 Scheduled arrival hour (destination-local)** [`q05_hour_train.arr_hour`]. It follows the
departure-hour pattern (Spearman 0.867 with departure hour [`q12_relationships_train`]). Flights
scheduled to arrive 00:00–03:00 local have high delay shares (0.34–0.47) with wide CIs. No training
flight is scheduled to arrive at 05:00, so that hour is empty in the figure.

**Q6 Calendar** [`q06_calendar_train`]. Day of week: Saturday is lowest (0.185) and Thursday highest
(0.310). Month (Jan–Aug only): from 0.227 (May) to 0.322 (Jul). **Training contains no
September–December flights**, so month cannot be learned for validation/test months. Volume also
varies by weekday (Saturday mean 744.6 flights/day vs. ~950–975 on weekdays [`q10_volume`]).

*Figure `eda_06_rate_by_calendar.png`. Shows weekday and month effects within training. Does not show
holiday effects; no holiday calendar is permitted (D8).*

**Q7 Carrier** [`q07_carrier_train`]. Among carriers with n ≥ 2,000, shares range from 0.196 (VX,
n=3,291) to 0.353 (EV, n=33,476); the large legacy carriers sit near 0.20 (AA 0.203, US 0.207,
DL 0.209). Small carriers have wide intervals: OO has **n=7** (0.714) and is
not a stable estimate; HA (243), AS (481), YV (351) and F9 (447) are also small.

*Figure `eda_07_rate_by_carrier.png`. Carrier labels carry n. Differences reflect carrier and also
route, hour and airport mix; they are not causal carrier effects.*

**Q8 Airports** [`q08_airports_train`]. Origin: EWR 0.285 (n=78,746), JFK 0.259 (73,627),
LGA 0.244 (65,354). Destinations: 102 in training, 8 with n < 30 and 12 with n < 100. Among the 50
destinations with n ≥ 1,000, shares range from 0.204 to 0.414. **64.7%** of destinations lie
outside the 95% band expected if every destination shared the overall rate, so destination
differences are larger than sampling noise alone.

**Q9 Routes** [`q09_routes_train`]. 218 routes in training; median 563 flights per route, 10% have
≤ 30. 22 routes have n < 30 and 36 have n < 100. Among the 73 routes with n ≥ 1,000, shares range
from 0.156 to 0.455; 53.2% of routes lie outside the 95% band.

*Figure `eda_08_funnel_dest_route.png`. Funnel plot: the grey band is the 95% range for a group of
size n if every group had the overall training rate. Points far outside at large n indicate real
heterogeneity; scatter at small n (left) is mostly noise. This is why route/destination rates need a
minimum sample size or shrinkage, and why small routes must not be presented as stable risks.*

**Q12 Distance and numeric relationships** [`q12_relationships_train`]. Delay share by distance
decile is nearly flat and non-monotone (0.230–0.297), lowest for the longest flights (> 1,598 mi).
Distance is almost uncorrelated with departure hour (Spearman −0.03), weekday (0.016) and month
(0.023). **Q11 Outliers:** distance ranges from 17 mi (EWR→LGA, one never-departed flight) to
4,983 mi (JFK→HNL); 715 rows go to HNL/ANC. Scheduled departure hours run 05–23 plus a single
01:xx flight [`q11_numeric`].

*Figure `eda_09_rate_by_distance_arrhour.png`. (a) Distance deciles. (b) Scheduled arrival hour
(destination-local; never combined with departure time into a duration, D12).*

## 3. Time: prevalence, volume, shift, unseen categories

**Q4 Prevalence over time** (all partitions, descriptive) [`q04_prevalence_over_time`]. Monthly
prevalence: Jan 0.235, Feb 0.238, Mar 0.236, Apr 0.288, May 0.227, Jun 0.315, Jul 0.322,
Aug 0.246, **Sep 0.137, Oct 0.176**, Nov 0.181, **Dec 0.336**. Daily prevalence ranges from 0.042 to
0.792 (median 0.205), and 27 days have ≥ 50% of flights delayed. The worst training day is 2013-03-08
(0.792, n=798). Daily delay comes in bursts lasting a few days, consistent with weather or
system-wide events that this dataset does not record.

*Figure `eda_03_daily_prevalence.png`. Daily share (dots) and trailing 7-day mean (line), with split
boundaries. Shows validation (Sep–Oct) is the calmest period of the year while test includes the
volatile December. **The validation set is easier than test**: a threshold or calibration tuned on
validation may transfer poorly. This will be measured in Stage 8, not tuned away.*

**Q10 Volume** [`q10_volume`]. Median daily departures are similar across partitions (964 train,
967 validation, 963 test). The lowest-volume days are 2013-11-28 (634, Thanksgiving) and
2013-11-29 (661). *Figure `eda_04_daily_volume.png`* shows the weekly cycle (Saturday dips).

**Q13 Distribution shift** [`q13_shift`]. Feature mix shifts only slightly. Total variation distance
(TVD), train vs validation / test: carrier 0.021 / 0.013, origin 0.030 / 0.021, departure hour
0.027 / 0.027, weekday 0.031 / 0.031, destination 0.054 / 0.051, route 0.069 / 0.064. **The main
shift is in the outcome rate (seasonality), not the schedule mix.**

*Figure `eda_11_feature_shift.png`. Carrier and departure-hour mix by partition (label-free).*

**Stability of entity delay rates within training** (Jan–Apr vs May–Aug, entities with n ≥ 200 in
both halves). Pearson correlation of rates: carrier 0.725 (13 carriers), destination 0.504 (65),
route 0.467 (123). Origin has only 3 entities, too few to interpret. **Autocorrelation of the daily
system-wide delay share:** 0.356 at lag 1, but only 0.041 at lag 2 and between −0.058 and 0.144 for
lags 2–14. Lag 1 is unavailable at prediction time (D10), so *very recent* system-wide delay carries
little usable signal. Entity-level rates over longer windows are moderately persistent.

*Figure `eda_10_stability_autocorrelation.png`. (a) Entity rates in the first vs second half of
training. (b) Daily-share autocorrelation (lag 1 greyed: unavailable). Does not show that history
features will improve a model; that must be tested on validation.*

**Q14 Rare and unseen categories** [`q14_unseen_rare`]. No carrier or origin is unseen. Destinations
unseen in training: ILM (validation and test) and LEX (test), 52 and 59 rows (≤ 0.11% of each
partition). Unseen routes: 2 in validation (55 rows), 5 in test (63 rows). Rows whose route has
< 30 training flights: 467 (validation), 433 (test). The pipeline must handle unseen and rare
categories explicitly.

## 4. Open questions for Stage 4 (hypotheses, not decisions)
1. **Departure hour** shows a strong, smooth effect. Candidate encodings are cyclic sin/cos (as the
   brief specifies) and/or one-hot hour; to be compared on validation.
2. **Arrival hour** is highly correlated with departure hour (0.867). It may add little beyond
   departure hour and distance; it adds a destination-local "arrives after midnight" signal.
3. **Month** cannot be learned for Sep–Dec. Recommendation: exclude month as a direct input. Seasonal
   level can only enter through permitted historical rates (≤ D−2).
4. **Weekday** effect is clear; include day of week (and possibly a weekend flag).
5. **Carrier / destination / route** need rare-category handling: a minimum training count with an
   "other" bucket for one-hot, and unknown handling for unseen values. Route identity has 218 levels
   with many small ones; a historical route rate with shrinkage may be preferable to one-hot.
6. **Historical features (≤ D−2):** entity-level rates over longer windows (expanding or tens of
   days) look more promising than short recency windows, given low lag ≥ 2 autocorrelation. Window
   lengths and minimum counts must be chosen on training/validation only.
7. **Distance** has a weak, non-monotone relation; keep it as a numeric input for XGBoost and examine
   binning or splines for Logistic Regression only if validation supports it.
8. **Validation/test prevalence mismatch** (0.157 vs 0.258) will affect the 70%-precision threshold
   and calibration transfer; report it explicitly in Stage 8.

## 5. Limitations of this EDA
- Univariate rates are confounded (hour, carrier, route and airport co-vary). Nothing here is causal.
- One year only; seasonal patterns cannot be separated from 2013-specific events.
- No weather data, so the delay bursts in Figure 3 are unexplained.
- Wilson intervals assume independent flights; same-day flights share conditions, so the true
  uncertainty is larger than shown.
