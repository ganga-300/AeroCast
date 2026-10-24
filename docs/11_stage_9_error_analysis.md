# Stage 9 — Descriptive Error Analysis (no further test-label release)

Code: `scripts/run_error_analysis.py`, helpers in `src/flight_delay/analysis/error_analysis.py`
(tests: `tests/test_error_analysis.py`). Results: `reports/error_analysis/stage9_results.json`.
Figures: `reports/figures/stage9/`.

**This is exploratory and descriptive.** Nothing here changes the frozen pipeline:
- no retraining, recalibration, threshold or model selection;
- the frozen evaluation (`docs/10`) is the reported test result and is unchanged.

## 0. Data access and what was possible
| source | used for | restriction |
|---|---|---|
| `reports/final/test_predictions.csv` | test predictions (no labels) | read-only; SHA-256 unchanged |
| `reports/final/test_evaluation.json` | saved **aggregate** test outcomes: monthly metrics and prevalence, overall reliability bins, calibration intercept/slope | read-only; SHA-256 unchanged |
| raw CSV | schedule fields only: `id, year, month, day, sched_dep_time, sched_arr_time, carrier, origin, dest, distance` | read with `usecols`, so no outcome column was loaded; guarded by `assert_label_free_columns` |
| `reports/models/validation_predictions.csv` | validation (development) predictions and labels | not test data |
| frozen artifacts | coefficients, importances; calibrator `transform` on validation | no fitting |

**Never used:**
- test labels, through any route; no second release (the access log still has 1 entry);
- test-period historical-rate features. Although they are legitimate model inputs, they are
  lagged summaries of test outcomes, so analysing them would be a back door to test labels.

**Not possible without another label release, and therefore not done:**
- false-positive / false-negative analysis;
- performance or calibration by carrier, route, airport or hour on test;
- reliability curves per month;
- day-level or holiday-window error;
- errors for unseen categories;
- test permutation importance.

The saved evaluation contains no per-flight outcomes and only an *overall* reliability curve.

All script checks PASS:
- the raw checksum matches;
- only label-free columns were read;
- every prediction row matched a schedule row;
- no prediction anomalies;
- frozen outputs are byte-identical before and after;
- exactly one logged release.

## 1. Observed findings
### 1.1 Prediction distributions and anomalies (label-free)
| period | n | LR mean | LR q10 / q50 / q90 | LR max | share > 0.5 |
|---|---|---|---|---|---|
| validation | 55,628 | 0.157 | 0.065 / 0.144 / 0.271 | 0.507 | 0.0000 |
| Nov | 26,971 | 0.168 | 0.069 / 0.154 / 0.287 | 0.497 | 0.0000 |
| Dec | 27,020 | 0.191 | 0.079 / 0.176 / 0.327 | 0.548 | 0.0009 |

- **No anomalies:** no duplicate ids, no non-finite or out-of-range values, no label column.
- **No extreme probabilities:** the calibrated LR never exceeds 0.548, and almost no flight is
  above 0.5 (23 December flights).
- **XGBoost + isotonic** has only 47 distinct values (a step function), capped at 0.377. Its December
  mean (0.171) barely differs from November (0.168), though more December flights fall in the top
  step (q90 0.260 → 0.377).
- **Raw XGBoost** averages about 0.50, from training-time class weighting (known from Stage 7).

### 1.2 November versus December (saved aggregates)
| | Nov | Dec |
|---|---|---|
| observed prevalence | 0.181 | 0.336 (×1.85) |
| LR mean prediction | 0.168 (×1.14 Dec/Nov) | 0.191 |
| LR calibration-in-the-large | −0.014 | **−0.145** |
| LR AP / ROC AUC | 0.274 / 0.634 | 0.427 / 0.606 |

**The delay rate nearly doubled; the model's predictions rose by 14%.** Across groups with ≥ 500
flights in each month, the December/November ratio of mean predictions is:
- carriers: 1.05–1.34;
- origins: 1.08–1.20;
- departure hours: 1.10–1.18.

So no group's predictions approached the observed doubling. Predictions did rise in late December:
Dec 20–31 mean 0.204 against Dec 1–19 mean 0.182.

### 1.3 Overall reliability (saved bins)
- **Logistic Regression:** every one of the 10 bins is under-predicted, by a nearly uniform
  0.06–0.09 (weighted mean gap 0.079). That is a level offset rather than a distortion confined to
  some risk range.
- **XGBoost:** gaps range from 0.05 to 0.13 (weighted 0.089).

### 1.4 Schedule-mix shift (label-free; total variation distance)
| feature | val vs Nov | val vs Dec | Nov vs Dec |
|---|---|---|---|
| carrier | 0.012 | 0.035 | 0.027 |
| origin | 0.008 | 0.009 | 0.010 |
| destination | 0.042 | 0.075 | 0.048 |
| route | 0.053 | 0.084 | 0.056 |
| departure hour | 0.026 | 0.029 | 0.016 |
| day of week | 0.067 | 0.046 | 0.073 |

- Mean distance: 1,045 (validation), 1,052 (Nov), 1,076 (Dec).
- Labelled flights per day: 912, 899, 872.

**The schedule composition the model sees changed only slightly** between November and December.

### 1.5 Prior-shift consistency check (descriptive; not applied)
The calibrators were fitted where the base rate was 0.157 (validation). If *only* the base rate
changed, with p(features | outcome) unchanged, Bayes' rule implies shifting every prediction's
log-odds by logit(π_month) − logit(0.157) (Saerens et al., 2002):

| model | month | mean pred | mean after pure prior-shift adjustment | saved prevalence | prior-implied log-odds shift | constant shift matching saved prevalence |
|---|---|---|---|---|---|---|
| LR | Nov | 0.168 | 0.191 | 0.181 | 0.172 | 0.099 |
| LR | Dec | 0.191 | 0.372 | 0.336 | 0.997 | 0.822 |
| XGB | Nov | 0.168 | 0.193 | 0.181 | 0.172 | 0.092 |
| XGB | Dec | 0.171 | 0.342 | 0.336 | 0.997 | 0.965 |

*Figure `stage9_distribution_prior_shift.png`: (a) the December prediction distribution is shifted
slightly right; (b) for December the pure prior-shift adjustment moves the mean prediction from
0.19 to 0.37, overshooting the observed 0.34.*

### 1.6 Unseen categories (label-free)
- 61 test flights are on routes never seen in training (57 to destinations ILM and LEX).
- Their mean LR prediction is 0.33, against 0.18 for all other flights. The likely mechanism is the
  shared "infrequent destination" one-hot column (LR coefficient +0.22).
- Their outcomes are not available, so whether this is accurate **cannot be assessed**.

### 1.7 Frozen-model interpretation (no data; caveats below)
- **LR largest coefficients** (standardised inputs):
  - `dep_hour_sin` −0.51, i.e. higher risk later in the day;
  - `hist_all_n_7d` −0.44 and `hist_all_n_28d` +0.26;
  - `carrier_EV` +0.31, `carrier_DL` −0.27;
  - `dest_BOS` −0.31;
  - `is_weekend` −0.28.
- **XGBoost total gain:** `dep_hour` 35%, `hist_all_n_28d` 10%, `arr_hour` 10%, then the
  system-wide and carrier historical rates.

**Caveats:**
- The two volume counts are correlated and have opposite signs, so they must not be interpreted
  individually.
- Coefficients and gain describe the fitted models. They are not causal effects, and they say
  nothing about test errors.

## 2. Hypotheses, consistent with the data but not established
**H1. December's under-prediction is largely a base-rate (prior) shift the day-before inputs barely
register.** Evidence:
- the constant log-odds shift that would match December's saved prevalence (0.82 for LR) is of the
  same order as a pure base-rate change implies (1.00);
- the reliability gap is nearly uniform across risk levels;
- schedule composition hardly changed while the delay rate nearly doubled.

This pattern is what a shift in factors *outside* the model's inputs would produce (for example
weather or holiday-period congestion). The dataset contains neither, so the cause cannot be
identified here.

**H2. The shift is not purely a prior shift.** Evidence:
- ROC AUC fell from validation (0.670 raw) to November (0.634) and December (0.606);
- the pooled test calibration slope is 0.70;
- the pure prior-shift adjustment overshoots December's prevalence (0.372 vs 0.336).

That suggests p(features | outcome) also changed, so the features' relationship with delays was
weaker in December. Part of the pooled slope may also come from combining two months with
different base rates. The saved aggregates cannot separate these.

**H3. Lagged history partly tracks the shift, but too late.** Predictions rose in late December,
when D−2 history began to reflect higher delay rates. With a 2-day lag and smoothed windows, the
model cannot react to sudden same-day disruption. This is consistent with the contract's stated
limitation, but day-level outcomes needed to test it are not available.

## 3. What this analysis does not do or claim
- It does not identify why December was worse, and it makes no causal claims.
- It does not evaluate performance for carriers, routes, airports, hours or unseen categories on
  test (labels unavailable).
- It does not propose or select a threshold, recalibration or replacement model.
- **The prior-shift adjustment is a diagnostic only.** Applying it would require knowing the test
  base rate in advance, which is not available at prediction time.

## 4. Limitations
- The only test outcome information is the aggregates saved by the single frozen evaluation:
  monthly metrics and one overall reliability curve. Monthly reliability curves were not
  pre-registered.
- Validation predictions under the final calibrator are in-sample for that calibrator. They are used
  only to compare distributions.
- The interpretation statements rest on one year of NYC data with no weather information.
- Adding `src/flight_delay/analysis/error_analysis.py` changes the package-code hash, so the FROZEN
  manifest no longer validates. That deliberately blocks any re-run of `scripts/evaluate_test.py`.
  The audit record is preserved by the stored hashes and the commit containing the evaluation.
