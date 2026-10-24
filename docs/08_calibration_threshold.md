# Stage 8 — Calibration and Operating-Threshold Selection (validation only)

Code: `src/flight_delay/evaluation/calibration.py`, `src/flight_delay/evaluation/threshold.py`.
Run: `.venv/bin/python scripts/calibrate_and_threshold.py` (≈ 15 s). Outputs:
`reports/calibration/stage8_results.json`, `reports/calibration/frozen_decisions_proposal.json`,
`reports/figures/stage8/*.png`, `models/stage8_*.joblib` (git-ignored). Tests:
`tests/test_calibration_threshold.py`.

**Boundaries kept.**
- Test rows were never scored and no test statistic was computed.
- No frozen manifest exists: the test-label lock is intact.
- The model configurations are the Stage 7 selections, refitted on Jan–Aug only (no search, no
  train+validation refit).
- The calibration and threshold rules are exactly D11 and D7, as codified in
  `configs/config.yaml` (`calibration`, `operating_point`).

**All numbers are validation development estimates**, and optimistic because validation was used for
model selection, early stopping, calibration selection and threshold selection.

## 1. Procedure (as confirmed)
1. Refit Logistic Regression (`logistic_regression_00`: C = 0.01, L2, lbfgs, unweighted; D29) and
   XGBoost (`xgboost_01`, comparison). Validation predictions match the recorded Stage 7
   predictions (max abs diff < 1e-9).
2. Calibration choice: none / sigmoid (Platt on log-odds) / isotonic. Fit on September, log loss on
   October, and the reverse. Select the lowest **mean held-out log loss**; within 1e-6, prefer
   the simpler method.
3. Threshold (D7, D11 §4.1):
   - use predictions the calibrator never saw: raw if the choice is *none*, otherwise
     cross-fitted (September rows calibrated by the October fit, and October rows by the
     September fit);
   - among thresholds with precision ≥ 0.70, take the highest recall, ties to the highest
     threshold;
   - if none qualifies, report infeasible.
4. Final calibrator: the selected method refitted on all validation predictions (the one that
   would be applied to test).

## 2. Calibration comparison (validation; 55,628 flights, prevalence 0.157)

**Logistic Regression (primary).**

| method | Oct log loss (fitted on Sep) | Sep log loss (fitted on Oct) | **mean held-out log loss** | mean held-out Brier |
|---|---|---|---|---|
| none | 0.44701 | 0.40875 | 0.42788 | 0.13159 |
| **sigmoid** | 0.44257 | 0.38516 | **0.41387** | 0.12666 |
| isotonic | 0.44771 | 0.38936 | 0.41854 | 0.12741 |

- **Selected: sigmoid.** The margin over isotonic is 0.0047 and over none 0.0140, so no tie-break
  was needed.
- Final calibrator (all validation): q = σ(1.0096 · logit(p) − 0.4961). The slope is about 1:
  Logistic Regression's *ranking spread* transfers, and the calibration mainly corrects the level.
  Raw mean prediction is 0.231 against 0.157 observed, because the delay rate fell from summer
  to autumn.

**XGBoost (comparison).**

| method | Oct (fitted on Sep) | Sep (fitted on Oct) | **mean held-out log loss** | Brier |
|---|---|---|---|---|
| none | 0.67668 | 0.66092 | 0.66880 | 0.23839 |
| sigmoid | 0.45127 | 0.38533 | 0.41830 | 0.12796 |
| **isotonic** | 0.45108 | 0.38457 | **0.41782** | 0.12778 |

**Selected: isotonic** (margin over sigmoid 0.00048, larger than the 1e-6 tolerance). Calibration
fixes XGBoost's distorted scale, which came from class weighting (raw mean prediction 0.488).

*Figure `stage8_reliability.png`: reliability curves of out-of-fold predictions. Uncalibrated
Logistic Regression over-predicts at every level on validation; sigmoid and isotonic lie close to
the diagonal. Calibration is assessed on validation only. Test calibration is unknown and may
differ (test prevalence is not used here).*

## 3. Threshold selection for 70% precision: infeasible
| model | threshold-eligible predictions | result | best attainable precision |
|---|---|---|---|
| **Logistic Regression** | cross-fitted sigmoid | **infeasible** | 0.313 (threshold 0.325, flags 1,994 flights, recall 0.072) |
| XGBoost | cross-fitted isotonic | **infeasible** | 0.377 (flags 657 flights, recall 0.028) |

**No operating threshold meets the 70% precision requirement on validation, for either model.**
The requirement was not relaxed and no threshold was manufactured.
- Recall at 70% precision is **0.0**.
- Its day-block bootstrap, with the threshold re-selected in every resample, gives point 0.0 and
  95% interval [0.000, 0.0005] for Logistic Regression and [0, 0] for XGBoost.
- By month (Logistic Regression, eligible predictions): best precision 0.272 in September. In
  October a precision of 1.0 is reached only by the top few flights (recall 0.003).

*Figure `stage8_pr_roc_validation.png`: the precision–recall curves stay far below the 0.70 line
(Logistic Regression peaks near 0.31). Figure `stage8_threshold_sweep_lr.png`: precision never
approaches 0.70 at any threshold with non-trivial recall.*

### 3.1 Diagnostic, not used for any decision: raw scores and the effect of cross-fitting
- On **raw** Logistic Regression probabilities, a threshold reaching 70% precision exists: the top
  **7 validation flights, 5 of them delayed** (precision 0.714, recall 0.0006).
- It is fragile. In the day-block bootstrap it is reachable in only **60%** of resamples.
  Operationally it would flag about one flight a week.
- **Why cross-fitting removes it.** September rows are calibrated with the October-fitted map and
  vice versa. The two maps shift the months differently, which reorders flights between months in
  the pooled ranking. This lowers pooled ranking quality: Average Precision 0.271 raw → 0.247
  cross-fitted; ROC AUC 0.670 → 0.659.
- For a *single* strictly increasing calibrator, thresholding calibrated or raw scores gives the
  same set of precision/recall pairs. So a threshold chosen on raw scores and translated through
  the final calibrator would avoid this artefact, without the calibrator seeing those labels in a
  way that biases precision or recall. That is **not** what D11 specifies, and it would not
  change the conclusion: 70% precision is reachable only for a handful of flights, unstably.

**XGBoost raw scores:** best precision 0.377, so the target is infeasible either way (reachable in
0.5% of resamples).

## 4. Threshold-free validation metrics of the eligible predictions
| model | AP [95% CI] | ROC AUC [95% CI] | Brier | log loss | mean predicted P |
|---|---|---|---|---|---|
| Logistic Regression (cross-fitted sigmoid) | 0.247 [0.191, 0.300] | 0.659 [0.628, 0.686] | 0.1270 | 0.4147 | 0.156 |
| XGBoost (cross-fitted isotonic) | 0.228 [0.184, 0.267] | 0.643 [0.615, 0.669] | 0.1282 | 0.4188 | 0.158 |
| Prior baseline (Stage 7) | 0.157 | 0.500 | 0.1437 | 0.4673 | 0.264 |

## 5. What this means
- The day-before model ranks flights better than chance (validation AP about 1.6–1.7× the
  prevalence; ROC AUC about 0.66–0.67). Calibration makes its probabilities usable as risk
  estimates on validation.
- **It cannot identify a meaningful set of flights with ≥ 70% probability of being delayed.** The
  highest-risk validation flights are delayed about 30% of the time (two to three times the base
  rate), not 70%. This is consistent with the contract's limitations: without same-day weather,
  congestion or aircraft-rotation information, delay is largely unpredictable a day ahead.
- **Do not claim that the 70% precision target is achieved.**

## 6. Checks and tests
**Script checks (8/8 PASS):**
- the calibration folds partition validation;
- both refits reproduce the Stage 7 predictions;
- threshold-eligible predictions are out-of-fold, verified on real data: flipping September
  labels leaves September's eligible predictions unchanged;
- only validation rows are used;
- the test lock is intact;
- the raw file is unchanged.

**Unit tests (11):**
- hand-calculated threshold table and selection;
- recall ties going to the highest threshold;
- exact-boundary precision (7/10) counting as meeting the target;
- tied scores flagged together;
- infeasible cases reported without a threshold;
- confusion matrix, with undefined precision returned as None, not 0;
- Platt recovering a known map (a = 2, b = −1);
- isotonic monotone and within [0, 1];
- cross-fit independence;
- the simpler-method tie-break;
- per-fold reporting.

**Mutation check.** Three planted violations are each caught by a failing test:
- in-sample calibration instead of cross-fitting;
- the tie-break preferring the more complex method;
- recall ties broken towards the lowest threshold.

## 7. Limitations
- Calibration was fitted and judged on September–October, the calmest months (prevalence 0.137
  and 0.176). A calibrator fitted there may under-predict in busier periods. This is unknown until
  the final evaluation and cannot be tuned away.
- Precision at the top of the ranking rests on very few flights, so its estimates are highly
  variable (§3.1).
- The validation estimates are optimistic (D21).
- The final calibrator is fitted on all validation predictions while the eligible predictions are
  cross-fitted. Because no threshold exists, the scale-mismatch measurement (D11 §4.1 step 3) does
  not apply.

## 8. Decisions requiring approval (see `docs/decision_register.md`)
- **D32: what to freeze for the operating point**, given that 70% precision is infeasible.
- **D33: whether to record the raw-score / cross-fitting observation** as a documented limitation
  only (recommended), or to revise D11 before test (not recommended at this stage: it would not
  change feasibility and would be a validation-informed rule change).
- **Freezing.** `reports/calibration/frozen_decisions_proposal.json` lists the model,
  preprocessing, hyperparameters, calibrator and threshold status. It becomes the frozen manifest
  only on your approval, at the start of the final-evaluation stage.
