# Stage 7 — Baseline Models (validation only)

Code: `src/flight_delay/models.py`, `src/flight_delay/evaluation/metrics.py`. Run:
`.venv/bin/python scripts/train_baselines.py` (≈ 80 s). Outputs: `reports/models/baseline_runs.json`
(all 22 fitted configurations), `reports/models/baseline_selection.json` (selection, intervals,
checks, reproducibility), `reports/models/validation_predictions.csv`, and `models/*.joblib`
(git-ignored). Tests: `tests/test_models.py`, `tests/test_metrics.py`.

**Boundaries kept.**
- No test label, prediction or statistic was computed: test rows were never scored.
- No threshold was selected, nothing was calibrated, and no threshold-based metric was computed.
- No refit on train+validation (D20).
- Feature definitions, split, contract and preprocessing are unchanged.

**These validation numbers are development estimates, not a final evaluation.** Validation was
used to choose configurations and to early-stop XGBoost, so the scores are optimistic (D21).

## 1. Setup
| item | value |
|---|---|
| Fitting data | 217,727 labelled flights, 2013-01-01..08-31; 57,423 positives; prevalence 0.2637 |
| Validation | 55,628 labelled flights, 2013-09-01..10-31; 8,731 positives; prevalence 0.1570 |
| Preprocessing | Stage 6 `TrainOnlyPreprocessor`, fitted on the training rows only (verified per run) |
| Seed | 42 (all models); refitting the selected models reproduced validation predictions exactly (max abs diff 0.0) |
| Libraries | scikit-learn 1.9.1, XGBoost 3.4.1 (needs `libomp`, installed with Homebrew with the user's approval), pandas 3.1.0, numpy 2.5.4 |

### Models and search space (fixed in `configs/config.yaml` before training)
| model | configuration |
|---|---|
| Most-frequent dummy | always P(delay) = 0. A trivial reference: it would be 84.3% "accurate" on validation while detecting no delays |
| Prior baseline | always predicts the training prevalence (0.2637). A probabilistic reference for Brier and log loss, kept separate from the dummy |
| Logistic Regression | L2, lbfgs, max_iter 5,000; C ∈ {0.01, 0.1, 1, 10} × class_weight ∈ {none, balanced}: 8 configs |
| XGBoost | `hist`, learning_rate 0.05, subsample 0.8, colsample 0.8, ≤ 2,000 trees, early stopping (50 rounds) on validation `aucpr`; max_depth ∈ {3, 5, 7} × min_child_weight ∈ {20, 1} × scale_pos_weight ∈ {1, training neg/pos = 2.79}: 12 configs |

**Selection rule (D26):** highest validation Average Precision per family. Exact ties (within
1e-9) go to the simpler configuration in grid order. Every configuration and its metrics are
recorded.

**Class imbalance (D25):** the imbalance is moderate (26.4% positives in training). Weighting was
tested as a grid option, and the weights come from training labels only:
- `class_weight="balanced"` for Logistic Regression;
- `scale_pos_weight` = training negatives/positives for XGBoost.

A test shows the weight does not change when validation labels change.

## 2. Validation comparison
Intervals are 95% day-block bootstrap intervals: 200 resamples of whole validation dates,
because same-day flights are not independent.

| model | selected config | val AP [95% CI] | val ROC AUC [95% CI] | val Brier | val log loss | mean predicted P |
|---|---|---|---|---|---|---|
| Most-frequent dummy | — | 0.157 (= prevalence) | 0.500 | 0.1570 | 5.421 | 0.000 |
| Prior baseline | — | 0.157 (= prevalence) | 0.500 | 0.1437 | 0.4673 | 0.264 |
| **Logistic Regression** | C = 0.01, no class weight | **0.2708 [0.215, 0.318]** | **0.6699 [0.641, 0.696]** | 0.1318 | 0.4284 | 0.231 |
| XGBoost | depth 3, mcw 20, scale_pos_weight 2.79, 31 trees | 0.2352 [0.187, 0.279] | 0.6498 [0.624, 0.676] | 0.2385 | 0.6690 | 0.488 |

The dummy's AP interval (0.133–0.178) is the uncertainty in validation prevalence itself.

**Paired difference, XGBoost − Logistic Regression** (same resampled days):
- AP **−0.036 [−0.056, −0.016]**;
- ROC AUC **−0.020 [−0.031, −0.011]**.

Both intervals exclude zero.

**By validation month:**

| month | prevalence | LR AP | LR ROC | XGB AP | XGB ROC |
|---|---|---|---|---|---|
| Sep | 0.137 | 0.218 | 0.666 | 0.213 | 0.658 |
| Oct | 0.176 | 0.318 | 0.673 | 0.252 | 0.639 |

**Training partition (in-sample, for diagnosing overfitting only):**
- Logistic Regression: AP 0.422, ROC 0.682.
- Selected XGBoost: AP 0.443, ROC 0.689.
- Deeper XGBoost configurations reached training AP 0.67–0.68 but validation AP ≤ 0.235.

## 3. Interpretation
- **Both models beat the baselines.** Logistic Regression's validation AP is 1.72× the prevalence,
  with ROC AUC 0.67. That is real but modest ranking ability, consistent with a day-before
  horizon that cannot see same-day weather or congestion (contract §6). These numbers do not show
  that any operating point reaches 70% precision; that is examined only in Stage 8.
- **Logistic Regression ranks better than XGBoost on validation.** The day-block bootstrap shows
  the gap is unlikely to be resampling noise, although validation was also used for selection.
- **Why XGBoost underperforms (hypotheses, not established).** Early stopping halted XGBoost
  very early: 31 trees for depth 3, and only 3–5 trees for some deep configurations. Deeper models
  fit training far better (AP up to 0.68) but validated worse. So patterns XGBoost can learn from
  January–August do not carry over to the calmer September–October period. Plausible reasons:
  - splits on absolute levels of historical rates, which shift between seasons;
  - interactions specific to summer operations.

  Logistic Regression's strong regularisation (selected C = 0.01, the smallest value in the grid)
  generalises better under this shift. Testing these explanations would need further
  validation-driven changes (see §5).
- **Class weighting does not improve ranking.**
  - Logistic Regression: best AP 0.2708 unweighted vs 0.2702 weighted.
  - XGBoost: 0.2347 unweighted vs 0.2352 weighted, a difference of 0.0005, far inside the
    bootstrap uncertainty.
  - Weighting badly distorts probabilities: best log loss 0.427 → 0.597 for Logistic Regression and
    0.452 → 0.669 for XGBoost. The selected XGBoost has a mean prediction of 0.488 against a
    validation prevalence of 0.157.
- **Probabilities are not calibrated on validation.** Even unweighted Logistic Regression predicts
  0.231 on average against 0.157 observed, because the delay rate shifted from training to
  validation. Its Brier score (0.1318) still beats the prior baseline (0.1437). Calibration is
  Stage 8 (D11).

## 4. Checks (all PASS; `baseline_selection.json`)
1. Every model is fitted on exactly the 217,727 labelled training rows (preprocessing date range
   ends 2013-08-31).
2. No test-period flight is used for fitting, early stopping or prediction (id sets disjoint).
3. Validation predictions are aligned to validation rows (index equality).
4. All probabilities are finite and in [0, 1].
5. All 8 Logistic Regression fits converged (61–160 iterations, below max_iter 5,000, no
   ConvergenceWarning).
6. XGBoost class weight = training negatives/positives (2.7917).
7. Refitting the selected models reproduces validation predictions exactly.
8. The test-label lock is intact (no frozen manifest, no access log).
9. The raw file is unchanged.

**Unit tests:** 14 model tests and 12 metric tests. They cover:
- training-only fitting;
- early stopping restricted to validation rows;
- row alignment under shuffling;
- determinism;
- class weight independent of validation labels;
- Logistic Regression unaffected by validation labels;
- dummy and prior behaviour;
- hand-calculated AP (5/6), ROC (3/4), Brier and log loss;
- undefined-metric errors;
- the bootstrap.

**Mutation check:** computing the class weight from validation labels, and removing the
validation-only early-stopping guard, are each caught by a failing test.

## 5. Limitations and open points
- **Optimistic validation.** Selection and early stopping both used validation, and XGBoost's
  stopping point was chosen on the same metric used to compare it. This makes XGBoost's score
  optimistic, yet it still trails Logistic Regression.
- **Grid boundary.** Logistic Regression's best C (0.01) is the smallest value tried. Validation AP
  varies little across the grid (0.2677–0.2708), so the model is insensitive to C, but a smaller C
  was not explored.
- **Near-tie in XGBoost selection.** The declared rule picked the class-weighted configuration
  over the unweighted one by 0.0005 AP. The weighted one has heavily distorted probabilities.
- **Single validation period, single year.** Validation (Sep–Oct) is the calmest period of 2013.
  Rankings may differ in other seasons.
- **No feature importance yet.** Interpretation and error analysis are Stage 9 (they will use
  validation, not test, until the final evaluation).
