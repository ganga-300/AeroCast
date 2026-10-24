# Final Test Evaluation (single, pre-registered run)

Source of every number: `reports/final/test_evaluation.json`. Produced by one run of
`scripts/evaluate_test.py` under the FROZEN manifest. Nothing in the pipeline was changed after
labels were released, and nothing will be changed because of these results (strict rules,
`docs/09`).

## 1. Audit trail
| item | value |
|---|---|
| Approved proposal | `reports/freeze/frozen_decisions_proposal.json`, SHA-256 `46df47cec7ec57cc23762e86571383b8a5551138b4a8284ef421fd022802f1c5` (verified before freezing) |
| FROZEN manifest | `reports/final/frozen_manifest.json`, SHA-256 `baa5aca6c84e21c73e9a3ee480b86bc255670929ae54bdfd829698e2bfcd85c6`; frozen_at 2026-10-24T15:50:23Z; all pinned hashes verified before release |
| Test predictions (no labels) | `reports/final/test_predictions.csv`, SHA-256 `5a67dd31509b2032e2396104d7dc0c7e8e5c5e132d8888e720378e05e60a10e5`. Written at 16:05:37.010Z, before the label release at 16:05:37.114Z |
| Label releases | **exactly 1** (`test_label_access_log.jsonl`: access_number 1, no repeat reason, manifest and proposal hashes recorded) |
| Run | one execution, exit code 0 (`evaluate_test_run.log`). It took about 15 minutes, almost all before the label release (feature build, integrity refits, scoring); the slowness was not investigated, since that would need a re-run. No automatic retry |

## 2. Partition
Test (2013-11-01..12-31): 53,991 labelled flights; 13,946 delayed; **prevalence 0.2583**. Validation
prevalence was 0.1570.

## 3. Registered metrics (95% day-block bootstrap intervals, 200 resamples, seed 42; undefined resamples: 0 for every interval)
| model | Average Precision | ROC AUC | Brier | Log loss | mean predicted | calibration-in-the-large | calibration intercept / slope |
|---|---|---|---|---|---|---|---|
| **LR + sigmoid (primary)** | **0.363** [0.311, 0.420] | **0.626** [0.599, 0.654] | 0.1904 [0.1691, 0.2164] | 0.5759 [0.5220, 0.6417] | 0.179 | **−0.079** | 0.060 / **0.704** |
| XGBoost + isotonic (comparison) | 0.330 [0.279, 0.386] | 0.609 [0.586, 0.634] | 0.1946 [0.1722, 0.2212] | 0.5872 [0.5309, 0.6544] | 0.169 | −0.089 | 0.031 / 0.660 |
| Prior baseline (P = 0.2637) | 0.258 [0.222, 0.302] | 0.500 (constant) | 0.1916 [0.1746, 0.2121] | 0.5714 [0.5344, 0.6158] | 0.264 | +0.005 | undefined (constant) |
| Most-frequent dummy (P = 0) | 0.258 [0.222, 0.302] | 0.500 (constant) | 0.2583 [0.2223, 0.3016] | 8.921 [7.679, 10.417] | 0.000 | −0.258 | undefined (constant) |

**Paired differences** (primary − other, same resampled days; AP/ROC > 0 and Brier/log loss < 0
favour the primary):

| comparison | ΔAP | ΔROC AUC | ΔBrier | ΔLog loss |
|---|---|---|---|---|
| LR − XGBoost | **+0.033** [0.018, 0.049] | **+0.017** [0.004, 0.030] | **−0.0042** [−0.0064, −0.0022] | **−0.0113** [−0.0189, −0.0047] |
| LR − prior baseline | **+0.105** [0.079, 0.135] | **+0.126** [0.099, 0.154] | −0.0012 [−0.0068, 0.0063] | +0.0045 [−0.0154, 0.0311] |

**Secondary, descriptive (uncalibrated):**
- Raw LR: AP 0.363, ROC 0.626, Brier 0.1850, log loss 0.5550, mean 0.259, calibration-in-the-large
  +0.001, slope 0.711.
- Raw XGBoost: AP 0.340, ROC 0.611, Brier 0.2452, log loss 0.6833, mean 0.500.

**Threshold-dependent metrics: not applicable (D32).** "No operating threshold — the 70% precision
target was infeasible on validation." Recall at 70% precision, precision, recall, F1 and the
confusion matrix were not computed.

## 4. November / December (descriptive)
| month | prevalence | LR AP | LR ROC | LR Brier | LR log loss | LR mean pred | LR calibration-in-the-large | prior Brier | prior log loss |
|---|---|---|---|---|---|---|---|---|---|
| Nov | 0.181 | 0.274 | 0.634 | 0.1435 | 0.4587 | 0.168 | −0.014 | 0.1551 | 0.4920 |
| Dec | 0.336 | 0.427 | 0.606 | 0.2372 | 0.6928 | 0.191 | **−0.145** | 0.2281 | 0.6506 |

XGBoost + isotonic: Nov AP 0.244, ROC 0.620; Dec AP 0.411, ROC 0.612.

## 5. Interpretation
**Descriptive findings (what the measurements show):**
1. **Ranking.** The frozen Logistic Regression ranks test flights better than chance and better than
   XGBoost. AP is 0.363 against prevalence 0.258 (1.40×); ROC AUC 0.626; the paired intervals
   exclude zero. Discrimination is weaker than on validation (ROC 0.659 cross-fitted, 0.670 raw).
2. **Calibration did not transfer.** The sigmoid calibrator fitted on calm September–October lowered
   probabilities. On test the model **under-predicts** (mean 0.179 against 0.258 observed; −0.145 in
   December), and the slope of 0.70 indicates predictions that are too spread out.
   - The calibrated model's Brier and log loss are **not distinguishable from the constant prior**:
     intervals for the differences include 0, and the log-loss point estimate is slightly worse.
   - In December the constant prior beats it on both.
   - The uncalibrated LR happened to be closer to the test base rate (descriptive only; no change made).

   This is the risk recorded in Stage 8 §7: calibration from the calmest months transferring poorly
   to a high-delay period.
3. **Seasonality dominates.** Performance and calibration differ sharply between November
   (prevalence 0.18, well calibrated) and December (0.34, under-predicted).
4. **No operating threshold.** As pre-registered. The descriptive test precision–recall curve peaks
   around 0.66 precision at very small recall; no threshold is read from it.

**What these results do not support:**
- Any claim of operational usefulness for decisions about individual flights.
- Any claim that the 70% precision target is achievable.
- Any claim that the probabilities are reliable risk estimates in high-delay periods.
- Causal statements about delay drivers.
- Generalisation beyond NYC departures in late 2013.

This is a retrospective chronological holdout, not a prospective deployment test.

## 6. Limitations
- **Prior test-period exposure** (aggregate prevalence, descriptive figures; `docs/05` §3.1). No
  predictions or metrics were seen before this run.
- **Validation reuse** (selection, early stopping, calibration, threshold) made validation estimates
  optimistic. The test gap partly reflects this and partly the seasonal shift.
- **Calibration under shift.** Calibration fitted on a single, low-delay validation period. No
  re-calibration is permitted on these results. Any recalibration would be a new, non-final
  experiment.
- **Single year; no weather or same-day information.** These are the main drivers of delay, and
  their absence caps achievable precision at the day-before horizon.
- **Error analysis** on test (plan Stage 9) has not been done. It would be exploratory and cannot
  alter the frozen pipeline.
