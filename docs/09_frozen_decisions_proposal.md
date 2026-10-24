# Frozen-Decisions PROPOSAL (for review before any test-label release)

> **Status: PROPOSAL, NOT THE UNLOCKING FILE.**
>
> **Proposal:** `reports/freeze/frozen_decisions_proposal.json`, SHA-256
> **`46df47cec7ec57cc23762e86571383b8a5551138b4a8284ef421fd022802f1c5`**. Built and verified by
> `.venv/bin/python scripts/prepare_frozen_proposal.py`: 21/21 checks.
>
> **It cannot release test labels.** Labels are released only by `unlock_test_labels`, which reads
> `reports/final/frozen_manifest.json` (a fixed path; it **does not exist**) and accepts it only if:
> - it has `status: "FROZEN"`;
> - its `approved_proposal_sha256` equals the proposal file on disk;
> - every bound decision field equals the proposal;
> - the config, raw data, model, calibrator, evaluation-script and `src` package hashes all match
>   the files on disk.
>
> The manifest will be created only after your explicit approval (via `experiment.build_manifest`).
> Re-running the proposal script changes its timestamp and therefore its SHA-256. The manifest
> binds to the exact version you approve.

No test label has been accessed, no test prediction scored and no test statistic computed.
`scripts/evaluate_test.py` exists but **has not been run**.

## 1. Prediction contract (unchanged)
| item | frozen value |
|---|---|
| Prediction timestamp | End of the day before the scheduled flight date (D3) |
| Target | `y = 1` if `arr_delay ≥ 15` min, `0` if below. Flights without a recorded arrival delay are excluded and never labelled (D4, D6) |
| Population | NYC departures (EWR/JFK/LGA), 2013, labelled flights only |
| Split | train 2013-01-01..08-31 · validation 2013-09-01..10-31 · test 2013-11-01..12-31 (D5) |
| Historical cutoff | flights with flight date ≤ D−2 only (`history_lag_days: 2`, D10) |
| Features | the 31 features in `docs/feature_dictionary.md` (D15–D17) |
| History settings | carrier/origin/dest/route; expanding + 28 d (entities), expanding + 7 d + 28 d (system-wide); `min_count` 20; shrinkage `m` 20 |
| Preprocessing | Stage 6 `TrainOnlyPreprocessor`, fitted on training rows only (D18, D19, D24); `min_category_frequency` 100 |
| Refit | none: models trained on Jan–Aug only (D20) |

## 2. Models to be scored
### 2.1 Primary: Logistic Regression + sigmoid calibration (`logistic_regression_sigmoid`)
| item | value |
|---|---|
| Configuration | `logistic_regression_00`: C = 0.01, L2, lbfgs, unweighted, max_iter 5,000, seed 42 (D26, D29) |
| Training | 217,727 labelled flights (57,423 positives); 142 preprocessed columns |
| Calibration | sigmoid, selected by D11 (mean held-out log loss: none 0.42788, **sigmoid 0.41387**, isotonic 0.41854) |
| Final calibrator | q = σ(**1.0096062** · logit p − 0.4961186), fitted on all 55,628 validation predictions |
| Operating threshold | **None. "No operating threshold — the 70% precision target was infeasible on validation."** (D32) |
| Validation (development, optimistic) | AP 0.247, ROC AUC 0.659, Brier 0.1270, log loss 0.4147 |
| Model artifact | `models/stage8_logistic_regression_model.joblib`, SHA-256 `166f5d0168328836b0800d06af2486a07657b7cf9414fe3d19b1b7e85503078b` |
| Calibrator artifact | `models/stage8_logistic_regression_calibrator_sigmoid.joblib`, SHA-256 `4724afd7d488d17bf6b00d68c6e2c99b89a3cd89ec0e902f23fdbe7d443f429f` |

### 2.2 Comparison: XGBoost + isotonic calibration (`xgboost_isotonic`)
| item | value |
|---|---|
| Configuration | `xgboost_01`: `hist`, max_depth 3, min_child_weight 20, scale_pos_weight 2.7916 (training negatives/positives), learning rate 0.05, subsample 0.8, colsample 0.8, early stopping on validation `aucpr` (31 trees), seed 42; 134 preprocessed columns |
| Calibration | isotonic, selected by D11 (none 0.66880, sigmoid 0.41830, **isotonic 0.41782**); 31-step map fitted on all validation predictions |
| Operating threshold | None (D32; best validation precision 0.377) |
| Validation (development) | AP 0.228, ROC AUC 0.643, Brier 0.1282, log loss 0.4188 |
| Model artifact | `models/stage8_xgboost_model.joblib`, SHA-256 `cf941cb79cc649c7b3ba8e06c5041ae29ecde6826eddb7230296e32dc242c3e1` |
| Calibrator artifact | `models/stage8_xgboost_calibrator_isotonic.joblib`, SHA-256 `a82b587db541e786c1f6a1e0dc2b235d337ea9c2404edb93c0dc39d65493d557` |

### 2.3 References and secondary predictions
- `prior_baseline`: constant training prevalence, P = 0.2637.
- `dummy_most_frequent`: constant P = 0.
- Secondary, descriptive only: `logistic_regression_raw` and `xgboost_raw` (uncalibrated probabilities).

## 3. Calibration and threshold procedure (D11, applied unchanged)
1. Compare none / sigmoid / isotonic: fit on September and score log loss on October, then the
   reverse. Select the lowest mean held-out log loss; within 1e-6, the simpler method wins.
2. Threshold: use validation predictions unseen by the calibrator (raw if *none*, otherwise
   cross-fitted by month). Choose precision ≥ 0.70 with maximum recall, ties to the highest
   threshold. If none exists, it is infeasible and the target is not relaxed.
3. Final calibrator: the selected method refitted on all validation predictions.

**Outcome:** infeasible for both models.

**Limitation (D33).** Month-specific cross-fitted maps reorder flights between months. On raw
Logistic Regression scores, 70% precision occurs only at the top 7 validation flights (5 delayed),
in 60% of day resamples. D11 is kept unchanged.

## 4. Test evaluation (pre-registered; `scripts/evaluate_test.py`, not run)
**Order of operations, enforced in code and checked statically by the proposal script:**
1. Load and validate the FROZEN manifest **before loading any data**.
2. Integrity:
   - load the hash-pinned artifacts;
   - a deterministic refit on **training rows only** must reproduce their validation
     predictions exactly (the script's single `.fit` call);
   - calibrator parameters must equal the manifest.
3. Score test features **without labels**. Save `reports/final/test_predictions.csv` (no labels)
   and its SHA-256 **before** any label is released.
4. Release labels once via `unlock_test_labels`. The log records the access number, the manifest
   SHA-256, the proposal SHA-256 and the predictions SHA-256 (in the purpose field).
5. Compute the registered metrics (`flight_delay.evaluation.final.evaluate_predictions`). Write
   `reports/final/test_evaluation.json` and figures.

**Static checks on the script:**
- no call to threshold selection, recall at precision, metrics at a threshold, calibration
  comparison or cross-fitting, `Calibrator`, `partition` or `build_manifest`;
- exactly one `.fit` call;
- the order of operations above.

### 4.1 Metric definitions
| metric | definition |
|---|---|
| Average Precision | step-wise AP (`average_precision_score`), not trapezoidal |
| ROC AUC | `roc_auc_score` |
| Brier score | mean squared error of probabilities |
| Log loss | binary cross-entropy, probabilities clipped to [1e−15, 1−1e−15] |
| Calibration-in-the-large | mean(q) − mean(y); positive = over-prediction |
| Calibration intercept and slope | unpenalised logistic regression of y on logit(clip(q, 1e−15, 1−1e−15)); slope = coefficient, intercept = constant; ideal (0, 1); "undefined (constant prediction)" for constants; descriptive only, never applied |
| Reliability | up to 10 quantile bins of q; bins with duplicate edges merged (isotonic step outputs may give fewer); per bin n, mean q, observed rate |
| Mean predicted probability | mean(q) |

### 4.2 What is reported
- **Partition summary:** labelled test rows, positives, prevalence.
- **For the primary, comparison and both reference models:** all metrics in §4.1, with 95%
  intervals for Average Precision, ROC AUC, Brier and log loss.
- **Secondary, descriptive:** the same point metrics for raw Logistic Regression and XGBoost;
  point metrics and calibration-in-the-large by month (2013-11, 2013-12); validation vs test
  prevalence and mean predicted probability.
- **Threshold-dependent metrics: NOT APPLICABLE (D32).** Recall at the 70% minimum-precision
  threshold, precision, recall, F1 and the confusion matrix at an operating threshold are not
  computed. The report states: "No operating threshold — the 70% precision target was infeasible
  on validation."
- **Figures:** precision–recall and ROC curves (descriptive, with the 0.70 line; no threshold read
  from them) and reliability diagrams.

### 4.3 Uncertainty
- **Method:** 95% percentile intervals from a day-block bootstrap. 200 resamples of whole test
  flight dates with replacement, seed 42 (D28). The same resampled days are used for every model
  and metric.
- **Undefined resamples:** excluded and counted (`n_undefined`, reported with every interval).
- **Paired differences:** primary LR − XGBoost and primary LR − prior baseline, for AP, ROC AUC,
  Brier and log loss. **Sign convention:** difference = metric(primary) − metric(other). AP and
  ROC AUC > 0 favour the primary model; Brier and log loss < 0 favour it.
- **Constant baselines:** use the same code. ROC AUC is 0.5 by construction (degenerate
  interval), AP equals the prevalence, and the calibration slope is undefined.

### 4.4 Access rule
Labels are released exactly once. The lock refuses a second release unless a repeat reason is
given. A repeat is allowed only to fix a software defect that changes no frozen decision, and
both outputs must be disclosed. Later test error analysis (plan Stage 9) is exploratory and
cannot change the frozen pipeline.

## 5. Prior exposure to test-period information (from Stage 5, docs/05 §3.1)
| where | what was seen |
|---|---|
| Stage 1 audit | monthly delay prevalence incl. Nov 0.181 and Dec 0.336; whole-year delay distribution |
| Stage 2 contract | test labelled rows 53,991, positives 13,946, prevalence 0.2583; D10 positive share 53.8% (all partitions) |
| Stage 3 EDA | daily/monthly prevalence through December (descriptive), label-free category mix, unseen categories |
| Stage 4 (earlier version) | test-period feature ranges incl. historical rates from test-period outcomes, fallback shares, unseen categories |
| Never | test predictions, metrics, calibration results, error analyses |

**Mitigation:** the calibration and threshold rules were fixed before any model existed; D17 used
label-free evidence only.

## 6. Strict rules after test access
1. Test results **cannot** be used to alter the model, hyperparameters, preprocessing, feature
   definitions, historical windows, calibration method or calibrator, threshold rule, operating
   point, split or prediction contract.
2. The test evaluation is reported once, as obtained, including if disappointing.
3. Any later analysis or model change is a **new, non-final experiment**, labelled as such.
4. No refit on train + validation (D20). Only the hash-pinned artifacts are scored.

## 7. Integrity identifiers bound by the manifest
| item | SHA-256 |
|---|---|
| Evaluation script `scripts/evaluate_test.py` | `91f77b36381365351c162117737a8e446d94db262d3d38eec3fd7833fb8af2ed` |
| Package code (`src/flight_delay/**/*.py`) | `1ee39dff2e2b3ca28f86573081e03c645f11039484c8781d41bac714db10b08b` |
| Config | `75e70698894feb0c1813c126250585a28fb1f034842f7364d441ebb39423e75b` |
| Raw data | `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa` |
| Training ids / validation ids | `382f3e11…40b208d0b6a` / `95fd38ae…4fdddcf59a6970214aca1c5c61e0889ae23914c34585bdd4f88` (full values in the JSON) |

Libraries: Python 3.13.13, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1 (+ Homebrew `libomp`).

**Consequence:** any edit to `src/`, `scripts/evaluate_test.py`, the config, the raw data or the
artifacts after approval makes the manifest invalid. The proposal would then need regenerating and
re-approval.

## 8. Verification (21/21 PASS; `checks` in the JSON)
**Configuration and decisions:**
- the config matches D−2, the 15-min target, 0.70 precision and the split;
- primary and comparison models per D29;
- calibration per D31;
- threshold infeasible per D32;
- the Stage 7 configurations were used.

**Artifacts:**
- models reproduce the recorded and refit validation predictions;
- calibrators match;
- preprocessing was fitted on training rows only.

**Lock:**
- this proposal is rejected if used as a manifest;
- the official lock refuses release without a manifest;
- no access logged.

**Evaluation script, static checks:**
- no forbidden calls;
- one `.fit` call;
- the order of operations;
- the raw file is unchanged.
