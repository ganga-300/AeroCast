# Stage 5 — Chronological Experimental Design

Code: `src/flight_delay/experiment.py`. Real-data check: `.venv/bin/python
scripts/check_experiment_design.py` → `reports/experiment/design_check.json`. Tests:
`tests/test_experiment.py`. No model was trained and no preprocessing was fitted.

## 0. Scope review (from the master plan, not redefined)
| Plan item | Where it is satisfied |
|---|---|
| Inspect date range, usable days, prevalence by period, split feasibility | Stage 1 audit, Stage 2 contract; re-verified here (§1) |
| 5.1 train < validation < test; one partition per date; exact boundaries and counts; sufficient examples; one-year limitation | §1, `partition()`, design check 1–2 |
| 5.2 Separate feature history from fitting data, conceptually and in code | §2, `assert_fit_rows()`, Stage 4 builder |
| 5.3 Test-set protection; validation uses; holdout vs prospective | §3–§4, `development_partitions()`, `unlock_test_labels()` |

**Stage numbering for what follows.** In the master plan, Stage 6 is *only* the preprocessing
pipeline. Model training is Stage 7, and metrics, threshold selection and calibration are Stage 8.
The latest instruction describes "Stage 6" as including model training, threshold, calibration
and final test evaluation. That corresponds to plan Stages 6–8 plus the final evaluation. This is
flagged for the user rather than silently merged.

## 1. Partitions (verified on the real data)
| partition | dates (inclusive) | days | labelled rows | positives | negatives | prevalence |
|---|---|---:|---:|---:|---:|---:|
| train | 2013-01-01 – 2013-08-31 | 243 | 217,727 | 57,423 | 160,304 | 0.2637 |
| validation | 2013-09-01 – 2013-10-31 | 61 | 55,628 | 8,731 | 46,897 | 0.1570 |
| test | 2013-11-01 – 2013-12-31 | 61 | 53,991 | withheld | withheld | withheld |

Guarantees asserted in code (`partition`):
- each partition's dates lie inside its configured range;
- every calendar day in each range has labelled flights;
- each flight date maps to exactly one partition;
- partitions are strictly ordered in time;
- no flight id appears in two partitions.

**Sufficiency.** Train and validation each have ≥ 500 positives and ≥ 500 negatives. The test check
returns only PASS/FAIL: it passed, and the counts are not reported.

**One-year limitation.** All data are from 2013. Training (Jan–Aug) never sees autumn or winter, so
seasonal differences between the partitions cannot be separated from 2013-specific events.
Validation (Sep–Oct) is the calmest period of the year (prevalence 0.157 vs 0.264 in training).

## 2. Feature history versus fitting data (5.2)
Two different questions, kept apart in code:

| question | answered by | data allowed |
|---|---|---|
| What did the system know about other flights when predicting flight *i* on date D? | `features.build` (historical features) | outcomes of any flight dated ≤ D−2, from any partition |
| Which labelled rows may the model learn its parameters from? | `experiment.assert_fit_rows` | labelled **training** rows only |

So a validation flight's features may summarise training-period outcomes and earlier
validation-period outcomes ≤ D−2. For the 28-day windows (label-free count by date):
- 1,553 validation rows have windows wholly inside the training period (early September);
- 29,604 validation rows have windows wholly inside the validation period (from Oct 1 onwards).

This mirrors deployment and does **not** mean the model is fitted on validation labels. The
feature builder never reads `split` (Stage 4 test), and the fitting check rejects any
non-training or unlabelled row.

**Boundary embargo note.** A model that predicts date D could, in deployment, only have been
trained on labels dated ≤ D−2. The model is trained on labels up to 2013-08-31, so this is
violated for one validation date only: 2013-09-01 (712 rows, 1.28% of validation). It does not
affect test (training ends two months before 2013-11-01). Dropping 2013-08-31 from training would
remove it, but that would change the confirmed split, so the split is unchanged. The effect is a
small optimism on validation day 1, disclosed here (D22).

## 3. Test-set protection (5.3)
- `development_partitions(table, cfg)` returns (train, validation) only. All modelling decisions
  must come from it.
- `holdout_features_without_labels(table, cfg)` returns test rows with `y` and `label_status`
  removed, so the frozen pipeline can score them.
- `partition()` returns test rows **without** label columns; the test sufficiency check is a
  PASS/FAIL function (`test_partition_sufficient`).
- `unlock_test_labels(...)` reads the fixed manifest `reports/final/frozen_manifest.json` and
  releases labels only if it is `status: "FROZEN"` and bound to the approved proposal and to the
  hashes of the config, raw data, artifacts, evaluation script and `src` code (see
  `docs/09_frozen_decisions_proposal.md`). Every release is appended to
  `reports/final/test_label_access_log.jsonl` with the manifest hash; a repeat needs a logged
  reason. The design check verifies there is no manifest and no logged access.
- All scripts now withhold test-label statistics: `check_contract.py` reports test
  positives/prevalence as "withheld"; `build_features.py` summarises train/validation only.
  Mechanical leakage checks that involve test rows report PASS/FAIL only.

**This is a retrospective chronological holdout**, not a prospective deployment test. It shows
how the frozen pipeline behaves on later 2013 data that was not used for any decision. It cannot
show behaviour in other years, other airports or under live data feeds.

### 3.1 Disclosure: test-period information seen before this stage
No model has ever been trained, so **no test metric, prediction, calibration result or error
analysis exists or has been seen**. Before the stricter protocol, these test-period quantities
were computed and documented:

| where | what was seen |
|---|---|
| Stage 1 audit (`dataset_audit.json`) | monthly delay prevalence incl. Nov (0.181) and Dec (0.336); whole-year delay distribution |
| Stage 2 contract (`docs/02`) | test labelled rows 53,991, positives 13,946, prevalence 0.2583; D10 positive share 53.8% computed over all partitions |
| Stage 3 EDA (`docs/03`, Figure 3) | daily/monthly prevalence through December (descriptive), label-free category mix and unseen categories |
| Stage 4 (`docs/04`, earlier `feature_validation.json`) | test-period feature ranges (incl. historical rates built from test-period outcomes), fallback shares, unseen categories |

**Implications.**
- Knowing the test base rate (higher than validation) creates a risk of choosing a calibration
  method or threshold rule *because* it suits a higher base rate.
- To neutralise this, the protocol in §4 fixes in advance, before any model exists, how the
  calibration method and threshold are chosen. Both are chosen by validation-only criteria.
- The Stage 4 feature-set change (D17) was made on label-free range evidence (a monotone count
  feature). It used no test outcome or metric.
- The report will repeat this disclosure.

## 4. Validation-use protocol (CONFIRMED by the user, 2026-10-24)
Fixed before any model is trained (D11, D20, D21). Codified in `configs/config.yaml`
(`calibration`, `operating_point`) and applied in Stage 8 (`docs/08_calibration_threshold.md`).

| step | data | rule |
|---|---|---|
| Preprocessing fit (imputer, category mapping, scaler) | train | fitted on training rows only (D18, D19) |
| Model fitting, all models | train (Jan–Aug) | final models are **not** refitted on train+validation (D20) |
| Model / feature-variant / hyperparameter selection | validation | Average Precision on validation (D21) |
| XGBoost early stopping | validation | permitted (D21) |
| Calibration method: none / sigmoid / isotonic | validation, Sep↔Oct cross-validation | fit on September, log loss on October, then the reverse; select the lowest **average held-out log loss** (D11) |
| Final calibrator | validation | the selected method refitted on all validation predictions; this is the calibrator applied to test |
| Operating threshold | validation only, out-of-fold where feasible | see §4.1 (D7, D11) |
| Final evaluation | test, once | frozen manifest, released through `unlock_test_labels`; never used to choose or change the model, calibration or threshold |

### 4.1 Threshold selection (D7 + D11)
1. Build validation predictions that the calibrator did not see:
   - if the selected method is *none*, use the model's validation probabilities directly. They are
     already out-of-sample because the model was trained on Jan–Aug only;
   - otherwise use **cross-fitted** calibrated probabilities: September rows calibrated by the
     calibrator fitted on October, and October rows by the one fitted on September.
2. On those predictions, keep thresholds with precision ≥ 0.70. Choose the one with the highest
   recall. Ties go to the highest threshold.
3. Freeze that probability threshold and apply it to the final calibrator's test probabilities.
   Because the final calibrator is fitted on all of validation, its scale can differ slightly from
   the cross-fitted ones. That residual mismatch will be reported alongside the test result.
4. If no threshold reaches 0.70 precision on validation, report it as infeasible. The requirement
   is not relaxed and no operating point is manufactured.

Tie rule for the calibration choice: if two methods have average held-out log loss equal within
1e-6, the simpler is preferred (none, then sigmoid, then isotonic). This keeps the choice
deterministic.

### 4.2 Why these rules
- **No refit (D20).** The calibrator and threshold are estimated from predictions on data the model
  has not seen. Refitting on train+validation would create a different model whose calibration
  and threshold were never estimated. The cost: Sep–Oct data are not used for fitting.
- **Optimism (D21).** Model selection, early stopping, calibration selection and threshold
  selection all reuse the same two validation months. Repeated validation-based choices tend to
  make validation scores **optimistic**. Validation results are therefore reported as development
  estimates only; the single test evaluation is the reported final estimate.
- **Month cross-validation (D11).** Calibrating on one month and scoring on another tests whether a
  calibration map transfers across time. Sep (prevalence 0.137) and Oct (0.176) differ, so this is
  a deliberately honest check, not an in-sample fit.

Validation sub-periods: Sep 27,010 rows (3,705 positives, prevalence 0.137); Oct 28,618 rows
(5,026 positives, 0.176).

## 5. Leakage risks remaining after Stage 5
| risk | status |
|---|---|
| Same-day / D−1 outcomes in features | eliminated by D10, verified Stage 4 |
| Test labels influencing decisions | locked in code; prior exposure limited to base rates and descriptive views (§3.1) |
| Validation reuse (selection, early stopping, calibration, threshold on two months) | validation scores optimistic (D21); threshold chosen on out-of-fold calibrated predictions where feasible; only test is reported as final |
| Final-calibrator vs cross-fitted scale mismatch at the threshold | to be measured and reported (§4.1 step 3) |
| Train/validation boundary (2013-09-01) | 712 rows, disclosed (D22) |
| Preprocessing fitted on non-training data | to be enforced and tested in plan Stage 6 (`assert_fit_rows`) |
