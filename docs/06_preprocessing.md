# Stage 6 — Training-Only Preprocessing Pipeline

Code: `src/flight_delay/preprocessing.py`. Real-data check: `.venv/bin/python
scripts/check_preprocessing.py` → `reports/preprocessing/preprocessing_check.json` (includes
the full ordered output column list per model). Tests: `tests/test_preprocessing.py`.

**Scope (plan Stage 6):** the raw-to-model-input transformation only. No model was trained, no
hyperparameter tuned, nothing calibrated, no threshold chosen, and no metric computed. Every
encoding choice below was fixed a priori; none was chosen from validation or test performance.

## 1. Pipeline
`build_feature_table` (Stage 4) → `development_partitions` / `holdout_features_without_labels`
(Stage 5) → `TrainOnlyPreprocessor(family).fit(train)` → `.transform(any partition)`.

`TrainOnlyPreprocessor` wraps one scikit-learn `ColumnTransformer` (scikit-learn 1.9.1):
- **`fit`** runs `assert_fit_rows` first, so it refuses validation, test or unlabelled rows. It can
  be called only once. It records `fit_info_` (row count, date range, SHA-256 of the fitted ids,
  output width).
- **`transform`** selects the declared input columns by name: input column order is irrelevant,
  and extra columns (labels, post-event fields) are ignored. It never refits, casts to float64,
  and validates the output: the exact columns learned at fit time, in order, no infinities, and
  missing values only where the model family allows them.
- **Input validation:** missing required columns raise `ValueError`; non-numeric values in a
  numeric feature raise `TypeError`; missing categorical values raise.

## 2. Encodings per model family (D24)
| feature group | Logistic Regression | XGBoost | reason |
|---|---|---|---|
| time of day | `dep/arr_hour_sin/cos`, standardised | `dep_hour`, `arr_hour` (integers) | a linear model cannot represent the 23:00→00:00 wrap from a raw hour; trees split on thresholds |
| day of week | one-hot (7) | integer 0–6 | linear model needs one coefficient per day; trees can split integers |
| `is_weekend` | passed through | passed through | already 0/1 |
| `distance` | standardised | passed through | scaling helps regularised LR; trees are scale-invariant |
| historical rates (12) | median imputation (training medians) → standardised | passed through, NaN kept | D18 |
| historical counts (6) | log1p → standardised | passed through | counts span orders of magnitude (e.g. route vs system-wide); log for the linear model only |
| `history_unavailable` | 1 if any historical rate is missing | same definition | D18: one stateless indicator, identical in every partition |
| `carrier`, `origin`, `dest` | one-hot with `min_category_frequency = 100` | same encoder | D19 |
| `route` | not encoded | not encoded | D19: represented by `hist_route_rate_*`, `hist_route_n_28d` |

Output width: **142 columns** (Logistic Regression), **134 columns** (XGBoost).

## 3. Missing history (D18)
- Feature generation leaves history missing (NaN) for the 1,759 labelled training flights on
  2013-01-01/02. Nothing is filled from future outcomes at that stage.
- **Logistic Regression:** `SimpleImputer(strategy="median")` fitted on training rows. Medians
  (e.g. `hist_all_rate_exp` 0.2451, `hist_all_rate_7d` 0.2650, `hist_all_rate_28d` 0.2534) are in
  the JSON report.
- **The imputed values for those January rows are statistics of the training feature column, so
  they reflect training-period information later than Jan 2.** This is acceptable:
  - it affects only training rows, whose labels the model is fitted on anyway;
  - it cannot reach validation or test, which have no missing history (verified PASS/FAIL);
  - an imputer fitted on validation or test is impossible by construction.
- The `history_unavailable` indicator lets the model treat imputed rows differently instead of
  trusting the imputed value. It is computed from the features themselves (no fitted state), so
  the definition is identical for every partition.
- **XGBoost:** NaN is passed through for native missing-value handling. It gets the same indicator.

## 4. Rare and unseen categories (D19)
Category lists and rare-category grouping are learned from **training counts only**.

| | rule | learned from training |
|---|---|---|
| rare category (< 100 training flights) | grouped into `<col>_infrequent_sklearn` | carrier: OO; destination: ANC, BZN, CAE, CHO, EYW, HDN, JAC, MTJ, MYR, PSP, SBN, TVC; origin: none |
| unseen category (validation/test) | mapped to the infrequent column when one exists; otherwise all zeros for that block | carrier → `carrier_infrequent_sklearn`; destination → `dest_infrequent_sklearn`; origin → all-zero block |
| matrix width | never changes after fit | verified for validation and test |
| route | not one-hot | its history features fall back to the system-wide rate below `min_count` (Stage 4) |

Validation (train-fitted mapping): 51 rows to the unseen destination ILM go to
`dest_infrequent_sklearn`; 42 rows to training-rare destinations and 17 rows of carrier OO use
their infrequent columns. Test-period mapping is checked PASS/FAIL only (identical width and
columns).

Both models see the same categorical treatment. The difference is how each uses it: Logistic
Regression gets one coefficient per column, including the shared infrequent column; trees can
split on any column.

## 5. Verification

### 5.1 Unit tests (`tests/test_preprocessing.py`, 18 tests, toy data)
- Fitting on validation rows, or on training plus a few validation rows, raises.
- No refit is allowed, and transforming before fitting raises.
- Transforming validation, a shifted distribution and test features leaves the pickled fitted
  transformer byte-identical.
- Standardisation equals `(x − training mean) / training sd` computed by hand.
- The rare training carrier is learned as infrequent. An unseen carrier maps to the infrequent
  column; an unseen destination with no infrequent group maps to an all-zero block. Width is
  identical across partitions.
- Cold start: the Logistic Regression imputer statistic equals the training median, its output
  has no NaN, XGBoost keeps NaN, and the indicator is 1 only on cold-start rows.
- Reordered input columns plus extra label/post-event columns give identical output. A missing
  column raises.
- Output is float64 with the expected shape and deterministic across fits. Input is not mutated.
- Inputs are declared, contract-permitted features only. `route` is not encoded.
- Every feature lands in exactly one preprocessing group.

**Mutation check.** Four planted bugs, each caught, then restored:
- removing the training-only guard: 2 tests failed;
- refitting inside `transform`: 9 failed;
- `handle_unknown="ignore"`: 2 failed;
- mean instead of median imputation: 1 failed.

### 5.2 Real-data check (`scripts/check_preprocessing.py`, 14/14 PASS)
For both families:
- fitted on exactly the 217,727 training rows (2013-01-01..08-31);
- no prohibited inputs;
- parameters unchanged after transforming validation and test;
- identical columns in all partitions;
- missing values only where allowed;
- indicator equals the cold-start definition in every partition.

Also: the test-label lock is intact and the raw file is unchanged.

## 6. Remaining risks and open items
- **Dense matrices.** Training is 217,727 × 142 float64 (≈ 0.25 GB) for Logistic Regression.
  Acceptable on this machine; sparse output is an option if memory becomes an issue.
- **Redundant inputs for Logistic Regression.** `is_weekend` is a linear combination of the
  day-of-week one-hot columns. Regularisation (Stage 7) handles this, but individual coefficients
  for those columns must not be interpreted on their own.
- **Rare-category threshold.** The cut-off of 100 training flights is an a priori choice; it was
  not tuned.
- **Not done yet (plan Stages 7–8):** the models, their regularisation and hyperparameters, early
  stopping, calibration and threshold selection.
