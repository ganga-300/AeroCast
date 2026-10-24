# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning — Phase 1 (ML)

Predict whether a flight departing New York City (EWR/JFK/LGA, 2013) will arrive at least
15 minutes late, using only information available at the end of the day before the flight.

**Status:** Stages 0–6 complete (audits, prediction contract, EDA, leakage-safe features,
chronological design with a test-set lock, training-only preprocessing).
Models and the report are not implemented yet; no model results exist.

## Setup
```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
```
Then place the dataset as described in [data/README.md](data/README.md).

## Commands
| Purpose | Command | Output |
|---|---|---|
| Dataset audit (Stage 1) | `.venv/bin/python scripts/audit_dataset.py` | `reports/audit/dataset_audit.{json,md}` |
| Contract consistency check (Stage 2) | `.venv/bin/python scripts/check_contract.py` | `reports/contract/contract_check.json`, `docs/feature_availability.md` |
| Exploratory analysis (Stage 3) | `.venv/bin/python scripts/run_eda.py` | `reports/eda/eda_results.json`, `reports/figures/eda/*.png` |
| Feature build + validation (Stage 4) | `.venv/bin/python scripts/build_features.py` | `reports/features/feature_validation.json`, `docs/feature_dictionary.md` |
| Experimental design check (Stage 5) | `.venv/bin/python scripts/check_experiment_design.py` | `reports/experiment/design_check.json` |
| Preprocessing check (Stage 6) | `.venv/bin/python scripts/check_preprocessing.py` | `reports/preprocessing/preprocessing_check.json` |
| Tests | `.venv/bin/python -m pytest -q` | — |

## Key documents
- [docs/00_workspace_audit.md](docs/00_workspace_audit.md): environment and workspace facts
- [docs/01_dataset_audit_summary.md](docs/01_dataset_audit_summary.md): interpreted data audit
- [docs/02_prediction_contract.md](docs/02_prediction_contract.md): prediction timestamp, target, population, leakage policy, split
- [docs/03_eda_findings.md](docs/03_eda_findings.md): EDA findings, figure captions, open questions for Stage 4
- [docs/04_feature_engineering.md](docs/04_feature_engineering.md): feature definitions, history rules, leakage verification
- [docs/feature_dictionary.md](docs/feature_dictionary.md): generated list of all model features
- [docs/05_experimental_design.md](docs/05_experimental_design.md): partitions, history vs fitting, test lock, validation protocol, test-exposure disclosure
- [docs/06_preprocessing.md](docs/06_preprocessing.md): per-model encodings, imputation, rare/unseen categories, verification
- [docs/decision_register.md](docs/decision_register.md): every design decision, its evidence and status
- [configs/config.yaml](configs/config.yaml): frozen decisions used by the code

## Scope note
The supplied Kaggle file contains only departures from the three NYC airports (it matches the
public `nycflights13` dataset), so conclusions apply to that population only.
