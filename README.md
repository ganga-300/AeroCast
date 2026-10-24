# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning — Phase 1 (ML)

Predict whether a flight departing New York City (EWR/JFK/LGA, 2013) will arrive at least
15 minutes late, using only information available at the end of the day before the flight.

**Status:** Stages 0–2 complete (workspace audit, dataset audit, prediction contract). EDA,
features, models and the report are not implemented yet; no model results exist.

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
| Tests | `.venv/bin/python -m pytest -q` | — |

## Key documents
- [docs/00_workspace_audit.md](docs/00_workspace_audit.md): environment and workspace facts
- [docs/01_dataset_audit_summary.md](docs/01_dataset_audit_summary.md): interpreted data audit
- [docs/02_prediction_contract.md](docs/02_prediction_contract.md): prediction timestamp, target, population, leakage policy, split
- [docs/decision_register.md](docs/decision_register.md): every design decision, its evidence and status
- [configs/config.yaml](configs/config.yaml): frozen decisions used by the code

## Scope note
The supplied Kaggle file contains only departures from the three NYC airports (it matches the
public `nycflights13` dataset), so conclusions apply to that population only.
