# Stage 0 — Workspace and Environment Audit

Audit date: 2026-10-24. Every item below was observed by running commands; nothing is assumed.

## Workspace
| Item | Finding |
|---|---|
| Working directory | `/Users/Ganga/Desktop/AeroCast` |
| Initial contents | Empty directory (no files, no hidden files) |
| Git | **Not a Git repository** (`git rev-parse` → "not a git repository"). No branch, no remotes. |
| Existing changes to preserve | None (directory was empty) |
| README / instructions / configs / tests / notebooks | None existed |
| Git policy (per user) | No `git init`, no commits, no GitHub linking. |
| OS / arch | macOS 26.6.2 (build 25G83), arm64 |
| Free disk space | ~100 GiB available on the data volume |

## Dataset supplied
| Item | Finding |
|---|---|
| Source archive | `/Users/Ganga/Downloads/archive (1).zip`, 10,797,806 bytes |
| Archive SHA-256 | `6e603cccc85d43c6c201543af454ee9a5fb0b6db2f77aa6e14dfc0b11cec468c` |
| Archive contents | One file: `flights.csv` (42,858,599 bytes, dated 2023-06-25 in the archive) |
| Extracted to | `data/raw/flights.csv`, **set read-only** (`chmod a-w`) so it cannot be modified in place |
| CSV SHA-256 | `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa` |
| Format | Plain-text, comma-separated, header row, 336,776 data rows |

No other dataset was downloaded, combined or used.

## Python environment
| Interpreter | Version | pandas | numpy | sklearn | xgboost | matplotlib |
|---|---|---|---|---|---|---|
| `~/miniconda3/bin/python3` (conda base, default on PATH) | 3.13.13 | missing | 2.5.2 | 1.9.0 | missing | missing |
| `/Library/Frameworks/.../3.14/bin/python3` | 3.14.5 | missing | missing | missing | missing | missing |
| `/usr/local/bin/python3` | 3.14.5 | missing | missing | missing | missing | missing |
| `/usr/bin/python3` (system) | 3.9.6 | missing | missing | missing | missing | missing |

Other tools: `uv 0.12.5` present; conda has only the `base` env; **`pdflatex` and `latexmk` are not installed** (the IEEE report cannot be compiled locally until a TeX distribution is available or Overleaf is used).

### Change made
No existing interpreter had pandas. To avoid modifying conda `base` or any system Python, an isolated
project virtual environment was created:

- `.venv/` (CPython 3.13.13, created with `uv venv` from the conda interpreter)
- Stage 1: `pandas 3.1.0` (pulled in `numpy 2.5.4`, `python-dateutil`, `six`, `tzdata`).
- Stage 2: `pyyaml 6.0.3` (config), `pytest 9.1.1` (tests), and the project itself in editable
  mode (`uv pip install -e .`, package `flight_delay` under `src/`).

Still to install in `.venv` when the relevant stage starts: matplotlib (EDA), scikit-learn and
xgboost (modelling). All versions are pinned in `requirements.txt`.

## Sensitive / must-not-share items
- No credentials, API keys or `.env` files exist in the workspace.
- `data/raw/flights.csv` (43 MB) and `.venv/` must never be shared/committed; a `.gitignore` will be
  added with the project skeleton as a safeguard, even though no repository exists.

## Blockers
None for Stage 1. Decisions blocking Stage 2+ are listed in `docs/decision_register.md`.
