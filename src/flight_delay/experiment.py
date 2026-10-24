"""Chronological experimental design and the test-label lock (Stages 5 and 8; D22, D23, D32, D34).

Responsibilities:

1. **Partitions.** Split the labelled modelling population by flight date and assert the
   chronological guarantees. `partition()` returns training and validation rows with labels
   but test rows **without** label columns.
2. **History vs. fitting data.** Historical features are computed once over all dates
   (`features.build`). Model fitting may only ever see training rows (`assert_fit_rows`).
3. **Test-label lock.** Test labels are released only by `unlock_test_labels`, which reads the
   manifest and writes the access log at fixed official paths. The manifest must be
   `status: "FROZEN"` and must bind exactly to the approved proposal and to the files on disk:
   proposal SHA-256, config, raw data, model and calibrator artifacts, the evaluation script and
   the `src` package code. Decision fields must equal the proposal, and the operating point must
   be the D32 statement. Each release is logged with the manifest hash. A repeat release needs
   an explicit reason, which is logged.

The lock is a procedural safeguard and an audit trail, not access control. The raw CSV
contains the outcomes, and any code could read it directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.config import PROJECT_ROOT
from flight_delay.data.load import modelling_population
from flight_delay.data.split import SPLIT_ORDER, assert_chronological, parse_boundaries

LABEL_COLUMNS = ("y", "label_status")
FROZEN_STATUS = "FROZEN"
D32_STATEMENT = "No operating threshold — the 70% precision target was infeasible on validation."

# Official locations, relative to the project root (not caller-chosen).
MANIFEST_REL = Path("reports/final/frozen_manifest.json")
ACCESS_LOG_REL = Path("reports/final/test_label_access_log.jsonl")
PROPOSAL_REL = Path("reports/freeze/frozen_decisions_proposal.json")
CONFIG_REL = Path("configs/config.yaml")
EVALUATION_SCRIPT_REL = Path("scripts/evaluate_test.py")
CODE_DIR_REL = Path("src/flight_delay")

# Manifest fields that must equal the approved proposal exactly.
BOUND_KEYS = ("primary_model", "comparison_model", "models", "operating_point_D32",
              "prediction_contract", "calibration_and_threshold_procedure_D11",
              "test_evaluation_plan", "evaluation_script", "code_sha256", "reproducibility")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class HoldoutLockedError(RuntimeError):
    """Raised when test labels would be read without a valid FROZEN manifest."""


# ---------------------------------------------------------------- hashing helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(root: Path = PROJECT_ROOT) -> str:
    """Hash of every .py file in the package (relative path + content), in sorted order."""
    h = hashlib.sha256()
    base = root / CODE_DIR_REL
    for p in sorted(base.rglob("*.py")):
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- partitions

def _labelled_partitions(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Labelled rows of every partition with chronological assertions. Private: only the lock
    and the pass/fail sufficiency check may see test labels."""
    pop = modelling_population(table)
    assert_chronological(pop["flight_date"], pop["split"])
    bounds = parse_boundaries(cfg["split"])
    parts = {}
    for name in SPLIT_ORDER:
        part = pop[pop["split"] == name]
        r = bounds[name]
        if not part["flight_date"].between(r.start, r.end).all():
            raise AssertionError(f"Rows outside the configured range in '{name}'")
        expected_days = pd.date_range(r.start, r.end, freq="D")
        missing = expected_days.difference(pd.DatetimeIndex(part["flight_date"].unique()))
        if len(missing):
            raise AssertionError(f"'{name}' has dates with no labelled flights: {list(missing[:5])}")
        parts[name] = part
    ids = [set(p["id"]) for p in parts.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1:]):
        raise AssertionError("A flight id appears in more than one partition")
    return parts


def partition(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Partitions of the labelled population; the test partition has its label columns removed."""
    parts = _labelled_partitions(table, cfg)
    parts["test"] = parts["test"].drop(columns=list(LABEL_COLUMNS))
    return parts


def development_partitions(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) labelled rows. The only data source for any modelling decision."""
    parts = partition(table, cfg)
    return parts["train"], parts["validation"]


def holdout_features_without_labels(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Test-period rows of the modelling population, without label columns."""
    return partition(table, cfg)["test"]


def test_partition_sufficient(table: pd.DataFrame, cfg: dict[str, Any], min_class_count: int) -> bool:
    """PASS/FAIL only: does the test partition contain >= min_class_count of each class?
    Counts are never returned."""
    y = _labelled_partitions(table, cfg)["test"]["y"]
    return bool(min(int((y == 1).sum()), int((y == 0).sum())) >= min_class_count)


test_partition_sufficient.__test__ = False  # not a pytest test despite the name


def assert_fit_rows(fit: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Model fitting (and any learned preprocessing) may use labelled training rows only."""
    r = parse_boundaries(cfg["split"])["train"]
    if not (fit["split"] == "train").all():
        raise AssertionError("Fitting data contains non-training rows")
    if not fit["flight_date"].between(r.start, r.end).all():
        raise AssertionError("Fitting data contains dates outside the training range")
    if fit["y"].isna().any():
        raise AssertionError("Fitting data contains unlabelled rows")


# ---------------------------------------------------------------- manifest

def build_manifest(approval_note: str, root: Path = PROJECT_ROOT, frozen_at: str | None = None) -> dict[str, Any]:
    """Construct (but do not write) a FROZEN manifest from the proposal currently on disk.
    Writing it to the official path is a separate, user-authorised step."""
    proposal_path = root / PROPOSAL_REL
    proposal = json.loads(proposal_path.read_text())
    return {"status": FROZEN_STATUS,
            "frozen_at": frozen_at or datetime.now(timezone.utc).isoformat(),
            "approval_note": approval_note,
            "approved_proposal_sha256": sha256_file(proposal_path),
            **{k: proposal[k] for k in BOUND_KEYS}}


def _fail(msg: str) -> None:
    raise HoldoutLockedError(msg + "; test labels stay locked")


def validate_manifest(manifest: Any, root: Path = PROJECT_ROOT) -> None:
    """Reject anything that is not a FROZEN manifest bound to the approved proposal and to the
    exact files on disk. Raises HoldoutLockedError on every failure."""
    if not isinstance(manifest, dict):
        _fail("Manifest is not a JSON object")
    if manifest.get("status") != FROZEN_STATUS:
        _fail(f"Manifest status is {manifest.get('status')!r}, not {FROZEN_STATUS!r}")
    missing = [k for k in ("frozen_at", "approval_note", "approved_proposal_sha256", *BOUND_KEYS)
               if k not in manifest]
    if missing:
        _fail(f"Frozen manifest incomplete; missing: {missing}")
    try:
        ts = datetime.fromisoformat(str(manifest["frozen_at"]))
    except ValueError:
        ts = None
    if ts is None or ts.tzinfo is None:
        _fail("frozen_at must be an ISO-8601 timestamp with a time zone")
    if not isinstance(manifest["approval_note"], str) or not manifest["approval_note"].strip():
        _fail("approval_note must be a non-empty string")

    proposal_path = root / PROPOSAL_REL
    if not proposal_path.exists():
        _fail(f"Approved proposal not found at {PROPOSAL_REL}")
    if not HEX64.match(str(manifest["approved_proposal_sha256"])) or \
            manifest["approved_proposal_sha256"] != sha256_file(proposal_path):
        _fail("approved_proposal_sha256 does not match the proposal on disk")
    proposal = json.loads(proposal_path.read_text())
    differing = [k for k in BOUND_KEYS if manifest[k] != proposal.get(k)]
    if differing:
        _fail(f"Manifest fields differ from the approved proposal: {differing}")

    if manifest["operating_point_D32"] != D32_STATEMENT:
        _fail("Operating point is not the D32 statement")
    if set(manifest["models"]) != {manifest["primary_model"], manifest["comparison_model"]}:
        _fail("Manifest models do not match primary/comparison model")
    for name, m in manifest["models"].items():
        thr = m.get("operating_threshold", {})
        if thr.get("status") != "none" or thr.get("statement") != D32_STATEMENT:
            _fail(f"Model {name} has an operating threshold; D32 requires none")
        if not m.get("hyperparameters") or not m.get("calibration", {}).get("method"):
            _fail(f"Model {name} lacks hyperparameters or calibration method")
        for kind in ("model", "calibrator"):
            art = m.get("artifacts", {}).get(kind, {})
            path = root / str(art.get("path", ""))
            if not art.get("path") or not path.is_file():
                _fail(f"{name} {kind} artifact missing: {art.get('path')}")
            if art.get("sha256") != sha256_file(path):
                _fail(f"{name} {kind} artifact hash differs from the manifest")
    if not manifest["prediction_contract"].get("features"):
        _fail("Feature list is empty")

    repro = manifest["reproducibility"]
    if repro.get("config_sha256") != sha256_file(root / CONFIG_REL):
        _fail("Config hash differs from the manifest")
    raw_rel = manifest["prediction_contract"].get("raw_path")
    if not raw_rel or repro.get("raw_sha256") != sha256_file(root / raw_rel):
        _fail("Raw data hash differs from the manifest")
    script = manifest["evaluation_script"]
    if script.get("path") != str(EVALUATION_SCRIPT_REL) or \
            script.get("sha256") != sha256_file(root / EVALUATION_SCRIPT_REL):
        _fail("Evaluation script differs from the approved version")
    if manifest["code_sha256"] != code_sha256(root):
        _fail("Package code differs from the approved version")


def load_and_validate_manifest(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], str]:
    """Read the official manifest, validate it, and return (manifest, its SHA-256)."""
    path = root / MANIFEST_REL
    if not path.exists():
        _fail(f"No frozen manifest at {MANIFEST_REL}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _fail(f"Manifest is not valid JSON ({e.msg})")
    validate_manifest(manifest, root)
    return manifest, sha256_file(path)


def previous_accesses(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    log = root / ACCESS_LOG_REL
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []


def unlock_test_labels(table: pd.DataFrame, cfg: dict[str, Any], purpose: str,
                       root: Path = PROJECT_ROOT, repeat_reason: str | None = None) -> pd.Series:
    """Release test labels (indexed by flight id) under a valid FROZEN manifest.

    Every release is appended to the official access log with the manifest hash. A release
    after an earlier one is refused unless `repeat_reason` is given (e.g. a disclosed
    software-defect re-run), and that reason is logged.
    """
    manifest, manifest_sha = load_and_validate_manifest(root)
    earlier = previous_accesses(root)
    if earlier and not (repeat_reason and repeat_reason.strip()):
        _fail(f"Test labels were already released {len(earlier)} time(s); a repeat needs a logged reason")
    test = _labelled_partitions(table, cfg)["test"]
    log = root / ACCESS_LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"access_number": len(earlier) + 1,
                            "accessed_at": datetime.now(timezone.utc).isoformat(),
                            "purpose": purpose, "repeat_reason": repeat_reason,
                            "manifest_path": str(MANIFEST_REL), "manifest_sha256": manifest_sha,
                            "approved_proposal_sha256": manifest["approved_proposal_sha256"],
                            "manifest_frozen_at": manifest["frozen_at"],
                            "n_rows": int(len(test))}) + "\n")
    return test.set_index("id")["y"]
