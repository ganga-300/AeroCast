"""Load and validate configs/config.yaml (the frozen experimental decisions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

REQUIRED_KEYS: dict[str, list[str]] = {
    "data": ["raw_path", "raw_sha256"],
    "prediction": ["timestamp_rule", "history_lag_days"],
    "target": ["source_column", "threshold_minutes"],
    "population": ["require_recorded_label"],
    "split": ["train", "validation", "test"],
    "operating_point": ["min_precision", "rule", "tie_break"],
    "features": ["history"],
}
HISTORY_KEYS = ["entities", "entity_windows_days", "global_windows_days", "min_count", "smoothing_m"]
SUPPORTED_TIMESTAMP_RULES = {"end_of_previous_day"}


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config and fail loudly if a required field is missing or invalid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            raise ValueError(f"Config missing section '{section}'")
        for key in keys:
            # Presence is required even when the value is deliberately null (e.g. an OPEN decision).
            if key not in cfg[section]:
                raise ValueError(f"Config missing field '{section}.{key}'")
    if "seed" not in cfg:
        raise ValueError("Config missing field 'seed'")

    rule = cfg["prediction"]["timestamp_rule"]
    if rule not in SUPPORTED_TIMESTAMP_RULES:
        raise ValueError(f"Unsupported prediction.timestamp_rule '{rule}'")
    lag = cfg["prediction"]["history_lag_days"]
    if lag is not None and (not isinstance(lag, int) or lag < 1):
        raise ValueError("prediction.history_lag_days must be null or an integer >= 1")
    thr = cfg["target"]["threshold_minutes"]
    if not isinstance(thr, (int, float)):
        raise ValueError("target.threshold_minutes must be numeric")
    p = cfg["operating_point"]["min_precision"]
    if not isinstance(p, (int, float)) or not 0 < p <= 1:
        raise ValueError("operating_point.min_precision must be in (0, 1]")
    hist = cfg["features"]["history"]
    for key in HISTORY_KEYS:
        if key not in hist:
            raise ValueError(f"Config missing field 'features.history.{key}'")
    windows = hist["entity_windows_days"] + hist["global_windows_days"]
    if any(not isinstance(w, int) or w < 1 for w in windows):
        raise ValueError("History windows must be positive integers (days)")
    if not isinstance(hist["min_count"], int) or hist["min_count"] < 1:
        raise ValueError("features.history.min_count must be an integer >= 1")
    if not isinstance(hist["smoothing_m"], (int, float)) or hist["smoothing_m"] < 0:
        raise ValueError("features.history.smoothing_m must be >= 0")
    if cfg["population"]["require_recorded_label"] is not True:
        raise ValueError("population.require_recorded_label must be true (decision D4)")

    from flight_delay.data.split import parse_boundaries  # local import avoids a cycle
    parse_boundaries(cfg["split"])


def require_history_lag(cfg: dict[str, Any]) -> int:
    """Return the confirmed history lag, refusing to proceed while decision D10 is open."""
    lag = cfg["prediction"]["history_lag_days"]
    if lag is None:
        raise ValueError(
            "prediction.history_lag_days is null: decision D10 (historical-outcome cutoff) "
            "is still OPEN. Historical features cannot be built until it is confirmed."
        )
    return lag


def resolve_path(relative: str) -> Path:
    """Resolve a config path relative to the project root."""
    return PROJECT_ROOT / relative
