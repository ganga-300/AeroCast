"""Operating-threshold selection (Stage 8; decision D7).

A flight is flagged as "will be delayed" when its probability p ≥ t. Candidate thresholds are
the distinct predicted values. Precision is only defined for thresholds that flag at least one
flight, which every candidate does by construction.

Rule (D7): among thresholds with precision ≥ min_precision, choose the one with the highest
recall; ties in recall go to the highest threshold. If none qualifies, the result is
explicitly infeasible; the requirement is never relaxed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRECISION_TOL = 1e-12  # guards float rounding at exact ratios such as 7/10


def threshold_table(y, p) -> pd.DataFrame:
    """Precision and recall at every distinct threshold, highest threshold first."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype="float64")
    if y.sum() == 0:
        raise ValueError("Recall is undefined without positive examples")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last_of_value = np.r_[ps[1:] != ps[:-1], True]  # flagging p >= t includes all ties at t
    t = ps[last_of_value]
    tp, fp = tp[last_of_value], fp[last_of_value]
    return pd.DataFrame({"threshold": t, "tp": tp, "fp": fp, "n_flagged": tp + fp,
                         "precision": tp / (tp + fp), "recall": tp / y.sum()})


def select_threshold(y, p, min_precision: float) -> dict[str, Any]:
    table = threshold_table(y, p)
    ok = table["tp"] >= min_precision * table["n_flagged"] - PRECISION_TOL
    if not ok.any():
        best = table.loc[table["precision"].idxmax()]
        return {"feasible": False, "min_precision": min_precision,
                "max_attainable_precision": float(best["precision"]),
                "at_threshold": float(best["threshold"]), "recall_there": float(best["recall"]),
                "n_flagged_there": int(best["n_flagged"]), "n_candidates": int(len(table))}
    cand = table[ok]
    top_recall = cand["recall"].max()
    chosen = cand[cand["recall"] == top_recall].sort_values("threshold", ascending=False).iloc[0]
    return {"feasible": True, "min_precision": min_precision, "threshold": float(chosen["threshold"]),
            "precision": float(chosen["precision"]), "recall": float(chosen["recall"]),
            "tp": int(chosen["tp"]), "fp": int(chosen["fp"]), "n_flagged": int(chosen["n_flagged"]),
            "n_candidates": int(len(table)), "n_meeting_requirement": int(ok.sum())}


def recall_at_precision(y, p, min_precision: float) -> float:
    """Highest recall among thresholds with precision ≥ min_precision; 0.0 if none qualifies."""
    r = select_threshold(y, p, min_precision)
    return r["recall"] if r["feasible"] else 0.0


def metrics_at_threshold(y, p, threshold: float) -> dict[str, Any]:
    """Confusion matrix and threshold metrics for a fixed threshold (p ≥ threshold → positive)."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype="float64")
    flag = p >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else None)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": tp + fp, "flagged_share": float(flag.mean()),
            "precision": precision, "recall": recall, "f1": f1}
