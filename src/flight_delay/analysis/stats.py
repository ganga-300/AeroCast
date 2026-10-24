"""Small, testable helpers for descriptive analysis.

Rates are always reported with their denominators and Wilson 95% intervals, so that small
groups are not presented as stable estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054


def wilson_interval(k, n, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion k/n. Returns NaN where n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = centre - half, centre + half
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


def rate_table(df: pd.DataFrame, by, y: str = "y") -> pd.DataFrame:
    """Positive rate of labelled rows per group, with counts and Wilson 95% CI.

    Rows with a missing label are ignored (never treated as negatives).
    """
    lab = df[df[y].notna()]
    g = lab.groupby(by, observed=True)[y]
    out = pd.DataFrame({"n": g.size(), "positives": g.sum().astype(int)})
    out["rate"] = out["positives"] / out["n"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["positives"], out["n"])
    return out


def category_shares(values: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Share of each category within each group (columns = groups, each column sums to 1)."""
    return pd.crosstab(values, groups, normalize="columns")


def total_variation_distance(p: pd.Series, q: pd.Series) -> float:
    """0.5 * sum |p - q| over the union of categories (0 = identical, 1 = disjoint)."""
    p, q = p.align(q, fill_value=0.0)
    return float(0.5 * (p - q).abs().sum())


def unseen_categories(train: pd.Series, other: pd.Series) -> dict:
    """Categories in `other` that never occur in `train`, and how many rows they cover."""
    unseen = sorted(set(other.dropna()) - set(train.dropna()))
    return {"categories": unseen, "n_categories": len(unseen),
            "rows": int(other.isin(unseen).sum()),
            "row_share": round(float(other.isin(unseen).mean()), 5) if len(other) else None}


def autocorrelation(series: pd.Series, lags) -> dict[int, float]:
    """Pearson autocorrelation of an evenly spaced series at the given lags."""
    return {int(l): round(float(series.autocorr(lag=l)), 4) for l in lags}
