"""Training-only preprocessing pipelines (Stage 6; decisions D18, D19, D24).

One scikit-learn ColumnTransformer per model family, wrapped in `TrainOnlyPreprocessor`:

* `fit` accepts labelled TRAINING rows only (`experiment.assert_fit_rows`). Imputation
  statistics, scaling parameters and category mappings are learned from training data alone.
* `transform` selects the declared input columns by name (input column order and extra
  columns such as labels are irrelevant), never refits, and validates the output: the same
  columns in the same order for every partition, float dtype, and missing values only where
  the model family allows them.

Per-family encodings are fixed a priori (not chosen on validation or test performance):

| group | Logistic Regression | XGBoost |
|---|---|---|
| time of day | sin/cos of departure and arrival hour, standardised | integer departure/arrival hour |
| day of week | one-hot | integer 0-6 |
| weekend flag | passed through | passed through |
| distance | standardised | passed through |
| historical rates | median-imputed (training medians) then standardised | passed through; NaN kept for native handling |
| historical counts | log1p then standardised | passed through |
| history_unavailable | 1 if any historical rate is missing (stateless; same rule everywhere) | same |
| carrier, origin, dest | one-hot, min training frequency, unseen -> infrequent column or all zeros | same one-hot |
| route | not encoded; represented by historical route features (D19) | same |
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from flight_delay.experiment import assert_fit_rows
from flight_delay.features.specs import FeatureSpec

FAMILIES = ("logistic_regression", "xgboost")
CYCLIC = ["dep_hour_sin", "dep_hour_cos", "arr_hour_sin", "arr_hour_cos"]
RAW_HOURS = ["dep_hour", "arr_hour"]
ONE_HOT = ["carrier", "origin", "dest"]
NOT_ENCODED = ["route"]
INDICATOR = "history_unavailable"


def history_unavailable(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless indicator: 1.0 if any historical rate is missing (cold start), else 0.0."""
    return X.isna().any(axis=1).astype("float64").to_frame(INDICATOR)


def _indicator_names(transformer, input_features) -> np.ndarray:
    return np.array([INDICATOR], dtype=object)


def column_groups(specs: list[FeatureSpec]) -> dict[str, list[str]]:
    """Partition the declared features into preprocessing groups; every feature must land in
    exactly one group so nothing is silently dropped or duplicated."""
    names = [s.name for s in specs]
    hist = [s for s in specs if s.group == "history"]
    groups = {
        "cyclic": CYCLIC, "raw_hours": RAW_HOURS, "day_of_week": ["day_of_week"],
        "weekend": ["is_weekend"], "distance": ["distance"], "one_hot": ONE_HOT,
        "not_encoded": NOT_ENCODED,
        "hist_rates": [s.name for s in hist if "_rate_" in s.name or "_share_" in s.name],
        "hist_counts": [s.name for s in hist if "_n_" in s.name],
    }
    placed = [c for cols in groups.values() for c in cols]
    if sorted(placed) != sorted(names) or len(placed) != len(set(placed)):
        raise ValueError(f"Features not placed exactly once: "
                         f"{sorted(set(names) ^ set(placed))}")
    return groups


def build_column_transformer(family: str, specs: list[FeatureSpec], cfg: dict[str, Any]) -> tuple[ColumnTransformer, list[str]]:
    """Unfitted transformer for a model family, plus the ordered list of input columns it reads."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family '{family}'")
    g = column_groups(specs)
    one_hot = OneHotEncoder(min_frequency=cfg["preprocessing"]["min_category_frequency"],
                            handle_unknown="infrequent_if_exist", sparse_output=False)
    indicator = FunctionTransformer(history_unavailable, feature_names_out=_indicator_names)
    if family == "logistic_regression":
        steps = [
            ("cyclic", StandardScaler(), g["cyclic"]),
            ("day_of_week", OneHotEncoder(handle_unknown="ignore", sparse_output=False), g["day_of_week"]),
            ("weekend", "passthrough", g["weekend"]),
            ("distance", StandardScaler(), g["distance"]),
            ("hist_rates", Pipeline([("impute", SimpleImputer(strategy="median")),
                                     ("scale", StandardScaler())]), g["hist_rates"]),
            ("hist_counts", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                      ("scale", StandardScaler())]), g["hist_counts"]),
            ("indicator", indicator, g["hist_rates"]),
            ("one_hot", one_hot, g["one_hot"]),
        ]
    else:
        steps = [
            ("raw_hours", "passthrough", g["raw_hours"]),
            ("day_of_week", "passthrough", g["day_of_week"]),
            ("weekend", "passthrough", g["weekend"]),
            ("distance", "passthrough", g["distance"]),
            ("hist_rates", "passthrough", g["hist_rates"]),
            ("hist_counts", "passthrough", g["hist_counts"]),
            ("indicator", indicator, g["hist_rates"]),
            ("one_hot", one_hot, g["one_hot"]),
        ]
    inputs = list(dict.fromkeys(c for _, _, cols in steps for c in cols))
    ct = ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas"), inputs


class TrainOnlyPreprocessor:
    """Fits on labelled training rows only; transforms any partition with identical output columns."""

    def __init__(self, family: str, specs: list[FeatureSpec], cfg: dict[str, Any]):
        self.family = family
        self.cfg = cfg
        self.specs = {s.name: s for s in specs}
        self.transformer, self.input_columns = build_column_transformer(family, specs, cfg)
        self.hist_rate_columns = column_groups(specs)["hist_rates"]
        self.feature_names_out_: list[str] | None = None
        self.fit_info_: dict | None = None

    def _select_inputs(self, rows: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns if c not in rows.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")
        X = rows[self.input_columns].copy()
        for c in self.input_columns:
            numeric = pd.api.types.is_numeric_dtype(X[c])
            if self.specs[c].kind == "numeric" and not numeric:
                raise TypeError(f"Feature '{c}' must be numeric, got {X[c].dtype}")
        cats = [c for c in ONE_HOT if c in X.columns]
        if X[cats].isna().any().any():
            raise ValueError("Categorical inputs contain missing values")
        X[cats] = X[cats].astype(object)  # plain strings for the encoder
        return X

    def fit(self, rows: pd.DataFrame) -> "TrainOnlyPreprocessor":
        if self.fit_info_ is not None:
            raise RuntimeError("Preprocessor is already fitted; create a new one to refit")
        assert_fit_rows(rows, self.cfg)
        self.transformer.fit(self._select_inputs(rows))
        self.feature_names_out_ = list(self.transformer.get_feature_names_out())
        ids = np.sort(rows["id"].to_numpy())
        self.fit_info_ = {
            "family": self.family, "n_rows": int(len(rows)),
            "date_min": str(rows["flight_date"].min().date()),
            "date_max": str(rows["flight_date"].max().date()),
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "n_output_columns": len(self.feature_names_out_),
        }
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.fit_info_ is None:
            raise RuntimeError("Preprocessor must be fitted on training rows before transform")
        # Pass-through integer columns (hours, weekday, weekend) become float64 model inputs.
        out = self.transformer.transform(self._select_inputs(rows)).astype("float64")
        self.validate_output(out)
        return out.set_axis(rows.index)

    def validate_output(self, out: pd.DataFrame) -> None:
        if list(out.columns) != self.feature_names_out_:
            raise AssertionError("Output columns differ from those learned at fit time")
        if not all(pd.api.types.is_float_dtype(t) for t in out.dtypes):
            raise AssertionError("Output must be floating point")
        values = out.to_numpy()
        if np.isinf(values).any():
            raise AssertionError("Output contains infinite values")
        nan_cols = out.columns[out.isna().any()].tolist()
        allowed = [] if self.family == "logistic_regression" else self.hist_rate_columns
        bad = [c for c in nan_cols if c not in allowed]
        if bad:
            raise AssertionError(f"Missing values in columns that must be complete: {bad}")

    def learned_state(self) -> dict[str, Any]:
        """Training-derived parameters, for documentation and change detection."""
        state: dict[str, Any] = {}
        for name, trans, cols in self.transformer.transformers_:
            if name == "one_hot":
                state["one_hot_categories"] = {c: [str(v) for v in cats]
                                               for c, cats in zip(cols, trans.categories_)}
                state["one_hot_infrequent"] = {
                    c: ([str(v) for v in inf] if inf is not None else [])
                    for c, inf in zip(cols, trans.infrequent_categories_)}
            elif isinstance(trans, Pipeline) and "impute" in trans.named_steps:
                state["hist_rate_impute_medians"] = dict(zip(cols, trans.named_steps["impute"].statistics_.round(6).tolist()))
            if isinstance(trans, StandardScaler):
                state[f"{name}_scaler_mean"] = dict(zip(cols, trans.mean_.round(6).tolist()))
        return state
