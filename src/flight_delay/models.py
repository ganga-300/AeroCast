"""Baseline models on top of the training-only preprocessing (Stage 7; D20, D21, D25-D27).

`FlightDelayModel` = `TrainOnlyPreprocessor` + estimator. It is fitted on labelled training
rows only and never refitted on validation or test (D20). XGBoost may use validation rows
for early stopping (D21); nothing else sees validation labels. Class-imbalance weights are
derived from training labels only.

Families:
* `dummy_most_frequent`: always predicts the majority class (P(delay) = 0). A required trivial
  reference; high accuracy under imbalance, but it detects no delays.
* `prior_baseline`: always predicts the training prevalence. A separate probabilistic reference
  for Brier score and log loss; not the same as the most-frequent dummy.
* `logistic_regression`: L2-regularised, lbfgs.
* `xgboost`: gradient-boosted trees (`hist`), early stopping on validation.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from flight_delay.experiment import assert_fit_rows
from flight_delay.features.specs import FeatureSpec
from flight_delay.preprocessing import TrainOnlyPreprocessor

FAMILIES = ("dummy_most_frequent", "prior_baseline", "logistic_regression", "xgboost")
PREPROCESSOR_FOR = {"dummy_most_frequent": "logistic_regression",
                    "prior_baseline": "logistic_regression",
                    "logistic_regression": "logistic_regression", "xgboost": "xgboost"}
TRAIN_BALANCED = "train_balanced"


def training_scale_pos_weight(y_train: pd.Series) -> float:
    """negatives / positives in the TRAINING labels (never validation or test)."""
    y = pd.Series(y_train).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("Training labels must contain both classes")
    return neg / pos


class FlightDelayModel:
    def __init__(self, family: str, params: dict[str, Any], specs: list[FeatureSpec],
                 cfg: dict[str, Any], seed: int):
        if family not in FAMILIES:
            raise ValueError(f"Unknown model family '{family}'")
        self.family, self.params, self.cfg, self.seed = family, dict(params), cfg, seed
        self.pre = TrainOnlyPreprocessor(PREPROCESSOR_FOR[family], specs, cfg)
        self.estimator = None
        self.train_info_: dict[str, Any] | None = None

    def _build(self, y_train: pd.Series):
        p = self.params
        if self.family == "dummy_most_frequent":
            return DummyClassifier(strategy="most_frequent"), {}
        if self.family == "prior_baseline":
            return DummyClassifier(strategy="prior"), {}
        if self.family == "logistic_regression":
            resolved = {"C": p["C"], "class_weight": p["class_weight"], "penalty": "l2",
                        "solver": "lbfgs", "max_iter": p["max_iter"]}
            return LogisticRegression(**{k: v for k, v in resolved.items() if k != "penalty"},
                                      random_state=self.seed), resolved
        spw = p["scale_pos_weight"]
        resolved = {
            "objective": "binary:logistic", "tree_method": "hist",
            "max_depth": p["max_depth"], "min_child_weight": p["min_child_weight"],
            "learning_rate": p["learning_rate"], "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"], "n_estimators": p["n_estimators_max"],
            "early_stopping_rounds": p["early_stopping_rounds"],
            "eval_metric": p["early_stopping_metric"],
            "scale_pos_weight": training_scale_pos_weight(y_train) if spw == TRAIN_BALANCED else float(spw),
        }
        return XGBClassifier(**resolved, random_state=self.seed, n_jobs=-1), resolved

    def fit(self, train_rows: pd.DataFrame, eval_rows: pd.DataFrame | None = None) -> "FlightDelayModel":
        if self.estimator is not None:
            raise RuntimeError("Model already fitted; create a new instance to refit")
        assert_fit_rows(train_rows, self.cfg)
        if (eval_rows is None) != (self.family != "xgboost"):
            raise ValueError("Early-stopping rows are required for XGBoost and forbidden otherwise")
        if eval_rows is not None and not ((eval_rows["split"] == "validation").all()
                                          and eval_rows["y"].notna().all()):
            raise AssertionError("Early stopping may use labelled validation rows only")
        self.pre.fit(train_rows)
        X, y = self.pre.transform(train_rows), train_rows["y"].astype(int)
        self.estimator, resolved = self._build(y)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if self.family == "xgboost":
                Xe, ye = self.pre.transform(eval_rows), eval_rows["y"].astype(int)
                self.estimator.fit(X, y, eval_set=[(Xe, ye)], verbose=False)
            else:
                self.estimator.fit(X, y)
        info: dict[str, Any] = {
            "family": self.family, "params": self.params, "resolved_params": resolved,
            "seed": self.seed, "fit_seconds": round(time.perf_counter() - start, 2),
            "preprocessing": self.pre.fit_info_, "n_features_in": int(X.shape[1]),
            "train_positives": int(y.sum()), "train_rows": int(len(y)),
        }
        if self.family == "logistic_regression":
            n_iter = int(np.max(self.estimator.n_iter_))
            info["n_iter"] = n_iter
            info["converged"] = (n_iter < self.params["max_iter"]
                                 and not any(issubclass(w.category, ConvergenceWarning) for w in caught))
        if self.family == "xgboost":
            info["best_iteration"] = int(self.estimator.best_iteration)
            info["n_trees_used"] = int(self.estimator.best_iteration) + 1
            info["early_stopping_rows"] = int(len(eval_rows))
        self.train_info_ = info
        return self

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        """P(arrival delay ≥ 15 min) for each row, aligned to `rows.index`."""
        if self.estimator is None:
            raise RuntimeError("Model must be fitted before predicting")
        X = self.pre.transform(rows)
        if self.family == "xgboost":
            p = self.estimator.predict_proba(X, iteration_range=(0, self.train_info_["n_trees_used"]))[:, 1]
        else:
            p = self.estimator.predict_proba(X)[:, list(self.estimator.classes_).index(1)] \
                if 1 in self.estimator.classes_ else np.zeros(len(X))
        p = pd.Series(np.asarray(p, dtype="float64"), index=rows.index, name="p")
        if len(p) != len(rows) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise AssertionError("Invalid predicted probabilities")
        return p
