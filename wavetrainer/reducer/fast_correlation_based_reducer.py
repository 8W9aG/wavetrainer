"""A reducer that implements Fast Correlation-Based Feature selection."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,consider-using-enumerate,too-many-locals,invalid-name
import json
import os
from typing import Self

import optuna
import pandas as pd

from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns
from .reducer import Reducer

_FAST_CORRELATION_REDUCER_FILENAME = "fast_correlation_reducer.json"
_FAST_CORRELATION_THRESHOLD = "fast_correlation_threshold"


def _get_fast_correlated_features_to_drop(
    df: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.7,
) -> list[str]:
    """
    Selects features by sorting them by their absolute correlation with the target,
    then iterating through to remove features that are highly correlated with
    already selected (better) features.
    """
    print(f"fast correlation reduction with threshold {threshold}")

    # Filter for numeric columns only to avoid errors during correlation
    sorted_cols = sorted(find_non_categorical_numeric_columns(df))
    X = df[sorted_cols]

    # Align X and y indices to ensure correct correlation calculation
    # (Crucial if df and y have been shuffled or filtered differently)
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y_aligned = y.loc[common_index]

    if X.empty:
        return []

    # 1. CALCULATE RELEVANCE (Correlation with Target)
    # efficient vectorized calculation
    initial_correlations = X.corrwith(y_aligned).abs()

    # 2. SORTING
    # Sort features by relevance to the target (highest first)
    sorted_features = initial_correlations.sort_values(ascending=False).index.tolist()

    selected_features: list[str] = []

    # 3. GREEDY REDUNDANCY REMOVAL
    # Check candidates against only the features we have ALREADY selected.
    for candidate in sorted_features:
        is_redundant = False

        for accepted_feature in selected_features:
            # Compute pairwise correlation on the fly
            pairwise_corr = X[candidate].corr(X[accepted_feature])

            if abs(pairwise_corr) > threshold:
                is_redundant = True
                # Optimization: Break immediately once redundancy is found.
                break

        if not is_redundant:
            selected_features.append(candidate)

    # Convert selected features into a list of features to DROP
    # The reducer framework typically expects a list of columns to remove.
    selected_set = set(selected_features)
    to_drop = [col for col in sorted_cols if col not in selected_set]

    return sorted(to_drop)


class FastCorrelationBasedReducer(Reducer):
    """
    A class that removes features that are highly correlated with other features
    that have a stronger correlation with the target variable.
    """

    _correlation_drop_features: dict[str, bool]

    def __init__(self) -> None:
        self._threshold = 0.0
        self._correlation_drop_features = {}

    @classmethod
    def name(cls) -> str:
        return "fast_correlation"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        # Range matching the typical usage for this algorithm
        self._threshold = trial.suggest_float(_FAST_CORRELATION_THRESHOLD, 0.7, 0.99)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _FAST_CORRELATION_REDUCER_FILENAME), encoding="utf8"
        ) as handle:
            self._correlation_drop_features = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _FAST_CORRELATION_REDUCER_FILENAME),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(self._correlation_drop_features, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        # This reducer requires a target variable 'y' to calculate relevance.
        if y is None:
            print("FastCorrelationBasedReducer skipped: No target 'y' provided.")
            self._correlation_drop_features = {}
            return self

        # Handle case where y is passed as a DataFrame (e.g. single column DF)
        if isinstance(y, pd.DataFrame):
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        drop_features = _get_fast_correlated_features_to_drop(
            df,
            y=y_series,
            threshold=self._threshold,
        )
        self._correlation_drop_features = {x: True for x in drop_features}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=list(self._correlation_drop_features.keys()), errors="ignore"
        )
