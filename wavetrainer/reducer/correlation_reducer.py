"""A reducer that removes correlation features."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,consider-using-enumerate,too-many-locals
import json
import os
from typing import Self

import numpy as np
import optuna
import pandas as pd

from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns
from .reducer import Reducer

_CORRELATION_REDUCER_FILENAME = "correlation_reducer.json"
_CORRELATION_REDUCER_THRESHOLD = "correlation_reducer_threshold"


def _get_correlated_features_to_drop_chunked(
    df: pd.DataFrame,
    threshold: float = 0.85,
    chunk_size: int = 10000,
    random_seed: int = 42,
) -> list[str]:
    """
    Chunked correlation feature reducer to control memory usage.
    Applies correlation pruning within chunks, then across surviving features.
    """
    print(f"correlation reduction with threshold {threshold}")
    np.seterr(over="ignore")
    np.random.seed(random_seed)
    sorted_cols = sorted(find_non_categorical_numeric_columns(df))
    df_numeric = df[sorted_cols].copy()
    junk_value = np.random.uniform(-1e9, 1e9)
    df_numeric = df_numeric.fillna(junk_value).astype(np.float32)

    # First pass: intra-chunk correlation pruning
    survivors = []
    to_drop_total = set()
    for i in range(0, len(sorted_cols), chunk_size):
        chunk_cols = sorted_cols[i : i + chunk_size]
        chunk_df = df_numeric[chunk_cols]
        chunk_corr = np.corrcoef(chunk_df.values, rowvar=False)
        abs_corr = np.abs(chunk_corr)

        to_drop = set()
        for j in range(len(chunk_cols)):
            if chunk_cols[j] in to_drop:
                continue
            for k in range(j + 1, len(chunk_cols)):
                if chunk_cols[k] in to_drop:
                    continue
                if abs_corr[j, k] > threshold:
                    to_drop.add(chunk_cols[k])

        survivors.extend([col for col in chunk_cols if col not in to_drop])
        to_drop_total.update(to_drop)

    # Second pass: global correlation among survivors
    if len(survivors) < 2:
        return sorted(to_drop_total)

    survivors_df = df_numeric[survivors]
    final_corr = np.corrcoef(survivors_df.values, rowvar=False)
    abs_corr = np.abs(final_corr)

    final_drop = set()
    for i in range(len(survivors)):
        if survivors[i] in final_drop:
            continue
        for j in range(i + 1, len(survivors)):
            if survivors[j] in final_drop:
                continue
            if abs_corr[i, j] > threshold:
                final_drop.add(survivors[j])

    to_drop_total.update(final_drop)
    return sorted(to_drop_total)


class CorrelationReducer(Reducer):
    """A class that removes correlated values from a dataset."""

    _correlation_drop_features: dict[str, bool]

    def __init__(self, correlation_chunk_size: int) -> None:
        self._threshold = 0.0
        self._correlation_drop_features = {}
        self._correlation_chunk_size = correlation_chunk_size

    @classmethod
    def name(cls) -> str:
        return "correlation"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._threshold = trial.suggest_float(_CORRELATION_REDUCER_THRESHOLD, 0.7, 0.99)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _CORRELATION_REDUCER_FILENAME), encoding="utf8"
        ) as handle:
            self._correlation_drop_features = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _CORRELATION_REDUCER_FILENAME), "w", encoding="utf8"
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
        drop_features = _get_correlated_features_to_drop_chunked(
            df,
            threshold=self._threshold,
            chunk_size=self._correlation_chunk_size,
        )
        self._correlation_drop_features = {x: True for x in drop_features}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=list(self._correlation_drop_features.keys()), errors="ignore"
        )
