"""A reducer that combines all the other reducers."""

# pylint: disable=line-too-long
import json
import os
import time
from typing import Self

import optuna
import pandas as pd

from .constant_reducer import ConstantReducer
from .correlation_reducer import CorrelationReducer
from .duplicate_reducer import DuplicateReducer
from .fast_correlation_based_reducer import FastCorrelationBasedReducer
from .nonnumeric_reducer import NonNumericReducer
from .pca_reducer import PCAReducer
from .reducer import Reducer
from .select_by_single_feature_performance_reducer import \
    SelectBySingleFeaturePerformanceReducer
from .smart_correlation_reducer import SmartCorrelationReducer
from .unseen_reducer import UnseenReducer

_COMBINED_REDUCER_FILE = "combined_reducer.json"
_REDUCERS_KEY = "reducers"
_REMOVED_COLUMNS_FILE = "combined_reducer_removed_columns.json"


class CombinedReducer(Reducer):
    """A reducer that combines a series of reducers."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    _folder: str | None

    def __init__(
        self,
        embedding_cols: list[list[str]] | None,
        correlation_chunk_size: int | None,
        insert_null: bool = False,
        include_correlation_reducer: bool = True,
    ):
        super().__init__()
        if correlation_chunk_size is None:
            correlation_chunk_size = 500
        self._reducers = [
            UnseenReducer(insert_null),
            NonNumericReducer(),
            PCAReducer(embedding_cols),
            FastCorrelationBasedReducer(),
        ]
        if include_correlation_reducer:
            self._reducers.append(
                CorrelationReducer(correlation_chunk_size=correlation_chunk_size)
            )
        self._folder = None
        self._embedding_cols = embedding_cols
        self._correlation_chunk_size = correlation_chunk_size
        self._insert_null = insert_null

    @classmethod
    def name(cls) -> str:
        return "combined"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        for reducer in self._reducers:
            reducer.set_options(trial, df)

    def load(self, folder: str) -> None:
        self._reducers = []
        with open(
            os.path.join(folder, _COMBINED_REDUCER_FILE), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            for reducer_name in params[_REDUCERS_KEY]:
                if reducer_name == ConstantReducer.name():
                    self._reducers.append(ConstantReducer())
                elif reducer_name == DuplicateReducer.name():
                    self._reducers.append(DuplicateReducer())
                elif reducer_name == CorrelationReducer.name():
                    self._reducers.append(
                        CorrelationReducer(self._correlation_chunk_size)
                    )
                elif reducer_name == NonNumericReducer.name():
                    self._reducers.append(NonNumericReducer())
                elif reducer_name == UnseenReducer.name():
                    self._reducers.append(UnseenReducer(self._insert_null))
                elif reducer_name == SmartCorrelationReducer.name():
                    self._reducers.append(SmartCorrelationReducer())
                elif reducer_name == SelectBySingleFeaturePerformanceReducer.name():
                    self._reducers.append(SelectBySingleFeaturePerformanceReducer())
                elif reducer_name == PCAReducer.name():
                    self._reducers.append(PCAReducer(self._embedding_cols))
                elif reducer_name == FastCorrelationBasedReducer.name():
                    self._reducers.append(FastCorrelationBasedReducer())
        for reducer in self._reducers:
            reducer.load(folder)
        self._folder = folder

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _COMBINED_REDUCER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _REDUCERS_KEY: [x.name() for x in self._reducers],
                },
                handle,
            )
        for reducer in self._reducers:
            reducer.save(folder, trial)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        removed_columns_dict = {}
        for reducer in self._reducers:
            start_reducer = time.time()
            before_columns = set(df.columns.values)
            df = reducer.fit_transform(df, y=y)
            after_columns = set(df.columns.values)
            removed_columns = before_columns.difference(after_columns)
            if removed_columns:
                removed_columns_dict[reducer.name()] = list(removed_columns)
            print(
                f"{reducer.name()} reducer took {time.time() - start_reducer} and removed {len(removed_columns)} features",
            )
        if self._folder is not None:
            with open(
                os.path.join(self._folder, _REMOVED_COLUMNS_FILE), encoding="utf8"
            ) as handle:
                json.dump(removed_columns_dict, handle)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for reducer in self._reducers:
            try:
                df = reducer.transform(df)
            except ValueError as exc:
                print("Failed to reduce %s", reducer.name())
                raise exc
        return df
