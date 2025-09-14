"""A reducer that reduces embeddings using PCA."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
import hashlib
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from sklearn.decomposition import PCA  # type: ignore

from .reducer import Reducer

_PCA_THRESHOLD = "pca_threshold"


class PCAReducer(Reducer):
    """A class that reduces embeddings using PCA."""

    _pcas: dict[str, PCA]

    @classmethod
    def name(cls) -> str:
        return "pca"

    def __init__(self, embedding_cols: list[list[str]] | None):
        super().__init__()
        self._embedding_cols = embedding_cols if embedding_cols is not None else []
        self._pcas = {}

    @property
    def _embedding_dict(self) -> dict[str, list[str]]:
        return {
            hashlib.sha256("|".join(sorted(x)).encode()).hexdigest(): x
            for x in self._embedding_cols
        }

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        if self._embedding_cols is None:
            return
        threshold = trial.suggest_float(_PCA_THRESHOLD, 0.7, 0.99)
        self._pcas = {k: PCA(n_components=threshold) for k in self._embedding_dict}

    def load(self, folder: str) -> None:
        for k in self._embedding_dict:
            self._pcas[k] = joblib.load(os.path.join(folder, f"{k}_pca_reducer.joblib"))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        for k, v in self._pcas.items():
            joblib.dump(v, os.path.join(folder, f"{k}_pca_reducer.joblib"))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if self._embedding_cols is None:
            return self
        for k, v in self._pcas.items():
            x_arr = df[self._embedding_dict[k]].dropna()
            if len(x_arr) == 0:
                continue
            v.fit(x_arr)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._embedding_cols is None:
            return df
        for k, v in self._pcas.items():
            cols = self._embedding_dict[k]
            try:
                compressed_embedding = v.transform(df[cols].fillna(0.0))
                embedding_len = compressed_embedding.shape[1]
                df[cols[:embedding_len]] = compressed_embedding
                df = df.drop(columns=cols[embedding_len:])
            except AttributeError:
                pass
        return df
