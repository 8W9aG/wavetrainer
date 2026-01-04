"""A normaliser that uses the power transformer."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from sklearn.preprocessing import PowerTransformer  # type: ignore

from .normaliser import Normaliser

_POWERTRANSFORMER_REDUCER_FILE = "power_transformer_normaliser.joblib"


class PowerTransformerNormaliser(Normaliser):
    """A class that normalises the training data with the power transformer."""

    def __init__(self):
        super().__init__()
        self._pt = PowerTransformer()

    @classmethod
    def name(cls) -> str:
        return "powertransformer"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        self._pt = joblib.load(os.path.join(folder, _POWERTRANSFORMER_REDUCER_FILE))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        joblib.dump(self._pt, os.path.join(folder, _POWERTRANSFORMER_REDUCER_FILE))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        self._pt.fit(df.to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.values[:] = self._pt.transform(df.to_numpy())
        return df
