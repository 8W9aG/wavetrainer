"""A calibrator that implements conformal quantile calibration."""

# pylint: disable=too-many-arguments,too-many-positional-arguments

import json
import os
from typing import Self

import numpy as np
import optuna
import pandas as pd

from ..model.model import QUANTILE_COLUMN_PREFIX, QUANTILES, Model
from .calibrator import Calibrator

_CALIBRATOR_FILENAME = "conformalquantile.json"


class ConformalQuantileCalibrator(Calibrator):
    """A class that performs conformal quantile calibrator."""

    _conformal_quantile: dict[str, float]

    def __init__(self, model: Model):
        super().__init__(model)
        self._conformal_quantile = {}

    @classmethod
    def name(cls) -> str:
        return "conformalquantile"

    def predictions_as_x(self, y: pd.Series | pd.DataFrame | None = None) -> bool:
        return True

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _CALIBRATOR_FILENAME), encoding="utf8"
        ) as handle:
            self._conformal_quantile = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _CALIBRATOR_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(self._conformal_quantile, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        quantile_columns = sorted(
            [
                x
                for x in df.columns.values.tolist()
                if x.startswith(QUANTILE_COLUMN_PREFIX)
            ]
        )
        y_q = df[quantile_columns].to_numpy()
        self._conformal_quantile = {}
        for count, quantile in enumerate(QUANTILES):
            y_residual = y_q[:, count] - y
            c_alpha = np.quantile(y_residual, 1.0 - quantile)
            self._conformal_quantile[str(quantile)] = float(c_alpha)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        quantile_columns = sorted(
            [
                x
                for x in df.columns.values.tolist()
                if x.startswith(QUANTILE_COLUMN_PREFIX)
            ]
        )
        for count, quantile in enumerate(QUANTILES):
            df[quantile_columns[count]] -= self._conformal_quantile[str(quantile)]
        return df
