"""A calibrator that implements venn abers."""

import logging
import os
from typing import Self

import joblib  # type: ignore
import numpy as np
import optuna
import pandas as pd
from venn_abers import VennAbers  # type: ignore

from ..model.model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model
from .calibrator import Calibrator

_CALIBRATOR_FILENAME = "vennabers.joblib"


class VennabersCalibrator(Calibrator):
    """A class that uses venn abers as a calibrator."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def __init__(self, model: Model):
        super().__init__(model)
        self._vennabers = VennAbers()

    @classmethod
    def name(cls) -> str:
        return "vennabers"

    def predictions_as_x(self, y: pd.Series | pd.DataFrame | None = None) -> bool:
        return True

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        self._vennabers = joblib.load(os.path.join(folder, _CALIBRATOR_FILENAME))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        joblib.dump(self._vennabers, os.path.join(folder, _CALIBRATOR_FILENAME))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        vennabers = self._vennabers
        if vennabers is None:
            raise ValueError("vennabers is null")
        if y is None:
            raise ValueError("y is null")
        prob_columns = sorted(
            [x for x in df.columns.values if x.startswith(PROBABILITY_COLUMN_PREFIX)]
        )
        probs = df[prob_columns].to_numpy()
        try:
            vennabers.fit(probs, y.to_numpy())
        except IndexError:
            logging.error(df)
            raise
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prob_columns = sorted(
            [x for x in df.columns.values if x.startswith(PROBABILITY_COLUMN_PREFIX)]
        )
        probs = df[prob_columns].to_numpy()
        df[PREDICTION_COLUMN] = probs.argmax(axis=1).astype(np.bool_)
        p_prime, _ = self._vennabers.predict_proba(probs)
        for i in range(p_prime.shape[1]):
            prob = p_prime[:, i]
            df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = prob
        return df
