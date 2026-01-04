"""A normaliser that combines all the other normalisers."""

# pylint: disable=line-too-long
import json
import os
from typing import Self

import optuna
import pandas as pd

from .normaliser import Normaliser
from .powertransformer_normaliser import PowerTransformerNormaliser

_COMBINED_NORMALISER_FILE = "combined_normaliser.json"
_NORMALISERS_KEY = "normalisers"


class CombinedNormaliser(Normaliser):
    """A normaliser that combines a series of normalisers."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    _folder: str | None

    def __init__(
        self,
        use_power_transformer: bool = False,
    ):
        super().__init__()
        self._normalisers = []
        if use_power_transformer:
            self._normalisers.append(PowerTransformerNormaliser())
        self._folder = None

    @classmethod
    def name(cls) -> str:
        return "combined"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        for normaliser in self._normalisers:
            normaliser.set_options(trial, df)

    def load(self, folder: str) -> None:
        self._normalisers = []
        with open(
            os.path.join(folder, _COMBINED_NORMALISER_FILE), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            for normaliser_name in params[_NORMALISERS_KEY]:
                if normaliser_name == PowerTransformerNormaliser.name():
                    self._normalisers.append(PowerTransformerNormaliser())
        for normaliser in self._normalisers:
            normaliser.load(folder)
        self._folder = folder

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _COMBINED_NORMALISER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _NORMALISERS_KEY: [x.name() for x in self._normalisers],
                },
                handle,
            )
        for normaliser in self._normalisers:
            normaliser.save(folder, trial)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        for normaliser in self._normalisers:
            df = normaliser.fit_transform(df, y=y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for normaliser in self._normalisers:
            try:
                df = normaliser.transform(df)
            except ValueError as exc:
                print("Failed to normalise %s", normaliser.name())
                raise exc
        return df
