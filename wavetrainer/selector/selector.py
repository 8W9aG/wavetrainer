"""The selector class."""

# pylint: disable=too-many-locals,line-too-long
import functools
import json
import logging
import os
from typing import Self

import optuna
import pandas as pd

from ..fit import Fit
from ..model.model import Model
from ..params import Params

_SELECTOR_FILE = "selector.json"
_FEATURE_IMPORTANCES_FILE = "feature_importances.json"


class Selector(Params, Fit):
    """The selector class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments,consider-using-enumerate

    _selector: list[str] | None
    _feature_importances: dict[str, float]

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._feature_ratio = 0.0
        self._steps = 0
        self._selector = None
        self._feature_importances = {}

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._feature_ratio = trial.suggest_float("feature_ratio", 0.0, 1.0)
        self._steps = trial.suggest_int("steps", 1, 10)

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _SELECTOR_FILE), encoding="utf8") as handle:
            self._selector = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(os.path.join(folder, _SELECTOR_FILE), "w", encoding="utf8") as handle:
            json.dump(self._selector, handle)
        with open(
            os.path.join(folder, _FEATURE_IMPORTANCES_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(self._feature_importances, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if not self._model.supports_importances:
            return self
        if not isinstance(y, pd.Series):
            raise ValueError("y is not a series.")
        total_columns = len(df.columns)
        if total_columns <= 1:
            return self
        print(
            f"Performing feature selection with {self._steps} steps and a total ratio of {self._feature_ratio}"
        )
        current_features = df.columns.values.tolist()

        def set_current_features(required_features: int):
            nonlocal current_features
            feature_importances, _ = self._model.feature_importances(None)
            if not feature_importances:
                return
            self._feature_importances = feature_importances
            current_features = sorted(
                [
                    x
                    for x in current_features
                    if x in feature_importances and feature_importances[x] > 0.0
                ],
                key=functools.partial(lambda x, f: f[x], f=feature_importances),
                reverse=True,
            )
            if not current_features:
                current_features = [list(feature_importances.keys())[0]]
            current_features = current_features[:required_features]
            print(
                f"Current Features:\n{pd.Series(data=[feature_importances[x] for x in current_features], index=current_features)}\n"
            )

        n_features = len(current_features)
        for i in range(self._steps):
            ratio_diff = 1.0 - self._feature_ratio
            ratio_step = ratio_diff / float(self._steps)
            current_ratio = 1.0 - (ratio_step * i)
            n_features = max(1, int(total_columns * current_ratio))
            print(
                f"Recursive Feature Elimination Step {i}, current features: {len(current_features)} required features: {n_features}"
            )
            if n_features >= len(current_features):
                continue

            self._model.reset()
            self._model.fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
            set_current_features(n_features)
            print(f"Reduced features to {len(current_features)}")
            df = df[current_features]
            if eval_x is not None:
                eval_x = eval_x[current_features]
        print(f"Final feature count: {len(current_features)}")

        self._selector = current_features

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) <= 1:
            return df
        selector = self._selector
        if selector is None:
            logging.warning("selector is null")
            return df
        return df[self._selector]  # type: ignore
