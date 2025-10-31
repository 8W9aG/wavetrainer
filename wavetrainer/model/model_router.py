"""A model class that routes to other models."""

import functools
import json
import os
from typing import Self

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score  # type: ignore

from ..model_type import ModelType, determine_model_type
from .catboost.catboost_model import CatboostModel
from .lightgbm.lightgbm_model import LightGBMModel
from .model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model
from .tabpfn.tabpfn_model import TabPFNModel
from .xgboost.xgboost_model import XGBoostModel

_MODEL_ROUTER_FILE = "model_router.json"
_MODEL_KEY = "model"
_FALSE_POSITIVE_REDUCTION_STEPS_KEY = "false_positive_reduction_steps"
_MODELS = {
    CatboostModel.name(): CatboostModel,
    TabPFNModel.name(): TabPFNModel,
    XGBoostModel.name(): XGBoostModel,
    LightGBMModel.name(): LightGBMModel,
}


class ModelRouter(Model):
    """A router that routes to a different weights class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    _model: Model | None
    _false_positive_reduction_steps: int | None

    def __init__(
        self,
        allowed_models: set[str] | None,
        max_false_positive_reduction_steps: int | None,
    ) -> None:
        super().__init__()
        self._model = None
        self._false_positive_reduction_steps = None
        self._allowed_models = (
            allowed_models if allowed_models is not None else set(_MODELS.keys())
        )
        self._max_false_positive_reduction_steps = max_false_positive_reduction_steps

    @classmethod
    def name(cls) -> str:
        return "router"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

    @property
    def supports_importances(self) -> bool:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.supports_importances

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.feature_importances(df)

    def provide_estimator(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.provide_estimator()

    def create_estimator(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.create_estimator()

    def reset(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        model.reset()

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.convert_df(df)

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._false_positive_reduction_steps = trial.suggest_int(
            _FALSE_POSITIVE_REDUCTION_STEPS_KEY,
            0,
            5
            if self._max_false_positive_reduction_steps is None
            else self._max_false_positive_reduction_steps,
        )
        model_name = trial.suggest_categorical(
            "model",
            [
                k
                for k, v in _MODELS.items()
                if v.supports_x(df) and k in self._allowed_models
            ],
        )
        print(f"Using {model_name} model")
        model = _MODELS[model_name]()
        model.set_options(trial, df)
        self._model = model

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _MODEL_ROUTER_FILE), encoding="utf8") as handle:
            params = json.load(handle)
            model = _MODELS[params[_MODEL_KEY]]()
        model.load(folder)
        self._model = model

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        model.save(folder, trial)
        with open(
            os.path.join(folder, _MODEL_ROUTER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _MODEL_KEY: model.name(),
                },
                handle,
            )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        false_positive_reduction_steps = self._false_positive_reduction_steps
        if false_positive_reduction_steps is None:
            false_positive_reduction_steps = 0
        for i in range(max(false_positive_reduction_steps, 1)):
            print(f"False Positive Reduction Step: {i + 1}")
            pred = model.fit_transform(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
            if (
                w is None
                or y is None
                or determine_model_type(y) == ModelType.REGRESSION
            ):
                break
            print(f"Accuracy: {accuracy_score(y, pred[PREDICTION_COLUMN])}")
            if false_positive_reduction_steps == 0:
                break
            pred["__wavetrain_correct"] = pred[PREDICTION_COLUMN] != y
            pred["__wavetrain_error_weight"] = pred["__wavetrain_correct"].astype(float)
            prob_columns = sorted(
                [
                    x
                    for x in pred.columns.values.tolist()
                    if x.startswith(PROBABILITY_COLUMN_PREFIX)
                ]
            )
            if prob_columns:

                def determine_error_weight(
                    row: pd.Series, prob_columns: list[str]
                ) -> float:
                    nonlocal y
                    if not row["__wavetrain_correct"]:
                        return abs(row[prob_columns[1 - int(y.loc[row.name])]])  # type: ignore
                    return 0.0

                pred["__wavetrain_error_weight"] = pred.apply(
                    functools.partial(
                        determine_error_weight,
                        prob_columns=prob_columns,
                    ),
                    axis=1,
                )
            w += pred["__wavetrain_error_weight"].fillna(0.0).clip(lower=0.000001)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.transform(df)
