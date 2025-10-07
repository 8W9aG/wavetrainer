"""A model that wraps catboost."""

# pylint: disable=line-too-long
import json
import os
from typing import Self

import optuna
import pandas as pd
import pytest_is_running
import torch
from catboost import CatBoost, Pool  # type: ignore

from ...model_type import ModelType, determine_model_type
from ..model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model
from .catboost_classifier_wrap import CatBoostClassifierWrapper
from .catboost_regressor_wrap import CatBoostRegressorWrapper

_MODEL_FILENAME = "model.cbm"
_MODEL_PARAMS_FILENAME = "model_params.json"
_MODEL_CATEGORICAL_FEATURES_FILENAME = "catboost_categorical_features.json"
_ITERATIONS_KEY = "iterations"
_LEARNING_RATE_KEY = "learning_rate"
_DEPTH_KEY = "depth"
_L2_LEAF_REG_KEY = "l2_leaf_reg"
_BOOSTING_TYPE_KEY = "boosting_type"
_MODEL_TYPE_KEY = "model_type"
_EARLY_STOPPING_ROUNDS = "early_stopping_rounds"
_BEST_ITERATION_KEY = "best_iteration"
_LOSS_FUNCTION_KEY = "loss_function"
_DEFAULT_LOSS_FUNCTION = "default"
_FOCALLOSS_LOSS_FUNCTION = "focalloss"
_GAMMA_KEY = "focalloss_gamma"
_ALPHA_KEY = "focalloss_alpha"
_BORDER_COUNT_KEY = "border_count"


class CatboostModel(Model):
    """A class that uses Catboost as a model."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments,too-many-instance-attributes

    _catboost: CatBoost | None
    _iterations: None | int
    _learning_rate: None | float
    _depth: None | int
    _l2_leaf_reg: None | float
    _boosting_type: None | str
    _model_type: None | ModelType
    _early_stopping_rounds: None | int
    _best_iteration: None | int
    _categorical_features: dict[str, bool]
    _loss_function: None | str
    _gamma: None | float
    _alpha: None | float
    _border_count: None | int

    @classmethod
    def name(cls) -> str:
        return "catboost"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self._catboost = None
        self._iterations = None
        self._learning_rate = None
        self._depth = None
        self._l2_leaf_reg = None
        self._boosting_type = None
        self._model_type = None
        self._early_stopping_rounds = None
        self._best_iteration = None
        self._categorical_features = {}
        self._loss_function = None
        self._gamma = None
        self._alpha = None
        self._border_count = None

    @property
    def supports_importances(self) -> bool:
        return True

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        def importances_to_dict(importances: pd.DataFrame | None) -> dict[str, float]:
            if importances is None:
                raise ValueError("importances is null")
            feature_ids = importances["Feature Id"].to_list()  # type: ignore
            feature_importances = importances["Importances"].to_list()  # type: ignore
            total = sum(feature_importances)
            return {
                feature_ids[x]: feature_importances[x] / total if total != 0.0 else 0.0
                for x in range(len(feature_ids))
            }

        catboost = self._provide_catboost()
        importances = catboost.get_feature_importance(prettified=True)
        global_importances = importances_to_dict(importances)  # type: ignore
        row_importances = []
        if df is not None:
            for _, row in df.iterrows():
                row_df = row.to_frame().T
                pred_pool = Pool(
                    row_df,
                    cat_features=row_df.select_dtypes(
                        include="category"
                    ).columns.tolist(),
                )
                importances = catboost.get_feature_importance(
                    data=pred_pool, prettified=True
                )
                row_importances.append(importances_to_dict(importances))  # type: ignore
        return global_importances, row_importances

    def provide_estimator(self):
        return self._provide_catboost()

    def create_estimator(self):
        return self._create_catboost()

    def reset(self):
        self._catboost = None
        self._best_iteration = None

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._iterations = trial.suggest_int(_ITERATIONS_KEY, 100, 10000)
        self._learning_rate = trial.suggest_float(_LEARNING_RATE_KEY, 0.001, 0.3)
        self._depth = trial.suggest_int(
            _DEPTH_KEY, 1, 2 if pytest_is_running.is_running() else 6
        )
        self._l2_leaf_reg = trial.suggest_float(_L2_LEAF_REG_KEY, 3.0, 50.0)
        self._boosting_type = trial.suggest_categorical(
            _BOOSTING_TYPE_KEY, ["Ordered", "Plain"]
        )
        self._early_stopping_rounds = trial.suggest_int(_EARLY_STOPPING_ROUNDS, 10, 500)
        self._best_iteration = trial.user_attrs.get(_BEST_ITERATION_KEY)
        # loss_functions = [_DEFAULT_LOSS_FUNCTION]
        # if not torch.cuda.is_available():
        #    loss_functions.append(_FOCALLOSS_LOSS_FUNCTION)
        # loss_function = trial.suggest_categorical(_LOSS_FUNCTION_KEY, loss_functions)
        # self._loss_function = loss_function
        # if loss_function == _FOCALLOSS_LOSS_FUNCTION:
        #    self._gamma = trial.suggest_float(_GAMMA_KEY, 0.5, 5.0)
        #    self._alpha = trial.suggest_float(_ALPHA_KEY, 0.05, 0.95)
        # self._border_count = trial.suggest_int(_BORDER_COUNT_KEY, 2, 512)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._iterations = params[_ITERATIONS_KEY]
            self._learning_rate = params[_LEARNING_RATE_KEY]
            self._depth = params[_DEPTH_KEY]
            self._l2_leaf_reg = params[_L2_LEAF_REG_KEY]
            self._boosting_type = params[_BOOSTING_TYPE_KEY]
            self._model_type = ModelType(params[_MODEL_TYPE_KEY])
            self._early_stopping_rounds = params[_EARLY_STOPPING_ROUNDS]
            self._best_iteration = params.get(_BEST_ITERATION_KEY)
            self._loss_function = params.get(_LOSS_FUNCTION_KEY, _DEFAULT_LOSS_FUNCTION)
            self._gamma = params.get(_GAMMA_KEY)
            self._alpha = params.get(_ALPHA_KEY)
            self._border_count = params.get(_BORDER_COUNT_KEY)
        with open(
            os.path.join(folder, _MODEL_CATEGORICAL_FEATURES_FILENAME), encoding="utf8"
        ) as handle:
            self._categorical_features = json.load(handle)
        catboost = self._provide_catboost()
        catboost.load_model(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _ITERATIONS_KEY: self._iterations,
                    _LEARNING_RATE_KEY: self._learning_rate,
                    _DEPTH_KEY: self._depth,
                    _L2_LEAF_REG_KEY: self._l2_leaf_reg,
                    _BOOSTING_TYPE_KEY: self._boosting_type,
                    _MODEL_TYPE_KEY: str(self._model_type),
                    _EARLY_STOPPING_ROUNDS: self._early_stopping_rounds,
                    _BEST_ITERATION_KEY: self._best_iteration,
                    _LOSS_FUNCTION_KEY: self._loss_function,
                    _GAMMA_KEY: self._gamma,
                    _ALPHA_KEY: self._alpha,
                    _BORDER_COUNT_KEY: self._border_count,
                },
                handle,
            )
        with open(
            os.path.join(folder, _MODEL_CATEGORICAL_FEATURES_FILENAME),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(self._categorical_features, handle)
        catboost = self._provide_catboost()
        catboost.save_model(os.path.join(folder, _MODEL_FILENAME))
        trial.set_user_attr(_BEST_ITERATION_KEY, self._best_iteration)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if y is None:
            raise ValueError("y is null.")
        self._model_type = determine_model_type(y)
        catboost = self._provide_catboost()
        self._categorical_features = {
            x: True for x in df.select_dtypes(include="category").columns.tolist()
        }

        train_pool = Pool(
            df,
            label=y,
            weight=w,
            cat_features=df.select_dtypes(include="category").columns.tolist(),
        )
        eval_pool = (
            Pool(
                eval_x,
                label=eval_y,
                cat_features=eval_x.select_dtypes(include="category").columns.tolist(),
            )
            if eval_x is not None
            else None
        )
        if eval_pool is None:
            print("Eval Pool is null")
        else:
            print("Has Eval Pool")
        catboost.fit(
            train_pool,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=True,
            metric_period=100,
            eval_set=eval_pool,
            use_best_model=True,
        )
        if self._best_iteration is None:
            self._best_iteration = catboost.get_best_iteration()
            print(f"Setting best iteration: {self._best_iteration}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.to_parquet("last_catboost_inference.parquet")
        for categorical_feature_column in self._categorical_features.keys():
            df[categorical_feature_column] = df[categorical_feature_column].astype(
                "category"
            )
        pred_pool = Pool(
            df,
            cat_features=df.select_dtypes(include="category").columns.tolist(),
        )
        catboost = self._provide_catboost()
        pred = catboost.predict(pred_pool)
        df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),
            },
        )
        if self._model_type != ModelType.REGRESSION:
            proba = catboost.predict_proba(pred_pool)  # type: ignore
            for i in range(proba.shape[1]):
                df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return df

    def _provide_catboost(self) -> CatBoost:
        catboost = self._catboost
        if catboost is None:
            catboost = self._create_catboost()
            self._catboost = catboost
        if catboost is None:
            raise ValueError("catboost is null")
        return catboost

    def _create_catboost(self) -> CatBoost:
        best_iteration = self._best_iteration
        iterations = best_iteration if best_iteration is not None else self._iterations
        print(
            f"Creating catboost model with depth {self._depth}, boosting type {self._boosting_type}, best iteration {best_iteration}, iterations {iterations}",
        )
        loss_function = None
        if (
            self._loss_function == _FOCALLOSS_LOSS_FUNCTION
            and self._alpha is not None
            and self._gamma is not None
            and self._model_type != ModelType.REGRESSION
        ):
            loss_function = f"Focal:focal_alpha={self._alpha};focal_gamma={self._gamma}"
        match self._model_type:
            case ModelType.BINARY:
                return CatBoostClassifierWrapper(
                    iterations=iterations,
                    learning_rate=self._learning_rate,
                    depth=self._depth,
                    l2_leaf_reg=self._l2_leaf_reg,
                    boosting_type=self._boosting_type,
                    early_stopping_rounds=self._early_stopping_rounds,
                    metric_period=100,
                    task_type="GPU" if torch.cuda.is_available() else "CPU",
                    devices="0" if torch.cuda.is_available() else None,
                    loss_function=loss_function,
                    gpu_cat_features_storage="CpuPinnedMemory",
                    border_count=self._border_count,
                )
            case ModelType.REGRESSION:
                return CatBoostRegressorWrapper(
                    iterations=iterations,
                    learning_rate=self._learning_rate,
                    depth=self._depth,
                    l2_leaf_reg=self._l2_leaf_reg,
                    boosting_type=self._boosting_type,
                    early_stopping_rounds=self._early_stopping_rounds,
                    metric_period=100,
                    task_type="GPU" if torch.cuda.is_available() else "CPU",
                    devices="0" if torch.cuda.is_available() else None,
                    gpu_cat_features_storage="CpuPinnedMemory",
                    border_count=self._border_count,
                )
            case ModelType.BINNED_BINARY:
                return CatBoostClassifierWrapper(
                    iterations=iterations,
                    learning_rate=self._learning_rate,
                    depth=self._depth,
                    l2_leaf_reg=self._l2_leaf_reg,
                    boosting_type=self._boosting_type,
                    early_stopping_rounds=self._early_stopping_rounds,
                    metric_period=100,
                    task_type="GPU" if torch.cuda.is_available() else "CPU",
                    devices="0" if torch.cuda.is_available() else None,
                    loss_function=loss_function,
                    gpu_cat_features_storage="CpuPinnedMemory",
                    border_count=self._border_count,
                )
            case ModelType.MULTI_CLASSIFICATION:
                return CatBoostClassifierWrapper(
                    iterations=iterations,
                    learning_rate=self._learning_rate,
                    depth=self._depth,
                    l2_leaf_reg=self._l2_leaf_reg,
                    boosting_type=self._boosting_type,
                    early_stopping_rounds=self._early_stopping_rounds,
                    metric_period=100,
                    task_type="GPU" if torch.cuda.is_available() else "CPU",
                    devices="0" if torch.cuda.is_available() else None,
                    loss_function=loss_function,
                    gpu_cat_features_storage="CpuPinnedMemory",
                    border_count=self._border_count,
                )
            case _:
                raise ValueError(f"Unrecognised model type: {self._model_type}")
