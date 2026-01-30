"""The trainer class."""

# pylint: disable=line-too-long,unused-argument
import datetime
import functools
import json
import logging
import os
import pickle
import time
from typing import Self

import numpy as np
import optuna
import pandas as pd
import tqdm
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             precision_score, r2_score, recall_score)

from .calibrator.calibrator_router import CalibratorRouter
from .crps import crps
from .exceptions import WavetrainException
from .fit import Fit
from .model.catboost.catboost_model import CatboostModel
from .model.model import (PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX,
                          QUANTILE_COLUMN_PREFIX, QUANTILES)
from .model.model_router import ModelRouter
from .model_type import ModelType, determine_model_type
from .normaliser.combined_normaliser import CombinedNormaliser
from .reducer.combined_reducer import CombinedReducer
from .selector.selector import Selector
from .threshold_callback import ThresholdCallback
from .weights.combined_weights import CombinedWeights
from .weights.weights import WEIGHTS_COLUMN
from .windower.windower import Windower

_SAMPLER_FILENAME = "sampler.pkl"
_STUDYDB_FILENAME = "study.db"
_PARAMS_FILENAME = "params.json"
_TRIAL_FILENAME = "trial.json"
_TRIALS_KEY = "trials"
_WALKFORWARD_TIMEDELTA_KEY = "walkforward_timedelta"
_DAYS_KEY = "days"
_SECONDS_KEY = "seconds"
_TEST_SIZE_KEY = "test_size"
_VALIDATION_SIZE_KEY = "validation_size"
_IDX_USR_ATTR_KEY = "idx"
_DT_COLUMN_KEY = "dt_column"
_MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY = "max_false_positive_reduction_steps"
_CORRELATION_CHUNK_SIZE_KEY = "correlation_chunk_size"
_BAD_OUTPUT = -1000.0


def _assign_bin(timestamp, bins: list[datetime.datetime]) -> int:
    for i in range(len(bins) - 1):
        if bins[i] <= timestamp < bins[i + 1]:
            return i
    return len(bins) - 2  # Assign to last bin if at the end


def _best_trial(
    study: optuna.Study, dt: datetime.datetime | None = None
) -> optuna.trial.FrozenTrial:
    if dt is None:
        return study.best_trial
    min_trial = None
    for trial in study.trials:
        if trial.value is None:
            continue
        dt_idx = datetime.datetime.fromisoformat(trial.user_attrs[_IDX_USR_ATTR_KEY])
        if dt_idx == dt:
            if min_trial is None:
                min_trial = trial
            elif min_trial.value is None:
                min_trial = trial
            elif min_trial.value > trial.value:
                min_trial = trial
    if min_trial is None:
        min_trial = study.best_trial
    return min_trial


class Trainer(Fit):
    """A class for training and predicting from an array of data."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-locals,too-many-branches,too-many-instance-attributes
    _cached_reducers: dict[str, CombinedReducer]
    _cached_normalizers: dict[str, CombinedNormaliser]
    _cached_models: dict[str, ModelRouter]
    _cached_selectors: dict[str, Selector]
    _cached_calibrators: dict[str, CalibratorRouter]

    def __init__(
        self,
        folder: str,
        walkforward_timedelta: datetime.timedelta | None = None,
        trials: int | None = None,
        test_size: float | datetime.timedelta | None = None,
        validation_size: float | datetime.timedelta | None = None,
        dt_column: str | None = None,
        max_train_timeout: datetime.timedelta | None = None,
        cutoff_dt: datetime.datetime | None = None,
        embedding_cols: list[list[str]] | None = None,
        allowed_models: set[str] | None = None,
        max_false_positive_reduction_steps: int | None = None,
        correlation_chunk_size: int | None = None,
        insert_nulls: bool = False,
        use_power_transformer: bool = False,
    ):
        tqdm.tqdm.pandas()

        self._folder = folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        if trials is None:
            trials = 100
        if test_size is None:
            test_size = 0.15
        if validation_size is None:
            validation_size = 0.15
        if cutoff_dt is None:
            cutoff_dt = datetime.datetime.now()
        if allowed_models is None:
            allowed_models = {CatboostModel.name()}

        params_file = os.path.join(self._folder, _PARAMS_FILENAME)
        if walkforward_timedelta is None:
            if not os.path.exists(params_file):
                raise ValueError(
                    "walkforward_timedelta must be specified on an untrained model."
                )
            with open(params_file, encoding="utf8") as handle:
                params = json.load(handle)
                walkforward_dict = params[_WALKFORWARD_TIMEDELTA_KEY]
                walkforward_timedelta = datetime.timedelta(
                    days=walkforward_dict.get(_DAYS_KEY),
                    seconds=walkforward_dict.get(_SECONDS_KEY),
                )
                trials = params[_TRIALS_KEY]
                test_size_value = params[_TEST_SIZE_KEY]
                if test_size_value is not None:
                    if isinstance(test_size_value, float):
                        test_size = test_size_value
                    else:
                        test_size = datetime.timedelta(
                            days=test_size_value.get(_DAYS_KEY),
                            seconds=test_size_value.get(_SECONDS_KEY),
                        )
                validation_size_value = params[_VALIDATION_SIZE_KEY]
                if validation_size_value is not None:
                    if isinstance(validation_size_value, float):
                        validation_size = validation_size_value
                    else:
                        validation_size = datetime.timedelta(
                            days=validation_size_value.get(_DAYS_KEY),
                            seconds=validation_size_value.get(_SECONDS_KEY),
                        )
                if dt_column is None:
                    dt_column = params[_DT_COLUMN_KEY]
                if max_false_positive_reduction_steps is None:
                    max_false_positive_reduction_steps = params.get(
                        _MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY
                    )
                if correlation_chunk_size is None:
                    correlation_chunk_size = params.get(_CORRELATION_CHUNK_SIZE_KEY)
        else:
            with open(params_file, "w", encoding="utf8") as handle:
                validation_size_value = None
                if validation_size is not None:
                    if isinstance(validation_size, float):
                        validation_size_value = validation_size
                    elif isinstance(validation_size, datetime.timedelta):
                        validation_size_value = {
                            _DAYS_KEY: validation_size.days,
                            _SECONDS_KEY: validation_size.seconds,
                        }
                test_size_value = None
                if test_size is not None:
                    if isinstance(test_size, float):
                        test_size_value = test_size
                    elif isinstance(test_size, datetime.timedelta):
                        test_size_value = {
                            _DAYS_KEY: test_size.days,
                            _SECONDS_KEY: test_size.seconds,
                        }
                json.dump(
                    {
                        _TRIALS_KEY: trials,
                        _WALKFORWARD_TIMEDELTA_KEY: {
                            _DAYS_KEY: walkforward_timedelta.days,
                            _SECONDS_KEY: walkforward_timedelta.seconds,
                        },
                        _TEST_SIZE_KEY: test_size_value,
                        _VALIDATION_SIZE_KEY: validation_size_value,
                        _DT_COLUMN_KEY: dt_column,
                        _MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY: max_false_positive_reduction_steps,
                        _CORRELATION_CHUNK_SIZE_KEY: correlation_chunk_size,
                    },
                    handle,
                )
        self._walkforward_timedelta = walkforward_timedelta
        self._trials = trials
        self._test_size = test_size
        self._validation_size = validation_size
        self._dt_column = dt_column
        self._max_train_timeout = max_train_timeout
        self._cutoff_dt = cutoff_dt
        self.embedding_cols = embedding_cols
        self._allowed_models = allowed_models
        self._max_false_positive_reduction_steps = max_false_positive_reduction_steps
        self._correlation_chunk_size = correlation_chunk_size
        self._insert_nulls = insert_nulls
        self._use_power_transformer = use_power_transformer
        self._cached_reducers = {}
        self._cached_normalizers = {}
        self._cached_models = {}
        self._cached_selectors = {}
        self._cached_calibrators = {}

    def _provide_study(self, column: str) -> optuna.Study:
        storage_name = f"sqlite:///{self._folder}/{column}/{_STUDYDB_FILENAME}"
        sampler_file = os.path.join(self._folder, column, _SAMPLER_FILENAME)
        restored_sampler = None
        if os.path.exists(sampler_file):
            with open(sampler_file, "rb") as handle:
                restored_sampler = pickle.load(handle)
        return optuna.create_study(
            study_name="wavetrain",
            storage=storage_name,
            load_if_exists=True,
            sampler=restored_sampler,
            directions=[
                # optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MINIMIZE,
            ],
        )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        """Perform a train on the data to fit to the targets."""
        if y is None:
            return self

        df = df.reindex(sorted(df.columns), axis=1)

        dt_index = (
            df.index
            if self._dt_column is None
            else pd.DatetimeIndex(pd.to_datetime(df[self._dt_column]))
        )
        df = df[dt_index < self._cutoff_dt]  # type: ignore
        y = y.iloc[: len(df)]
        dt_index = dt_index[: len(df)]

        def _fit_column(y_series: pd.Series):
            column_dir = os.path.join(self._folder, str(y_series.name))
            if not os.path.exists(column_dir):
                os.mkdir(column_dir)

            study = self._provide_study(str(y_series.name))
            did_fit = False

            def _fit(
                trial: optuna.Trial | optuna.trial.FrozenTrial,
                x: pd.DataFrame,
                y_series: pd.Series,
                save: bool,
                split_idx: datetime.datetime,
                no_evaluation: bool,
                # ) -> tuple[float, float]:
            ) -> float:
                nonlocal did_fit
                print(f"Beginning trial for: {split_idx.isoformat()}")
                trial.set_user_attr(_IDX_USR_ATTR_KEY, split_idx.isoformat())
                folder = os.path.join(
                    self._folder, str(y_series.name), split_idx.isoformat()
                )
                new_folder = not os.path.exists(folder)
                os.makedirs(folder, exist_ok=True)
                trial_file = os.path.join(folder, _TRIAL_FILENAME)
                if os.path.exists(trial_file):
                    with open(trial_file, encoding="utf8") as handle:
                        trial_info = json.load(handle)
                        if trial_info["number"] == trial.number:
                            logging.info(
                                "Found trial %d previously executed, skipping...",
                                trial.number,
                            )
                            # return tuple(trial_info["output"])
                            return tuple(trial_info["output"])[0]
                        print("Retraining for different trial number.")
                did_fit = True

                train_dt_index = dt_index[: len(x)]
                x_train = x[train_dt_index < split_idx]  # type: ignore
                x_test = x[train_dt_index >= split_idx]  # type: ignore
                y_train = y_series[train_dt_index < split_idx]  # type: ignore
                y_test = y_series[train_dt_index >= split_idx]  # type: ignore

                try:
                    # Window the data
                    start_windower = time.time()
                    windower = Windower(self._dt_column)
                    windower.set_options(trial, x)
                    x_train = windower.fit_transform(x_train)
                    y_train = y_train[-len(x_train) :]
                    if len(y_train.unique()) <= 1:
                        if new_folder:
                            os.removedirs(folder)
                        logging.warning("Y train only contains 1 unique datapoint.")
                        # return _BAD_OUTPUT, -_BAD_OUTPUT
                        # return _BAD_OUTPUT, -_BAD_OUTPUT
                        return -_BAD_OUTPUT
                    print(f"Windowing took {time.time() - start_windower}")

                    # Perform common reductions
                    start_reducer = time.time()
                    reducer = CombinedReducer(
                        self.embedding_cols,
                        self._correlation_chunk_size,
                        self._insert_nulls,
                    )
                    reducer.set_options(trial, x)
                    x_train = reducer.fit_transform(x_train, y=y_train)
                    x_test = reducer.transform(x_test)
                    print(f"Reducing took {time.time() - start_reducer}")

                    # Perform common normalisations
                    start_normaliser = time.time()
                    normaliser = CombinedNormaliser(
                        use_power_transformer=self._use_power_transformer
                    )
                    normaliser.set_options(trial, x)
                    x_train = normaliser.fit_transform(x_train, y=y_train)
                    x_test = normaliser.transform(x_test)
                    print(f"Normalising took {time.time() - start_normaliser}")

                    # Calculate the row weights
                    start_row_weights = time.time()
                    weights = CombinedWeights()
                    weights.set_options(trial, x)
                    w = weights.fit(x_train, y=y_train).transform(y_train.to_frame())[
                        WEIGHTS_COLUMN
                    ]
                    print(f"Row weights took {time.time() - start_row_weights}")

                    # Create model
                    model = ModelRouter(
                        self._allowed_models, self._max_false_positive_reduction_steps
                    )
                    model.set_options(trial, x)

                    # Train
                    start_selector = time.time()
                    selector = Selector(model)
                    selector.set_options(trial, x)
                    selector.fit(
                        x_train,
                        y=y_train,
                        w=w,
                        eval_x=x_test if not no_evaluation else None,
                        eval_y=y_test if not no_evaluation else None,
                    )
                    x_train = selector.transform(x_train)
                    x_test = selector.transform(x_test)
                    print(f"Selection took {time.time() - start_selector}")
                    start_train = time.time()
                    model.fit(
                        x_train,
                        y=y_train,
                        w=w,
                        eval_x=x_test if not no_evaluation else None,
                        eval_y=y_test if not no_evaluation else None,
                    )
                    y_pred = model.transform(x_test)
                    print(f"Training took {time.time() - start_train}")

                    # Calibrate
                    start_calibrate = time.time()
                    calibrator = CalibratorRouter(model)
                    calibrator.set_options(trial, x)
                    calibrator.fit(
                        y_pred if calibrator.predictions_as_x(y_test) else x_test,
                        y=y_test,
                    )
                    print(f"Calibrating took {time.time() - start_calibrate}")

                    # Output
                    y_pred = calibrator.transform(
                        y_pred if calibrator.predictions_as_x(y_test) else x_test
                    )
                    output = 0.0
                    loss = 0.0
                    pvalue = 0.0
                    log_loss_value = 0.0
                    model_type = determine_model_type(y_series)
                    if model_type == ModelType.REGRESSION:
                        output = float(r2_score(y_test, y_pred[[PREDICTION_COLUMN]]))
                        print(f"R2: {output}")
                    elif model_type == ModelType.QUANTILE_REGRESSION:
                        y_q = y_pred[
                            sorted(
                                [
                                    x
                                    for x in y_pred.columns.values.tolist()
                                    if QUANTILE_COLUMN_PREFIX in x
                                ]
                            )
                        ]
                        output = float(
                            crps(y_test.to_numpy(), y_q.to_numpy(), np.array(QUANTILES))
                        )
                        loss = output
                        print(f"CRPS: {output}")
                    else:
                        output = float(f1_score(y_test, y_pred[[PREDICTION_COLUMN]]))
                        print(f"F1: {output}")
                        prob_col = PROBABILITY_COLUMN_PREFIX + str(1)
                        if prob_col in y_pred.columns.values.tolist():
                            y_prob = y_pred[[prob_col]]
                            loss = float(brier_score_loss(y_test, y_prob))
                            log_loss_value = 1.0
                            try:
                                log_loss_value = float(
                                    log_loss(y_test.astype(float), y_prob)
                                )
                            except ValueError as exc:
                                print(str(exc))
                            pvalue = calibrator.p_value
                            print(f"Brier: {loss}")
                            print(f"Log Loss: {log_loss_value}")
                            print(f"P-Value: {pvalue}")
                            # print(
                            #    f"Stratified Brier: {stratified_brier_score_loss(y_test, y_prob)}"
                            # )
                        print(
                            f"Accuracy: {float(accuracy_score(y_test, y_pred[[PREDICTION_COLUMN]]))}"
                        )
                        print(
                            f"Precision: {float(precision_score(y_test, y_pred[[PREDICTION_COLUMN]], zero_division=0.0))}"  # type: ignore
                        )
                        print(
                            f"Recall: {float(recall_score(y_test, y_pred[[PREDICTION_COLUMN]]))}"
                        )

                    if save:
                        windower.save(folder, trial)
                        normaliser.save(folder, trial)
                        reducer.save(folder, trial)
                        weights.save(folder, trial)
                        model.save(folder, trial)
                        selector.save(folder, trial)
                        calibrator.save(folder, trial)
                        with open(trial_file, "w", encoding="utf8") as handle:
                            json.dump(
                                {
                                    "number": trial.number,
                                    "output": [output, loss],
                                },
                                handle,
                            )

                    # return output, loss
                    # return pvalue, loss
                    return loss
                except WavetrainException as exc:
                    print(str(exc))
                    logging.warning(str(exc))
                    if new_folder:
                        os.removedirs(folder)
                    # return _BAD_OUTPUT, -_BAD_OUTPUT
                    # return _BAD_OUTPUT, -_BAD_OUTPUT
                    return -_BAD_OUTPUT

            start_validation_index = (
                dt_index.to_list()[-int(len(dt_index) * self._validation_size) - 1]
                if isinstance(self._validation_size, float)
                else dt_index[
                    dt_index >= (dt_index.to_list()[-1] - self._validation_size)  # type: ignore
                ].to_list()[0]
            )
            test_df = df[dt_index < start_validation_index]
            test_dt_index = (
                test_df.index if self._dt_column is None else test_df[self._dt_column]
            )
            start_test_index = (
                test_dt_index.to_list()[-int(len(test_dt_index) * self._test_size)]
                if isinstance(self._test_size, float)
                else test_dt_index[
                    test_dt_index >= (start_validation_index - self._test_size)  # type: ignore
                ].to_list()[0]
            )

            # def test_objective(trial: optuna.Trial) -> tuple[float, float]:
            def test_objective(trial: optuna.Trial) -> float:
                return _fit(
                    trial,
                    test_df,
                    y_series[dt_index < start_validation_index],
                    False,
                    start_test_index.to_pydatetime(),
                    False,
                )

            initial_trials = 0
            if self._trials is not None:
                initial_trials = max(self._trials - len(study.trials), 0)
            if initial_trials > 0:
                study.optimize(
                    test_objective,
                    n_trials=initial_trials,
                    show_progress_bar=True,
                    timeout=None
                    if self._max_train_timeout is None
                    else self._max_train_timeout.total_seconds(),
                )
            while (
                _best_trial(study).values is None
                or _best_trial(study).values == (_BAD_OUTPUT, -_BAD_OUTPUT)
            ) and len(study.trials) < 1000:
                logging.info("Performing extra train")
                study.optimize(
                    test_objective,
                    n_trials=1,
                    show_progress_bar=True,
                    timeout=None
                    if self._max_train_timeout is None
                    else self._max_train_timeout.total_seconds(),
                )

            train_len = len(df[dt_index < start_test_index])
            test_len = len(
                dt_index[
                    (dt_index >= start_test_index)
                    & (dt_index <= start_validation_index)
                ]
            )

            last_processed_dt = None
            # If we are resuming, find the last processed datetime
            for trial in study.trials:
                dt_idx = datetime.datetime.fromisoformat(
                    trial.user_attrs[_IDX_USR_ATTR_KEY]
                )
                if last_processed_dt is None:
                    last_processed_dt = dt_idx
                else:
                    last_processed_dt = max(last_processed_dt, dt_idx)

            for count, test_idx in tqdm.tqdm(
                enumerate(dt_index[dt_index >= start_test_index])
            ):
                test_dt = test_idx.to_pydatetime()
                test_df = df.iloc[: train_len + count + test_len]
                test_series = y_series.iloc[: train_len + count + test_len]
                found = False
                for trial in study.trials:
                    dt_idx = datetime.datetime.fromisoformat(
                        trial.user_attrs[_IDX_USR_ATTR_KEY]
                    )
                    if dt_idx == test_dt:
                        found = True
                        break
                if found:
                    last_processed_dt = test_dt
                    _fit(
                        _best_trial(study, dt=test_dt),
                        test_df.copy(),
                        test_series,
                        True,
                        test_idx,
                        False,
                    )
                    continue
                if (
                    last_processed_dt is not None
                    and test_dt < last_processed_dt + self._walkforward_timedelta
                ):
                    continue

                if len(test_df) <= 3:
                    continue

                if test_idx < start_validation_index:

                    def validate_objective(
                        trial: optuna.Trial,
                        idx: datetime.datetime,
                        series: pd.Series,
                        # ) -> tuple[float, float]:
                    ) -> float:
                        return _fit(trial, test_df.copy(), series, False, idx, False)

                    study.optimize(
                        functools.partial(
                            validate_objective, idx=test_idx, series=test_series
                        ),
                        n_trials=min(
                            self._trials if self._trials is not None else 5, 5
                        ),
                        timeout=None
                        if self._max_train_timeout is None
                        else self._max_train_timeout.total_seconds(),
                        callbacks=[
                            ThresholdCallback(
                                0.0
                                if study.best_trial.value is None
                                else study.best_trial.value
                            )
                        ],
                    )
                else:
                    break

                value = _fit(
                    _best_trial(study, dt=test_dt),
                    test_df.copy(),
                    test_series,
                    True,
                    test_idx,
                    False,
                )
                print(f"Saved model with value: {value}")
                last_processed_dt = test_idx

            if did_fit:
                target_names = ["F1", "Brier"]
                # fig = optuna.visualization.plot_pareto_front(
                #    study, target_names=target_names
                # )
                # fig.write_image(
                #    os.path.join(column_dir, "pareto_frontier.png"),
                #    format="png",
                #    width=800,
                #    height=600,
                # )
                for target_name in target_names:
                    try:
                        fig = optuna.visualization.plot_param_importances(
                            study, target=lambda t: t.values[0], target_name=target_name
                        )
                        fig.write_image(
                            os.path.join(column_dir, f"{target_name}_frontier.png"),
                            format="png",
                            width=800,
                            height=600,
                        )
                    except RuntimeError as exc:
                        print(str(exc))

        if isinstance(y, pd.Series):
            _fit_column(y)
        else:
            for col in y.columns:
                print(f"Training column: {col}")
                y_train_series = y[col]
                y_train_series.attrs = y.attrs
                _fit_column(y_train_series)

        return self

    def transform(
        self, df: pd.DataFrame, optimistic: bool = False, ignore_no_dates: bool = False
    ) -> pd.DataFrame:
        """Predict the expected values of the data."""
        # tqdm.tqdm.pandas(desc="Inferring...")
        input_df = df.copy()
        df = df.reindex(sorted(df.columns), axis=1)
        feature_columns = df.columns.values
        dt_index = (
            df.index
            if self._dt_column is None
            else pd.DatetimeIndex(pd.to_datetime(df[self._dt_column]))
        )

        for column in os.listdir(self._folder):
            column_path = os.path.join(self._folder, column)
            if not os.path.isdir(column_path):
                continue
            dates = []
            for date_str in os.listdir(column_path):
                date_path = os.path.join(column_path, date_str)
                if not os.path.isdir(date_path):
                    continue
                if not os.listdir(date_path):
                    continue
                dates.append(datetime.datetime.fromisoformat(date_str))
            if not dates:
                if ignore_no_dates:
                    continue
                raise ValueError(f"no dates found for {column}.")
            dates = sorted(dates)
            bins: list[datetime.datetime] = sorted(
                [dt_index.min().to_pydatetime()]
                + dates
                + [(dt_index.max() + pd.Timedelta(days=1)).to_pydatetime()]
            )

            def perform_predictions(
                group: pd.DataFrame,
                column_path: str,
                column: str,
                dates: list[datetime.datetime],
            ) -> pd.DataFrame:
                group_dt_index = (
                    group.index
                    if self._dt_column is None
                    else pd.DatetimeIndex(pd.to_datetime(group[self._dt_column]))
                )

                max_date = group_dt_index.max()
                if not optimistic:
                    if isinstance(self._validation_size, float):
                        max_date = group_dt_index[
                            -int(len(group_dt_index) * self._validation_size)
                        ]
                    elif isinstance(self._validation_size, int):
                        max_date = group_dt_index[-self._validation_size]
                    else:
                        max_date = max_date - self._validation_size

                filtered_dates = [x for x in dates if x < max_date]
                if not filtered_dates:
                    filtered_dates = [dates[-1]]
                date_str = filtered_dates[-1].isoformat()
                folder = os.path.join(column_path, date_str)
                # print(f"Loading {folder}")

                try:
                    reducer = self._provide_reducer(folder)
                    normaliser = self._provide_normalizer(folder)
                    model = self._provide_model(folder)
                    selector = self._provide_selector(folder)
                    calibrator = self._provide_calibrator(folder)

                    x_pred = reducer.transform(group[feature_columns])
                    x_pred = normaliser.transform(x_pred)
                    x_pred = selector.transform(x_pred)
                    y_pred = model.transform(x_pred)
                    y_pred = calibrator.transform(
                        y_pred if calibrator.predictions_as_x(None) else x_pred
                    )
                    for new_column in y_pred.columns.values:
                        group["_".join([column, new_column])] = y_pred[new_column]
                except FileNotFoundError as exc:
                    print(f"Model {folder} failed:")
                    print(str(exc))
                except AttributeError as exc:
                    print(f"Model {folder} failed:")
                    print(str(exc))
                    raise exc

                return group

            old_index = dt_index.copy()
            df_group = df.groupby(
                dt_index.map(functools.partial(_assign_bin, bins=bins))
            )
            if len(df_group) == 1:
                df = df_group.apply(  # type: ignore
                    functools.partial(
                        perform_predictions,
                        column_path=column_path,
                        column=column,
                        dates=dates,
                    )
                )
            else:
                df = df_group.progress_apply(  # type: ignore
                    functools.partial(
                        perform_predictions,
                        column_path=column_path,
                        column=column,
                        dates=dates,
                    )
                )
            if self._dt_column is None:
                df = df.set_index(old_index)

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        for col in input_df.columns.values:
            if col in df.columns.values:
                continue
            df[col] = input_df[col]

        return df

    def feature_importances(
        self, df: pd.DataFrame, latest_date_only: bool = False
    ) -> dict[str, tuple[dict[str, float], list[dict[str, float]]]]:
        """Find the feature importances for the rolling models."""
        feature_importances = {}

        for column in os.listdir(self._folder):
            column_path = os.path.join(self._folder, column)
            if not os.path.isdir(column_path):
                continue
            date_paths = sorted(
                [
                    os.path.join(column_path, x)
                    for x in list(os.listdir(column_path))
                    if os.path.isdir(os.path.join(column_path, x))
                    and os.listdir(os.path.join(column_path, x))
                ]
            )
            if latest_date_only:
                date_paths = date_paths[-1:]
            for date_path in tqdm.tqdm(date_paths):
                try:
                    reducer = CombinedReducer(
                        self.embedding_cols,
                        self._correlation_chunk_size,
                        self._insert_nulls,
                    )
                    reducer.load(date_path)

                    model = ModelRouter(None, None)
                    model.load(date_path)

                    selector = Selector(model)
                    selector.load(date_path)

                    x_pred = reducer.transform(df)
                    x_pred = selector.transform(x_pred)

                    feature_importances[date_path.split("/")[-1]] = (
                        model.feature_importances(x_pred)
                    )
                except FileNotFoundError as exc:
                    logging.warning(str(exc))

        return feature_importances

    def _provide_reducer(self, folder: str) -> CombinedReducer:
        if folder not in self._cached_reducers:
            reducer = CombinedReducer(
                self.embedding_cols,
                self._correlation_chunk_size,
                self._insert_nulls,
            )
            reducer.load(folder)
            self._cached_reducers[folder] = reducer
        return self._cached_reducers[folder]

    def _provide_normalizer(self, folder: str) -> CombinedNormaliser:
        if folder not in self._cached_normalizers:
            normalizer = CombinedNormaliser(
                use_power_transformer=self._use_power_transformer
            )
            normalizer.load(folder)
            self._cached_normalizers[folder] = normalizer
        return self._cached_normalizers[folder]

    def _provide_model(self, folder: str) -> ModelRouter:
        if folder not in self._cached_models:
            model = ModelRouter(None, None)
            model.load(folder)
            self._cached_models[folder] = model
        return self._cached_models[folder]

    def _provide_selector(self, folder: str) -> Selector:
        if folder not in self._cached_selectors:
            selector = Selector(self._provide_model(folder))
            selector.load(folder)
            self._cached_selectors[folder] = selector
        return self._cached_selectors[folder]

    def _provide_calibrator(self, folder: str) -> CalibratorRouter:
        if folder not in self._cached_calibrators:
            calibrator = CalibratorRouter(self._provide_model(folder))
            calibrator.load(folder)
            self._cached_calibrators[folder] = calibrator
        return self._cached_calibrators[folder]
