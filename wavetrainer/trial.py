"""Fit one Optuna trial of one walk-forward fold."""

# pylint: disable=line-too-long
import datetime
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import filelock
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             precision_score, r2_score, recall_score)

from .calibrator.calibrator_router import CalibratorRouter
from .crps import crps
from .exceptions import WavetrainException
from .model.model import (PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX,
                          QUANTILE_COLUMN_PREFIX, QUANTILES)
from .model.model_router import ModelRouter
from .model_type import ModelType, determine_model_type
from .normaliser.combined_normaliser import CombinedNormaliser
from .reducer.combined_reducer import CombinedReducer
from .selector.selector import Selector
from .shared_frame import SharedFrame
from .weights.combined_weights import CombinedWeights
from .weights.weights import WEIGHTS_COLUMN
from .windower.windower import Windower

if TYPE_CHECKING:
    from .trainer import Trainer

# Written into a fold folder with the number of the trial saved there.
_TRIAL_FILENAME = "trial.json"
# The trial user attribute holding the start of the trial's walk-forward fold.
IDX_USR_ATTR_KEY = "idx"
# Negated, the loss given to a trial whose fold cannot be fitted.
BAD_OUTPUT = -1000.0

# Trials for the same walk-forward fold run concurrently (in threads of one
# process, or in the worker processes of a TrialWorkerPool) and share the same
# fold folder path. Creating (os.makedirs) and tearing down (os.removedirs)
# that folder must be serialised per path, otherwise one trial can remove the
# directory out from under another that just created or is still using it,
# which os.makedirs(exist_ok=True) does not protect against on its own. A file
# lock serialises across processes; the thread lock in front of it keeps
# threads of one process from busy-polling the file lock. Kept at module level
# (rather than on Trainer) so Trainer instances stay picklable.
_FOLD_LOCKS: dict[str, threading.Lock] = {}
_FOLD_LOCKS_GUARD = threading.Lock()


class FoldLock:
    """A reusable lock on one fold folder, across threads and processes.

    Holds a thread lock shared by every FoldLock on the same folder in this
    process, then a file lock on "<folder>.lock" shared with other processes.
    """

    def __init__(self, folder: str):
        with _FOLD_LOCKS_GUARD:
            lock = _FOLD_LOCKS.get(folder)
            if lock is None:
                lock = threading.Lock()
                _FOLD_LOCKS[folder] = lock
        self._folder = folder
        self._thread_lock = lock
        self._file_lock = filelock.FileLock(folder + ".lock")

    def __enter__(self) -> None:
        # The lock file sits beside the fold folder, so its parent must exist.
        os.makedirs(os.path.dirname(self._folder), exist_ok=True)
        self._thread_lock.acquire()  # pylint: disable=consider-using-with
        try:
            self._file_lock.acquire()
        except BaseException:
            # Don't leave the thread lock held if the file lock failed.
            self._thread_lock.release()
            raise

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._file_lock.release()
        finally:
            self._thread_lock.release()


def fold_lock(folder: str) -> FoldLock:
    """Lock a fold folder while creating or removing it."""
    return FoldLock(folder)


def best_trial(
    study: optuna.Study, dt: datetime.datetime | None = None
) -> optuna.trial.FrozenTrial:
    """Find the best trial overall, or for the fold starting at dt."""
    if dt is None:
        return study.best_trial
    min_trial = None
    for trial in study.trials:
        if trial.value is None:
            continue
        dt_idx = datetime.datetime.fromisoformat(trial.user_attrs[IDX_USR_ATTR_KEY])
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


def fit_trial(
    trainer: "Trainer",
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    x: pd.DataFrame,
    y_series: pd.Series,
    dt_index: pd.Index,
    save: bool,
    split_idx: datetime.datetime,
    no_evaluation: bool,
    # ) -> tuple[float, float]:
) -> float:
    """Fit a trial's pipeline on one fold and return its loss, saving it if asked."""
    # pylint: disable=protected-access,too-many-statements,too-many-locals,too-many-branches,too-many-arguments,too-many-positional-arguments
    print(f"Beginning trial for: {split_idx.isoformat()}")
    trial.set_user_attr(IDX_USR_ATTR_KEY, split_idx.isoformat())
    folder = os.path.join(trainer._folder, str(y_series.name), split_idx.isoformat())
    lock = fold_lock(folder)
    with lock:
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

    train_dt_index = dt_index[: len(x)]
    x_train: pd.DataFrame = x[train_dt_index < split_idx]  # type: ignore
    x_test: pd.DataFrame = x[train_dt_index >= split_idx]  # type: ignore
    y_train = y_series[train_dt_index < split_idx]  # type: ignore
    y_test = y_series[train_dt_index >= split_idx]  # type: ignore

    try:
        # Window the data
        start_windower = time.time()
        windower = Windower(trainer._dt_column)
        windower.set_options(trial, x)
        x_train = windower.fit_transform(x_train)
        y_train = y_train[-len(x_train) :]
        if len(y_train.unique()) <= 1:
            if new_folder:
                with lock:
                    try:
                        os.removedirs(folder)
                    except OSError:
                        # Another trial may have already removed
                        # it, or written into it in the meantime.
                        pass
            logging.warning("Y train only contains 1 unique datapoint.")
            # return BAD_OUTPUT, -BAD_OUTPUT
            # return BAD_OUTPUT, -BAD_OUTPUT
            return -BAD_OUTPUT
        print(f"Windowing took {time.time() - start_windower}")

        # Perform common reductions
        start_reducer = time.time()
        reducer = CombinedReducer(
            trainer.embedding_cols,
            trainer._correlation_chunk_size,
            trainer._insert_nulls,
            include_correlation_reducer=trainer._use_correlation_reducer,
        )
        reducer.set_options(trial, x)
        x_train = reducer.fit_transform(x_train, y=y_train)
        x_test = reducer.transform(x_test)
        print(f"Reducing took {time.time() - start_reducer}")

        # Perform common normalisations
        start_normaliser = time.time()
        normaliser = CombinedNormaliser(
            use_power_transformer=trainer._use_power_transformer
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
            trainer._allowed_models, trainer._max_false_positive_reduction_steps
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
        stats = {}
        if model_type == ModelType.REGRESSION:
            output = float(r2_score(y_test, y_pred[[PREDICTION_COLUMN]]))
            # The study minimises the loss, and higher R2 is better. Within a
            # fold this ranks trials the same as MSE would, but unlike MSE it
            # doesn't depend on the target's scale, so trials from different
            # folds can be compared (study.best_trial does that).
            loss = -output
            print(f"R2: {output}")
            stats["R2"] = output
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
            output = float(crps(y_test.to_numpy(), y_q.to_numpy(), np.array(QUANTILES)))
            loss = output
            print(f"CRPS: {output}")
            stats["CRPS"] = output
        else:
            output = float(f1_score(y_test, y_pred[[PREDICTION_COLUMN]]))
            print(f"F1: {output}")
            prob_col = PROBABILITY_COLUMN_PREFIX + str(1)
            if prob_col in y_pred.columns.values.tolist():
                y_prob = y_pred[[prob_col]]
                loss = float(brier_score_loss(y_test, y_prob))
                log_loss_value = 1.0
                try:
                    log_loss_value = float(log_loss(y_test.astype(float), y_prob))
                except ValueError as exc:
                    print(str(exc))
                pvalue = calibrator.p_value
                print(f"Brier: {loss}")
                print(f"Log Loss: {log_loss_value}")
                print(f"P-Value: {pvalue}")
                stats["brier"] = loss
                stats["log_loss"] = log_loss_value
                stats["p_value"] = pvalue
                # print(
                #    f"Stratified Brier: {stratified_brier_score_loss(y_test, y_prob)}"
                # )
            accuracy_value = float(accuracy_score(y_test, y_pred[[PREDICTION_COLUMN]]))
            precision_value = float(
                precision_score(
                    y_test,
                    y_pred[[PREDICTION_COLUMN]],
                    zero_division=0.0,  # type: ignore
                )
            )
            recall_value = float(recall_score(y_test, y_pred[[PREDICTION_COLUMN]]))
            print(f"Accuracy: {accuracy_value}")
            print(
                f"Precision: {precision_value}"  # type: ignore
            )
            print(f"Recall: {recall_value}")
            stats["accuracy"] = accuracy_value
            stats["precision"] = precision_value
            stats["recall"] = recall_value

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
            with open(
                os.path.join(folder, "stats.json"), "w", encoding="utf8"
            ) as handle:
                json.dump(
                    stats,
                    handle,
                )
            # transform() caches each column's fold listing; drop it so the
            # new fold is picked up. This is only needed here: search trials
            # never save, so they can't change the listing, and the folder
            # only counts as a fold once it has files in it.
            trainer._cached_fold_dates.pop(os.path.dirname(folder), None)

        # return output, loss
        # return pvalue, loss
        return loss
    except WavetrainException as exc:
        print(str(exc))
        logging.warning(str(exc))
        if new_folder:
            with lock:
                try:
                    os.removedirs(folder)
                except OSError:
                    # Another trial may have already removed
                    # it, or written into it in the meantime.
                    pass
        # return BAD_OUTPUT, -BAD_OUTPUT
        # return BAD_OUTPUT, -BAD_OUTPUT
        return -BAD_OUTPUT


class TrialObjective:
    """A picklable Optuna objective that fits one walk-forward fold.

    `frame` is either the DataFrame itself (trials run in this process) or a
    SharedFrame that a worker process memory-maps, so the training frame is
    never pickled per trial. `rows` selects the fold's rows positionally.
    """

    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        trainer: "Trainer",
        frame: pd.DataFrame | SharedFrame,
        rows: slice | np.ndarray | None,
        y_series: pd.Series,
        dt_index: pd.Index,
        split_idx: datetime.datetime,
        copy: bool,
    ):
        self._trainer = trainer
        self._frame = frame
        self._rows = rows
        self._y_series = y_series
        self._dt_index = dt_index
        self._split_idx = split_idx
        self._copy = copy

    def __call__(self, trial: optuna.Trial) -> float:
        # In a worker, the first call maps the shared frame; later calls
        # reuse it.
        x = self._frame if isinstance(self._frame, pd.DataFrame) else self._frame.load()
        if self._rows is not None:
            x = x.iloc[self._rows]
        # Validation folds copy their rows, as they did before trials could
        # run in workers; a shared frame is read-only, so this also gives
        # the trial a writable frame.
        if self._copy:
            x = x.copy()
        # Search trials are evaluated but never saved; Trainer.fit saves the
        # best trial itself once the batch has finished.
        return fit_trial(
            self._trainer,
            trial,
            x,
            self._y_series,
            self._dt_index,
            False,
            self._split_idx,
            False,
        )
