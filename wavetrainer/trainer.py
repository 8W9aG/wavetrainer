"""The trainer class."""

# pylint: disable=line-too-long,unused-argument
import contextlib
import datetime
import logging
import os
import pickle
import threading
from typing import Callable, Self

import numpy as np
import optuna
import pandas as pd
import tqdm

from .fit import Fit
from .inference import TrainerInference
from .model.catboost.catboost_model import CatboostModel
from .shared_frame import SharedFrame
from .study import SAMPLER_FILENAME, provide_study
from .threshold_callback import ThresholdCallback
from .trainer_params import TrainerParams, load_params, save_params
from .trial import (BAD_OUTPUT, IDX_USR_ATTR_KEY, TrialObjective, best_trial,
                    fit_trial)
from .trial_pool import TrialWorkerPool, exit_on_sigterm, worker_count


class Trainer(TrainerInference, Fit):
    """A class for training and predicting from an array of data."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-locals,too-many-branches,too-many-instance-attributes
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
        use_correlation_reducer: bool = True,
        random_mix_ratio: float = 0.20,
        n_jobs: int = 1,
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

        params = TrainerParams(
            walkforward_timedelta=walkforward_timedelta,
            trials=trials,
            test_size=test_size,
            validation_size=validation_size,
            dt_column=dt_column,
            max_false_positive_reduction_steps=max_false_positive_reduction_steps,
            correlation_chunk_size=correlation_chunk_size,
            random_mix_ratio=random_mix_ratio,
        )
        if walkforward_timedelta is None:
            params = load_params(self._folder, params)
        else:
            save_params(self._folder, params)
        assert params.walkforward_timedelta is not None
        self._walkforward_timedelta = params.walkforward_timedelta
        self._trials = params.trials
        self._test_size = params.test_size
        self._validation_size = params.validation_size
        self._dt_column = params.dt_column
        self._max_train_timeout = max_train_timeout
        self._cutoff_dt = cutoff_dt
        self.embedding_cols = embedding_cols
        self._allowed_models = allowed_models
        self._max_false_positive_reduction_steps = (
            params.max_false_positive_reduction_steps
        )
        self._correlation_chunk_size = params.correlation_chunk_size
        self._insert_nulls = insert_nulls
        self._use_power_transformer = use_power_transformer
        self._use_correlation_reducer = use_correlation_reducer
        self._random_mix_ratio = params.random_mix_ratio
        self._n_jobs = n_jobs
        self._cached_reducers = {}
        self._cached_normalizers = {}
        self._cached_models = {}
        self._cached_selectors = {}
        self._cached_calibrators = {}
        self._cached_fold_dates = {}

    def _provide_study(self, column: str) -> tuple[optuna.Study, str | None]:
        return provide_study(self._folder, column, self._n_jobs, self._random_mix_ratio)

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

        # How many processes trials run in; 1 runs them here, as before.
        workers = worker_count(self._n_jobs)
        timeout = (
            None
            if self._max_train_timeout is None
            else self._max_train_timeout.total_seconds()
        )
        # Created on the first batch of trials that runs in parallel.
        pool: TrialWorkerPool | None = None
        shared_frame: SharedFrame | None = None

        def _optimize(
            study: optuna.Study,
            journal_file: str | None,
            make_objective: Callable[[SharedFrame | None], TrialObjective],
            n_trials: int,
            callbacks: list[ThresholdCallback] | None = None,
            show_progress_bar: bool = False,
        ) -> None:
            """Run a batch of trials, in this process or across the workers.

            make_objective gets None to build an objective over the frame in
            this process, or the SharedFrame to build one for the workers.
            """
            nonlocal pool, shared_frame
            # One trial isn't worth a worker, and an SQLite study (no
            # journal) can only be written from one process.
            if workers <= 1 or n_trials <= 1 or journal_file is None:
                study.optimize(
                    make_objective(None),
                    n_trials=n_trials,
                    show_progress_bar=show_progress_bar,
                    timeout=timeout,
                    callbacks=callbacks,
                    n_jobs=1,
                )
                return
            # The frame is written and the workers started on the first
            # parallel batch of a fit, then reused by every later batch and
            # column of that fit.
            if shared_frame is None:
                shared_frame = SharedFrame.write(df, self._folder)
            if pool is None:
                pool = TrialWorkerPool(workers)
            pool.optimize(
                study,
                journal_file,
                make_objective(shared_frame),
                n_trials,
                timeout=timeout,
                callbacks=callbacks or (),
                show_progress_bar=show_progress_bar,
            )

        def _fit_column(y_series: pd.Series):
            column_dir = os.path.join(self._folder, str(y_series.name))
            if not os.path.exists(column_dir):
                os.mkdir(column_dir)

            study, journal_file = self._provide_study(str(y_series.name))
            trials_before = len(study.trials)

            start_validation_index = (
                dt_index.to_list()[-int(len(dt_index) * self._validation_size) - 1]
                if isinstance(self._validation_size, float)
                else dt_index[
                    dt_index >= (dt_index.to_list()[-1] - self._validation_size)  # type: ignore
                ].to_list()[0]
            )
            test_mask = dt_index < start_validation_index
            test_df = df[test_mask]
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

            # Workers select the rows from the shared frame. With sorted dates
            # they are a prefix, and a slice is a view rather than a copy.
            test_positions = np.flatnonzero(test_mask)
            test_rows: slice | np.ndarray = test_positions
            if len(test_positions) and test_positions[-1] == len(test_positions) - 1:
                test_rows = slice(0, len(test_positions))

            def test_objective(shared: SharedFrame | None) -> TrialObjective:
                return TrialObjective(
                    self,
                    test_df if shared is None else shared,
                    None if shared is None else test_rows,
                    y_series[test_mask],
                    dt_index,
                    start_test_index.to_pydatetime(),
                    copy=False,
                )

            initial_trials = 0
            if self._trials is not None:
                initial_trials = max(self._trials - len(study.trials), 0)
            if initial_trials > 0:
                _optimize(
                    study,
                    journal_file,
                    test_objective,
                    initial_trials,
                    show_progress_bar=True,
                )
            while (
                best_trial(study).values is None
                or best_trial(study).values == (BAD_OUTPUT, -BAD_OUTPUT)
            ) and len(study.trials) < 1000:
                logging.info("Performing extra train")
                _optimize(
                    study, journal_file, test_objective, 1, show_progress_bar=True
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
                if IDX_USR_ATTR_KEY not in trial.user_attrs:
                    # A trial whose worker died before it got this far.
                    continue
                dt_idx = datetime.datetime.fromisoformat(
                    trial.user_attrs[IDX_USR_ATTR_KEY]
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
                    if IDX_USR_ATTR_KEY not in trial.user_attrs:
                        continue
                    dt_idx = datetime.datetime.fromisoformat(
                        trial.user_attrs[IDX_USR_ATTR_KEY]
                    )
                    if dt_idx == test_dt:
                        found = True
                        break
                if found:
                    last_processed_dt = test_dt
                    fit_trial(
                        self,
                        best_trial(study, dt=test_dt),
                        test_df.copy(),
                        test_series,
                        dt_index,
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
                    # Called straight away by _optimize below, so binding the
                    # loop variables late is fine.
                    # pylint: disable=cell-var-from-loop
                    def validate_objective(
                        shared: SharedFrame | None,
                    ) -> TrialObjective:
                        return TrialObjective(
                            self,
                            test_df if shared is None else shared,
                            None if shared is None else slice(0, len(test_df)),
                            test_series,
                            dt_index,
                            test_idx,
                            copy=True,
                        )

                    _optimize(
                        study,
                        journal_file,
                        validate_objective,
                        min(self._trials if self._trials is not None else 5, 5),
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

                # The best trial is fitted and saved here, in this process,
                # once the batch's trials (in workers or not) have finished.
                value = fit_trial(
                    self,
                    best_trial(study, dt=test_dt),
                    test_df.copy(),
                    test_series,
                    dt_index,
                    True,
                    test_idx,
                    False,
                )
                print(f"Saved model with value: {value}")
                last_processed_dt = test_idx

            # Trials ran (here or in workers) if the study gained any.
            if len(study.trials) > trials_before:
                sampler_file = os.path.join(column_dir, SAMPLER_FILENAME)
                with open(sampler_file, "wb") as handle:
                    pickle.dump(study.sampler, handle)

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
                        # fig.write_image() renders via kaleido, which
                        # launches its own headless Chromium subprocess
                        # with no timeout of its own -- confirmed for
                        # real: this hung indefinitely (7+ hours, every
                        # thread idle, zero forward progress) on a real
                        # training run, and the `except RuntimeError`
                        # below can't catch a hang, only a raised
                        # exception. Run it on a daemon thread with a
                        # bounded join instead: a stuck render gets
                        # abandoned (this one diagnostic PNG is simply
                        # skipped) rather than blocking every remaining
                        # column's training forever. `daemon=True` also
                        # means a genuinely wedged render thread can't
                        # block the process from exiting later, either.
                        image_path = os.path.join(
                            column_dir, f"{target_name}_frontier.png"
                        )
                        render_thread = threading.Thread(
                            target=fig.write_image,
                            kwargs={
                                "file": image_path,
                                "format": "png",
                                "width": 800,
                                "height": 600,
                            },
                            daemon=True,
                        )
                        render_thread.start()
                        render_thread.join(timeout=30)
                        if render_thread.is_alive():
                            print(
                                f"{target_name} frontier plot render exceeded "
                                "30s, skipping (kaleido render abandoned)."
                            )
                    except RuntimeError as exc:
                        print(str(exc))

        # Make SIGTERM run the finally below, so the workers are stopped and
        # the shared frame deleted rather than left behind.
        sigterm = exit_on_sigterm() if workers > 1 else contextlib.nullcontext()
        with sigterm:
            try:
                if isinstance(y, pd.Series):
                    _fit_column(y)
                else:
                    for col in y.columns:
                        print(f"Training column: {col}")
                        y_train_series = y[col]
                        y_train_series.attrs = y.attrs
                        _fit_column(y_train_series)
            finally:
                if pool is not None:
                    pool.close()
                if shared_frame is not None:
                    shared_frame.remove()

        return self
