"""Predicting from, and inspecting, the models a Trainer has saved."""

# pylint: disable=line-too-long
import datetime
import functools
import logging
import os

import pandas as pd
import tqdm
from joblib import Parallel, delayed  # type: ignore

from .calibrator.calibrator_router import CalibratorRouter
from .model.model_router import ModelRouter
from .normaliser.combined_normaliser import CombinedNormaliser
from .reducer.combined_reducer import CombinedReducer
from .selector.selector import Selector


def _assign_bin(timestamp, bins: list[datetime.datetime]) -> int:
    """Find which of the date bins a timestamp falls in."""
    for i in range(len(bins) - 1):
        if bins[i] <= timestamp < bins[i + 1]:
            return i
    return len(bins) - 2  # Assign to last bin if at the end


class TrainerInference:
    """The prediction half of Trainer, which mixes it in.

    Kept apart from training to keep each module a manageable size. The
    attributes below are set by Trainer.__init__; they are declared here so
    type checkers know this class can use them.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-locals,too-few-public-methods
    _folder: str
    _dt_column: str | None
    _validation_size: float | datetime.timedelta
    embedding_cols: list[list[str]] | None
    _correlation_chunk_size: int | None
    _insert_nulls: bool
    _use_correlation_reducer: bool
    _use_power_transformer: bool
    _cached_reducers: dict[str, CombinedReducer]
    _cached_normalizers: dict[str, CombinedNormaliser]
    _cached_models: dict[str, ModelRouter]
    _cached_selectors: dict[str, Selector]
    _cached_calibrators: dict[str, CalibratorRouter]

    def transform(
        self,
        df: pd.DataFrame,
        optimistic: bool = False,
        ignore_no_dates: bool = False,
        n_jobs: int = -1,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict the expected values of the data running columns in parallel.

        `columns`, when given, restricts inference to just these trained
        column names instead of every column found under `self._folder`.
        A caller that already knows some columns' answers by other means
        (e.g. a known-not-happening event) can skip paying for their
        inference entirely, rather than compute them and discard the
        result. Unrecognised names are silently ignored, same as
        `ignore_no_dates` already does for a listed folder with no
        trained dates -- this only ever narrows the existing on-disk
        set, never widens it.
        """
        input_df = df.copy()
        df = df.reindex(sorted(df.columns), axis=1)
        feature_columns = df.columns.values
        dt_index = (
            df.index
            if self._dt_column is None
            else pd.DatetimeIndex(pd.to_datetime(df[self._dt_column]))
        )

        # --- INNER FUNCTION 1: Process a single column directory ---
        def process_single_column(column: str):
            column_path = os.path.join(self._folder, column)
            if not os.path.isdir(column_path):
                return None

            dates = []
            for date_str in os.listdir(column_path):
                date_path = os.path.join(column_path, date_str)
                if not os.path.isdir(date_path) or not os.listdir(date_path):
                    continue
                dates.append(datetime.datetime.fromisoformat(date_str))

            if not dates:
                if ignore_no_dates:
                    return None
                raise ValueError(f"no dates found for {column}.")

            dates = sorted(dates)
            bins: list[datetime.datetime] = sorted(
                [dt_index.min().to_pydatetime()]
                + dates
                + [(dt_index.max() + pd.Timedelta(days=1)).to_pydatetime()]
            )

            # --- INNER FUNCTION 2: The actual prediction logic ---
            def perform_predictions(
                group: pd.DataFrame,
                column_path: str,
                column: str,
                dates: list[datetime.datetime],
            ) -> pd.DataFrame:
                group = group.copy()  # Avoid SettingWithCopyWarning in parallel threads

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

                except (FileNotFoundError, AttributeError) as exc:
                    # We keep this error print so you don't lose debugging info,
                    # but removed the standard progress print to keep tqdm clean.
                    print(f"\nModel {folder} failed:\n{str(exc)}")
                    if isinstance(exc, AttributeError):
                        raise exc
                except KeyError as exc:
                    print(
                        f"\nDIAG KeyError in column={column!r} folder={folder!r}: {exc}",
                        flush=True,
                    )
                    raise exc

                return group

            # Apply the grouped predictions
            old_index = dt_index.copy()
            df_group = df.groupby(
                dt_index.map(functools.partial(_assign_bin, bins=bins))
            )

            if len(df_group) == 1:
                df_out = df_group.apply(
                    functools.partial(
                        perform_predictions,
                        column_path=column_path,
                        column=column,
                        dates=dates,
                    )
                )
            else:
                # Using standard apply to avoid messy parallel tqdm console output
                df_out = df_group.apply(
                    functools.partial(
                        perform_predictions,
                        column_path=column_path,
                        column=column,
                        dates=dates,
                    )
                )

            if self._dt_column is None:
                df_out = df_out.set_index(old_index)

            if isinstance(df_out.index, pd.MultiIndex):
                df_out = df_out.droplevel(0)

            # EXTRACT: Only return the new prediction columns to prevent heavy IPC memory overhead
            new_cols = [c for c in df_out.columns if c not in df.columns]
            return df_out[new_cols]

        # --- PARALLEL EXECUTION ---
        all_columns = os.listdir(self._folder)
        columns_to_process = (
            all_columns
            if columns is None
            else [c for c in columns if c in set(all_columns)]
        )

        # Wrap the Parallel call in tqdm and set return_as="generator"
        parallel_results = list(
            tqdm.tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(process_single_column)(column)
                    for column in columns_to_process
                ),
                total=len(columns_to_process),
                desc="Inferring Columns",
            )
        )

        # Filter out skipped directories (None) and empty dataframes
        valid_results = [
            res for res in parallel_results if res is not None and not res.empty
        ]

        # Stitch all the newly created columns back onto the main dataframe
        if valid_results:
            df = pd.concat([df] + valid_results, axis=1)

        # Ensure no original input columns were dropped
        for col in input_df.columns.values:
            if col not in df.columns.values:
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
                        include_correlation_reducer=self._use_correlation_reducer,
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
                include_correlation_reducer=self._use_correlation_reducer,
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
