"""A reducer that implements Blocked Fast Correlation-Based Feature selection."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,consider-using-enumerate,too-many-locals,invalid-name,too-many-statements
import json
import os
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
import tqdm

from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns
from .reducer import Reducer

_FAST_CORRELATION_REDUCER_FILENAME = "fast_correlation_reducer.json"
_FAST_CORRELATION_THRESHOLD = "fast_correlation_threshold"


def _get_fast_correlated_features_to_drop_blocked(
    df: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.7,
    block_size: int = 500,  # Process 500 features at a time
) -> list[str]:
    """
    Optimized Feature Selection using Blocked Matrix Operations.

    Instead of iterating 1-by-1, this processes features in blocks.
    It compares an entire block of candidates against all previously selected
    features simultaneously using matrix multiplication.
    """
    print(
        f"Blocked correlation reduction (Threshold: {threshold}, Block: {block_size})"
    )

    # 1. Setup and Alignment
    sorted_cols = sorted(find_non_categorical_numeric_columns(df))
    common_index = df.index.intersection(y.index)

    # Use float32 to cut memory usage in half and speed up BLAS operations
    X_values = df.loc[common_index, sorted_cols].values.astype(np.float32)
    y_values = y.loc[common_index].values.astype(np.float32)

    n_samples, n_features = X_values.shape
    if n_features == 0:
        return []

    print("Standardizing data...")
    # 2. Standardization (Z-score)
    # Pre-calculating this allows us to use Dot Product for correlation

    # Simple imputation for NaNs (Mean)
    col_means = np.nanmean(X_values, axis=0)
    inds = np.where(np.isnan(X_values))
    X_values[inds] = np.take(col_means, inds[1])

    # Normalize X
    X_mean = np.mean(X_values, axis=0)
    X_std = np.std(X_values, axis=0)
    X_std[X_std == 0] = 1.0
    X_values = (X_values - X_mean) / X_std

    # Normalize y
    y_mean = np.mean(y_values)
    y_std = np.std(y_values)
    if y_std == 0:
        y_std = 1.0
    y_values = (y_values - y_mean) / y_std

    # Scale by sqrt(n) so dot product == correlation
    scale_factor = np.sqrt(n_samples)
    X_values /= scale_factor
    y_values /= scale_factor

    print("Computing target correlations...")
    # 3. Sort by Relevance
    target_corrs = np.abs(X_values.T @ y_values)
    sorted_indices = np.argsort(-target_corrs)

    # We will store the INDICES of selected features here
    selected_indices = []

    # We maintain a matrix of selected features for fast comparison
    # Shape: (n_samples, n_selected_so_far)
    selected_matrix = None

    print("Processing blocks...")

    # 4. Blocked Iteration
    # We iterate through the sorted features in chunks
    for i in tqdm.tqdm(range(0, n_features, block_size)):
        # Identify the current block of candidate indices
        chunk_indices = sorted_indices[i : i + block_size]
        chunk_data = X_values[:, chunk_indices]  # Shape: (samples, block_size)

        # Mask: True = Keep, False = Drop
        keep_mask = np.ones(len(chunk_indices), dtype=bool)

        # A. External Redundancy Check
        # Check this entire block against ALL previously selected features
        if selected_matrix is not None:
            # Matrix Mult: (n_selected, samples) @ (samples, block_size) -> (n_selected, block_size)
            # This computes correlations between every kept feature and every new candidate
            correlations = np.abs(selected_matrix.T @ chunk_data)

            # For each candidate in block, find max correlation with ANY kept feature
            max_corrs = np.max(correlations, axis=0)

            # Mark as dropped if too correlated with existing
            keep_mask = keep_mask & (max_corrs <= threshold)

        # If everything in the block was dropped, move on
        if not np.any(keep_mask):
            continue

        # B. Internal Redundancy Check
        # We still need to clean up the block itself (e.g. Feat 1 and Feat 2 in block are redundant)
        # We iterate only through the *survivors* of step A

        # Indices relative to the chunk (0 to block_size)
        surviving_rel_indices = np.where(keep_mask)[0]

        final_block_survivors: list[Any] = []

        for rel_idx in surviving_rel_indices:
            # Candidate vector
            candidate_vec = chunk_data[:, rel_idx]
            is_redundant = False

            # Check against other survivors *within this block* that we just accepted
            for accepted_vec in final_block_survivors:
                corr = np.abs(np.dot(accepted_vec, candidate_vec))
                if corr > threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                final_block_survivors.append(candidate_vec)
                # Map back to original global index
                original_idx = chunk_indices[rel_idx]
                selected_indices.append(original_idx)

        # C. Update the Selected Matrix
        # Append new survivors to the comparison matrix for the next loop
        if final_block_survivors:
            new_block_matrix = np.stack(final_block_survivors, axis=1)
            if selected_matrix is None:
                selected_matrix = new_block_matrix
            else:
                selected_matrix = np.hstack([selected_matrix, new_block_matrix])

    # Convert indices back to column names
    selected_set = set(sorted_cols[i] for i in selected_indices)
    to_drop = [col for col in sorted_cols if col not in selected_set]

    print(f"Done. Dropping {len(to_drop)} features.")
    return sorted(to_drop)


class FastCorrelationBasedReducer(Reducer):
    """
    A class that removes features that are highly correlated with other features
    that have a stronger correlation with the target variable.
    """

    _correlation_drop_features: dict[str, bool]

    def __init__(self, block_size: int = 500) -> None:
        self._threshold = 0.0
        self._correlation_drop_features = {}
        self._block_size = block_size

    @classmethod
    def name(cls) -> str:
        return "fast_correlation"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        try:
            self._threshold = trial.suggest_float(
                _FAST_CORRELATION_THRESHOLD, 0.7, 0.99
            )
        except ValueError:
            self._threshold = 0.7

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _FAST_CORRELATION_REDUCER_FILENAME), encoding="utf8"
        ) as handle:
            self._correlation_drop_features = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _FAST_CORRELATION_REDUCER_FILENAME),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(self._correlation_drop_features, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if y is None:
            print("FastCorrelationBasedReducer skipped: No target 'y' provided.")
            self._correlation_drop_features = {}
            return self

        if isinstance(y, pd.DataFrame):
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        drop_features = _get_fast_correlated_features_to_drop_blocked(
            df,
            y=y_series,
            threshold=self._threshold,
            block_size=self._block_size,
        )
        self._correlation_drop_features = {x: True for x in drop_features}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=list(self._correlation_drop_features.keys()), errors="ignore"
        )
