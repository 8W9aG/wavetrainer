"""A normaliser that uses the power transformer."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,protected-access
import json
import os
import warnings
from typing import Self

import joblib  # type: ignore
import numpy as np
import optuna
import pandas as pd
import scipy  # type: ignore
from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
from sklearn.preprocessing import PowerTransformer  # type: ignore

from ..exceptions import WavetrainException
from .normaliser import Normaliser

_POWERTRANSFORMER_REDUCER_FILE = "power_transformer_normaliser.joblib"
_POWERTRANSFORMER_COLUMNS_FILENAME = "power_transformer_columns.json"


class PowerTransformerNormaliser(Normaliser):
    """A class that normalises the training data with the power transformer."""

    _pt_cols: list[str]

    def __init__(self):
        super().__init__()
        self._pt = PowerTransformer()
        self._pt_cols = []

    @classmethod
    def name(cls) -> str:
        return "powertransformer"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InconsistentVersionWarning)
            self._pt = joblib.load(os.path.join(folder, _POWERTRANSFORMER_REDUCER_FILE))

        with open(
            os.path.join(folder, _POWERTRANSFORMER_COLUMNS_FILENAME),
            "r",
            encoding="utf8",
        ) as handle:
            self._pt_cols = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        joblib.dump(self._pt, os.path.join(folder, _POWERTRANSFORMER_REDUCER_FILE))
        with open(
            os.path.join(folder, _POWERTRANSFORMER_COLUMNS_FILENAME),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(self._pt_cols, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._pt_cols = []
            for col in df.columns.values.tolist():
                try:
                    PowerTransformer().fit(df[[col]])
                    self._pt_cols.append(col)
                except scipy.optimize._optimize.BracketError:
                    pass
            self._pt.fit(df[self._pt_cols].to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df[self._pt_cols].values[:] = np.nan_to_num(
                    self._pt.transform(df[self._pt_cols].to_numpy()),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
        except ValueError as exc:
            # --- Diagnostic Check ---
            subset = df[self._pt_cols]

            print("\n" + "=" * 50)
            print("🚨 DATA VALIDATION ERROR DIAGNOSTIC 🚨")

            # Check specifically for infinite values
            inf_mask = np.isinf(subset)
            if inf_mask.any().any():
                bad_cols = [col for col in subset.columns if inf_mask[col].any()]
                print(f"-> Found infinities (inf or -inf) in columns: {bad_cols}")

                for col in bad_cols:
                    print(f"\nOffending values in '{col}':")
                    # Filter and print only the rows where this specific column has an infinity
                    print(subset[col][inf_mask[col]])
            else:
                # Fallback: If it's not strictly 'inf', it's an unusually large number
                print(
                    "-> No strict infinities found. Values are likely exceeding float64 capacity."
                )
                print(
                    "\nTop 5 largest absolute values by column to help you hunt it down:"
                )
                print(subset.abs().max().sort_values(ascending=False).head(5))

            print("=" * 50 + "\n")

            # Raise the exception as usual so the pipeline stops and you can read the logs
            raise WavetrainException() from exc

        return df
