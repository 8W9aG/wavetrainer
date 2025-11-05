"""A calibrator class that routes to other calibrators."""

import io
import json
import logging
import os
from typing import Self

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from pycaleva import CalibrationEvaluator  # type: ignore
from sklearn.calibration import calibration_curve  # type: ignore

from ..model.model import PROBABILITY_COLUMN_PREFIX, Model
from ..model_type import ModelType, determine_model_type
from .calibrator import Calibrator
from .vennabers_calibrator import VennabersCalibrator

_CALIBRATOR_ROUTER_FILE = "calibrator_router.json"
_CALIBRATOR_KEY = "calibrator"
_CALIBRATORS = {
    VennabersCalibrator.name(): VennabersCalibrator,
}


class CalibratorRouter(Calibrator):
    """A router that routes to a different calibrator class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    _calibrator: Calibrator | None
    _calibration_buffer: io.BytesIO | None
    _p_value: float

    def __init__(self, model: Model):
        super().__init__(model)
        self._calibrator = None
        self._calibration_buffer = None
        self._p_value = 0.0

    @classmethod
    def name(cls) -> str:
        return "router"

    @property
    def p_value(self) -> float:
        """Find the p value of the fitted calibration."""
        return self._p_value

    def predictions_as_x(self, y: pd.Series | pd.DataFrame | None = None) -> bool:
        calibrator = self._calibrator
        if calibrator is not None:
            return calibrator.predictions_as_x(None)
        return True

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        calibrator = self._calibrator
        if calibrator is None:
            return
        calibrator.set_options(trial, df)

    def load(self, folder: str) -> None:
        file_path = os.path.join(folder, _CALIBRATOR_ROUTER_FILE)
        if not os.path.exists(file_path):
            logging.warning("file %s does not exist", file_path)
            return
        with open(file_path, encoding="utf8") as handle:
            params = json.load(handle)
            calibrator = _CALIBRATORS[params[_CALIBRATOR_KEY]](self._model)
        calibrator.load(folder)
        self._calibrator = calibrator

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        calibrator = self._calibrator
        if calibrator is None:
            logging.warning("calibrator is null")
            return
        calibrator.save(folder, trial)
        with open(
            os.path.join(folder, _CALIBRATOR_ROUTER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _CALIBRATOR_KEY: calibrator.name(),
                },
                handle,
            )
        calibration_buffer = self._calibration_buffer
        if calibration_buffer is not None:
            with open(os.path.join(folder, "calibration.png"), "wb") as handle:
                handle.write(calibration_buffer.getvalue())
            self._calibration_buffer = None

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        # pylint: disable=no-else-return
        calibrator: Calibrator | None = None
        if y is None:
            raise ValueError("y is null")
        if determine_model_type(y) == ModelType.REGRESSION:
            return self
        else:
            calibrator = VennabersCalibrator(self._model)
        calibrator.fit(df, y=y, w=w)
        self._calibrator = calibrator

        pred_prob = calibrator.transform(df)
        pred_prob = pred_prob.drop(
            columns=[
                x
                for x in pred_prob.columns.values.tolist()
                if not x.startswith(PROBABILITY_COLUMN_PREFIX)
            ],
            errors="ignore",
        )
        try:
            ce = CalibrationEvaluator(
                y.to_numpy(),
                pred_prob[PROBABILITY_COLUMN_PREFIX + str(1)].to_numpy(),
                outsample=True,
                n_groups="auto",
            )
            hosmer_lemeshow = ce.hosmerlemeshow()
            print(f"Hosmer Lemeshow: {hosmer_lemeshow}")
            self._p_value = hosmer_lemeshow.pvalue

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y.to_numpy(),
                pred_prob[PROBABILITY_COLUMN_PREFIX + str(1)].to_numpy(),
                n_bins=10,
                strategy="uniform",
            )
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title("Calibration Curve (Reliability Diagram)")
            plt.legend()
            self._calibration_buffer = io.BytesIO()
            plt.savefig(
                self._calibration_buffer, format="png", dpi=300, bbox_inches="tight"
            )
            self._calibration_buffer.seek(0)
            plt.close()
        except ValueError as exc:
            print(str(exc))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        calibrator = self._calibrator
        if calibrator is None:
            logging.warning("calibrator is null")
            return df
        return calibrator.transform(df)
