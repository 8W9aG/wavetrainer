"""The prototype model class."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
from typing import Any

import pandas as pd

from ..fit import Fit
from ..params import Params

PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN_PREFIX = "probability_"
QUANTILE_COLUMN_PREFIX = "quantile_"
QUANTILES = sorted([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])


class Model(Params, Fit):
    """The prototype model class."""

    @classmethod
    def name(cls) -> str:
        """The name of the model."""
        raise NotImplementedError("name not implemented in parent class.")

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        """Whether the model supports the X values."""
        raise NotImplementedError("supports_x not implemented in parent class.")

    @property
    def supports_importances(self) -> bool:
        """Whether this model supports feature importances."""
        raise NotImplementedError(
            "supports_importances not implemented in parent class."
        )

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """The feature importances of this model."""
        raise NotImplementedError(
            "feature_importances not implemented in parent class."
        )

    def provide_estimator(self) -> Any:
        """Provides the current estimator."""
        raise NotImplementedError("provides_estimator not implemented in parent class.")

    def create_estimator(self) -> Any:
        """Creates a new estimator."""
        raise NotImplementedError("creates_estimator not implemented in parent class.")

    def reset(self) -> None:
        """Resets a model."""
        raise NotImplementedError("reset not implemented in parent class.")

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts a dataframe for use with a model."""
        raise NotImplementedError("convert_df not implemented in parent class.")
