"""A wrapper for catboost regressor to handle some edge cases."""

from catboost import CatBoostRegressor  # type: ignore

from .catboost_kwargs import handle_fit_kwargs
from .catboost_lock import CATBOOST_LOCK


class CatBoostRegressorWrapper(CatBoostRegressor):
    """A wrapper for the catboost regressor."""

    def fit(self, *args, **kwargs):
        args, kwargs = handle_fit_kwargs(*args, **kwargs)
        with CATBOOST_LOCK:
            return super().fit(*args, **kwargs)
