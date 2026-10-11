"""A wrapper for catboost classifier to handle some edge cases."""

# pylint: disable=duplicate-code

from catboost import CatBoostClassifier  # type: ignore

from .catboost_kwargs import handle_fit_kwargs
from .catboost_lock import CATBOOST_LOCK


class CatBoostClassifierWrapper(CatBoostClassifier):
    """A wrapper for the catboost classifier."""

    def fit(self, *args, **kwargs):
        args, kwargs = handle_fit_kwargs(*args, **kwargs)
        with CATBOOST_LOCK:
            return super().fit(*args, **kwargs)
