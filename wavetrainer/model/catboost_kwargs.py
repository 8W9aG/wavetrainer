"""A list of constant catboost kwargs."""

from typing import Any

import numpy as np
from catboost import Pool  # type: ignore

ORIGINAL_X_ARG_KEY = "original_x"
EVAL_SET_ARG_KEY = "eval_set"
CAT_FEATURES_ARG_KEY = "cat_features"


def handle_fit_kwargs(*args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Handles keyword args coming into a catboost fit method."""
    if ORIGINAL_X_ARG_KEY in kwargs:
        df = kwargs[ORIGINAL_X_ARG_KEY]
        eval_x, eval_y = kwargs[EVAL_SET_ARG_KEY]
        cat_features = kwargs[CAT_FEATURES_ARG_KEY]
        args_list = list(args)
        fit_x = args_list[0]
        fix_x_cp = fit_x.copy()

        # Stupid code to ensure eval is feature equivalent to train data
        included_columns = []
        for i in range(fix_x_cp.shape[1]):
            arr_col_values = fix_x_cp[:, i]
            for col in df.columns:
                if col in included_columns:
                    continue
                df_col_values = df[col].values
                if np.allclose(df_col_values, arr_col_values, equal_nan=True):
                    included_columns.append(col)
                    break
        # We also need to update cat_features or catboost will yell at us
        cat_features = list(
            set(list(kwargs.get(CAT_FEATURES_ARG_KEY, []))) & set(included_columns)
        )
        args_list[0] = df[included_columns]
        args = tuple(args_list)

        eval_x = eval_x[included_columns]
        kwargs[EVAL_SET_ARG_KEY] = Pool(
            eval_x,
            label=eval_y,
            cat_features=cat_features,
        )
        kwargs[CAT_FEATURES_ARG_KEY] = cat_features

        del kwargs[ORIGINAL_X_ARG_KEY]
    return args, kwargs
