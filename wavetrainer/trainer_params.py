"""The Trainer settings persisted to params.json, so a trained model can be reloaded."""

import dataclasses
import datetime
import json
import os
from typing import Any

_PARAMS_FILENAME = "params.json"
_TRIALS_KEY = "trials"
_WALKFORWARD_TIMEDELTA_KEY = "walkforward_timedelta"
_DAYS_KEY = "days"
_SECONDS_KEY = "seconds"
_TEST_SIZE_KEY = "test_size"
_VALIDATION_SIZE_KEY = "validation_size"
_DT_COLUMN_KEY = "dt_column"
_MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY = "max_false_positive_reduction_steps"
_CORRELATION_CHUNK_SIZE_KEY = "correlation_chunk_size"
_RANDOM_MIX_RATIO_KEY = "random_mix_ratio"


@dataclasses.dataclass
class TrainerParams:
    """The Trainer settings that are saved alongside its models."""

    # pylint: disable=too-many-instance-attributes
    walkforward_timedelta: datetime.timedelta | None
    trials: int
    test_size: float | datetime.timedelta
    validation_size: float | datetime.timedelta
    dt_column: str | None
    max_false_positive_reduction_steps: int | None
    correlation_chunk_size: int | None
    random_mix_ratio: float


def _size_to_json(
    size: float | datetime.timedelta | None,
) -> float | dict[str, int] | None:
    if isinstance(size, float):
        return size
    if isinstance(size, datetime.timedelta):
        return {_DAYS_KEY: size.days, _SECONDS_KEY: size.seconds}
    return None


def _size_from_json(
    value: Any, default: float | datetime.timedelta
) -> float | datetime.timedelta:
    if value is None:
        return default
    if isinstance(value, float):
        return value
    return datetime.timedelta(
        days=value.get(_DAYS_KEY), seconds=value.get(_SECONDS_KEY)
    )


def save_params(folder: str, params: TrainerParams) -> None:
    """Write the params to the folder's params.json."""
    walkforward_timedelta = params.walkforward_timedelta
    if walkforward_timedelta is None:
        raise ValueError("walkforward_timedelta must be set to save params.")
    with open(os.path.join(folder, _PARAMS_FILENAME), "w", encoding="utf8") as handle:
        json.dump(
            {
                _TRIALS_KEY: params.trials,
                _WALKFORWARD_TIMEDELTA_KEY: {
                    _DAYS_KEY: walkforward_timedelta.days,
                    _SECONDS_KEY: walkforward_timedelta.seconds,
                },
                _TEST_SIZE_KEY: _size_to_json(params.test_size),
                _VALIDATION_SIZE_KEY: _size_to_json(params.validation_size),
                _DT_COLUMN_KEY: params.dt_column,
                _MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY: params.max_false_positive_reduction_steps,
                _CORRELATION_CHUNK_SIZE_KEY: params.correlation_chunk_size,
                _RANDOM_MIX_RATIO_KEY: params.random_mix_ratio,
            },
            handle,
        )


def load_params(folder: str, params: TrainerParams) -> TrainerParams:
    """Read the folder's params.json over the given params.

    The saved walk-forward timedelta, trials, sizes and mix ratio win. The
    saved dt column and reduction settings only fill in ones left unset.
    """
    params_file = os.path.join(folder, _PARAMS_FILENAME)
    if not os.path.exists(params_file):
        raise ValueError(
            "walkforward_timedelta must be specified on an untrained model."
        )
    with open(params_file, encoding="utf8") as handle:
        saved = json.load(handle)
    walkforward_dict = saved[_WALKFORWARD_TIMEDELTA_KEY]
    return TrainerParams(
        walkforward_timedelta=datetime.timedelta(
            days=walkforward_dict.get(_DAYS_KEY),
            seconds=walkforward_dict.get(_SECONDS_KEY),
        ),
        trials=saved[_TRIALS_KEY],
        test_size=_size_from_json(saved[_TEST_SIZE_KEY], params.test_size),
        validation_size=_size_from_json(
            saved[_VALIDATION_SIZE_KEY], params.validation_size
        ),
        dt_column=(
            saved[_DT_COLUMN_KEY] if params.dt_column is None else params.dt_column
        ),
        max_false_positive_reduction_steps=(
            saved.get(_MAX_FALSE_POSITIVE_REDUCTION_STEPS_KEY)
            if params.max_false_positive_reduction_steps is None
            else params.max_false_positive_reduction_steps
        ),
        correlation_chunk_size=(
            saved.get(_CORRELATION_CHUNK_SIZE_KEY)
            if params.correlation_chunk_size is None
            else params.correlation_chunk_size
        ),
        random_mix_ratio=saved.get(_RANDOM_MIX_RATIO_KEY, 0.20),
    )
