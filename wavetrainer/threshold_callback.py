"""An optuna threshold callback to ensure a trial beats the best trial."""

# pylint: disable=too-few-public-methods
import optuna


class ThresholdCallback:
    """A callback that stops training once a better trial has been found."""

    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.value is not None and trial.value < self._threshold:
            study.stop()
