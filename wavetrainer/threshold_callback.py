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
            return
        # study.stop() only stops the optimize loop of the process calling
        # it. When trials run in several worker processes sharing a storage,
        # a better trial found by another worker shows up here instead.
        try:
            best_value = study.best_value
        except ValueError:
            return
        if best_value < self._threshold:
            study.stop()
