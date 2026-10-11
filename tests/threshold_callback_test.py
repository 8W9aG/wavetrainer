"""Tests for the threshold callback."""
import unittest

import optuna

from wavetrainer.threshold_callback import ThresholdCallback


def _finished(value):
    return optuna.trial.create_trial(value=value, params={}, distributions={})


class TestThresholdCallback(unittest.TestCase):

    def _study(self, *values):
        study = optuna.create_study()
        for value in values:
            study.add_trial(_finished(value))
        stopped = []
        study.stop = lambda: stopped.append(True)
        return study, stopped

    def test_better_trial_stops(self):
        study, stopped = self._study(0.5)
        ThresholdCallback(1.0)(study, _finished(0.5))
        self.assertEqual([True], stopped)

    def test_equal_or_worse_trial_does_not_stop(self):
        study, stopped = self._study(1.0, 2.0)
        ThresholdCallback(1.0)(study, _finished(1.0))
        ThresholdCallback(1.0)(study, _finished(2.0))
        self.assertEqual([], stopped)

    def test_failed_trial_with_no_finished_trials_does_not_stop(self):
        study, stopped = self._study()
        failed = optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL)
        ThresholdCallback(1.0)(study, failed)
        self.assertEqual([], stopped)

    def test_better_trial_elsewhere_in_storage_stops(self):
        # Another worker's trial beat the threshold; this one did not.
        study, stopped = self._study(0.2)
        ThresholdCallback(1.0)(study, _finished(3.0))
        self.assertEqual([True], stopped)
