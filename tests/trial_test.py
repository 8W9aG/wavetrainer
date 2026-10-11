"""Tests for fitting one trial of one walk-forward fold."""
import datetime
import multiprocessing
import os
import pickle
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import optuna
import pandas as pd

from wavetrainer.trainer import Trainer
from wavetrainer.trial import BAD_OUTPUT, TrialObjective, best_trial, fold_lock
from wavetrainer.shared_frame import SharedFrame

_HOLD_SECONDS = 2.0


def _hold_lock(folder: str, held_file: str) -> None:
    with fold_lock(folder):
        with open(held_file, "w", encoding="utf8") as handle:
            handle.write("")
        time.sleep(_HOLD_SECONDS)


def _add_trial(study, value, idx):
    study.add_trial(
        optuna.trial.create_trial(
            value=value,
            params={},
            distributions={},
            user_attrs={} if idx is None else {"idx": idx.isoformat()},
        )
    )


def _fail_trial(study, idx):
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL, user_attrs={"idx": idx.isoformat()}
        )
    )


class TestBestTrial(unittest.TestCase):

    def setUp(self):
        self._first = datetime.datetime(2022, 1, 1)
        self._second = datetime.datetime(2022, 1, 8)
        self._study = optuna.create_study()
        _add_trial(self._study, 0.5, self._first)
        _add_trial(self._study, 0.3, self._first)
        _add_trial(self._study, 0.4, self._second)
        _add_trial(self._study, 0.1, self._first)
        _fail_trial(self._study, self._second)

    def test_without_dt_is_the_study_best(self):
        self.assertEqual(0.1, best_trial(self._study).value)

    def test_with_dt_is_the_best_for_that_fold(self):
        self.assertEqual(0.4, best_trial(self._study, dt=self._second).value)
        self.assertEqual(0.1, best_trial(self._study, dt=self._first).value)

    def test_with_unknown_dt_falls_back_to_study_best(self):
        unknown = datetime.datetime(2023, 1, 1)
        self.assertEqual(0.1, best_trial(self._study, dt=unknown).value)


class TestFoldLock(unittest.TestCase):

    def test_reusable_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = os.path.join(tmpdir, "column", "2022-01-01T00:00:00")
            lock = fold_lock(folder)
            for _ in range(3):
                with lock:
                    self.assertTrue(os.path.isdir(os.path.dirname(folder)))
                    os.makedirs(folder, exist_ok=True)
                with lock:
                    os.removedirs(folder)
            # Removing the fold stops at its column folder, which still
            # holds the lock file.
            self.assertTrue(os.path.isdir(os.path.dirname(folder)))

    def test_released_after_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = fold_lock(os.path.join(tmpdir, "column", "fold"))
            with self.assertRaises(RuntimeError):
                with lock:
                    raise RuntimeError("boom")
            with lock:
                pass

    def test_excludes_other_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = os.path.join(tmpdir, "column", "2022-01-01T00:00:00")
            held_file = os.path.join(tmpdir, "held")
            ctx = multiprocessing.get_context("spawn")
            process = ctx.Process(target=_hold_lock, args=(folder, held_file))
            process.start()
            try:
                deadline = time.monotonic() + 120
                while not os.path.exists(held_file) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(os.path.exists(held_file))
                start = time.monotonic()
                with fold_lock(folder):
                    waited = time.monotonic() - start
            finally:
                process.join(timeout=60)
            self.assertEqual(0, process.exitcode)
            self.assertGreater(waited, _HOLD_SECONDS / 4)


class TestTrialObjective(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._trainer = Trainer(
            self._tmpdir.name, walkforward_timedelta=datetime.timedelta(days=7)
        )
        index = pd.date_range("2022-01-01", periods=20, freq="D")
        self._df = pd.DataFrame(
            {"a": np.arange(20.0), "b": np.arange(20.0) * 2}, index=index
        )
        self._y = pd.Series(np.arange(20) % 2 == 0, index=index, name="y")
        self._split = datetime.datetime(2022, 1, 15)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _objective(self, frame, rows, copy):
        return TrialObjective(
            self._trainer, frame, rows, self._y, self._df.index, self._split, copy=copy
        )

    def _call(self, objective):
        with mock.patch("wavetrainer.trial.fit_trial", return_value=0.25) as fit:
            self.assertEqual(0.25, objective(mock.Mock()))
        args = fit.call_args.args
        self.assertIs(self._trainer, args[0])
        self.assertIs(self._y, args[3])
        self.assertEqual(self._split, args[6])
        # Trials are never saved, and always evaluated.
        self.assertFalse(args[5])
        self.assertFalse(args[7])
        return args[2]

    def test_in_process_frame_is_passed_through(self):
        x = self._call(self._objective(self._df, None, copy=False))
        self.assertIs(self._df, x)

    def test_in_process_frame_is_copied(self):
        x = self._call(self._objective(self._df, None, copy=True))
        self.assertIsNot(self._df, x)
        pd.testing.assert_frame_equal(self._df, x)

    def test_shared_frame_slice(self):
        shared = SharedFrame.write(self._df, self._tmpdir.name)
        x = self._call(self._objective(shared, slice(0, 12), copy=False))
        pd.testing.assert_frame_equal(self._df.iloc[:12], x)

    def test_shared_frame_positions(self):
        shared = SharedFrame.write(self._df, self._tmpdir.name)
        rows = np.array([0, 2, 5, 9])
        x = self._call(self._objective(shared, rows, copy=False))
        pd.testing.assert_frame_equal(self._df.iloc[rows], x)

    def test_shared_frame_copy_is_writable(self):
        shared = SharedFrame.write(self._df, self._tmpdir.name)
        x = self._call(self._objective(shared, slice(0, 12), copy=True))
        x.iloc[0, 0] = -1.0
        self.assertEqual(0.0, shared.load().iloc[0, 0])

    def test_pickles_without_the_shared_frame(self):
        wide = pd.DataFrame(np.zeros((200, 500)), index=pd.date_range("2022-01-01", periods=200))
        shared = SharedFrame.write(wide, self._tmpdir.name)
        objective = TrialObjective(
            self._trainer, shared, slice(0, 150), self._y, wide.index, self._split, copy=False
        )
        payload = pickle.dumps(objective)
        self.assertLess(len(payload), wide.memory_usage().sum() / 10)
        restored = pickle.loads(payload)
        x = self._call_restored(restored)
        pd.testing.assert_frame_equal(wide.iloc[:150], x)

    def _call_restored(self, objective):
        with mock.patch("wavetrainer.trial.fit_trial", return_value=0.0) as fit:
            objective(mock.Mock())
        return fit.call_args.args[2]

    def test_bad_output_is_a_large_loss(self):
        # fit_trial returns -BAD_OUTPUT for an unusable fold, which must
        # lose to any real loss when the study minimises.
        self.assertGreater(-BAD_OUTPUT, 1.0)
