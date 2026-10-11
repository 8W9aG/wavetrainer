"""Tests for creating and resuming a column's study."""
import os
import pickle
import tempfile
import unittest

import optuna

from wavetrainer.drift_mixed_sampler import DriftMixedSampler
from wavetrainer.study import provide_study


def _objective(trial: optuna.Trial) -> float:
    trial.set_user_attr("idx", "2022-01-01T00:00:00")
    return trial.suggest_float("x", -1.0, 1.0) ** 2


class TestProvideStudy(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._folder = self._tmpdir.name
        os.mkdir(os.path.join(self._folder, "y"))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _path(self, name):
        return os.path.join(self._folder, "y", name)

    def test_sequential_uses_sqlite(self):
        study, journal_file = provide_study(self._folder, "y", 1, 0.2)
        self.assertIsNone(journal_file)
        self.assertTrue(os.path.exists(self._path("study.db")))
        self.assertFalse(os.path.exists(self._path("study.journal")))
        self.assertIsInstance(study.sampler, DriftMixedSampler)

    def test_parallel_uses_journal(self):
        study, journal_file = provide_study(self._folder, "y", 2, 0.2)
        self.assertEqual(self._path("study.journal"), journal_file)
        self.assertFalse(os.path.exists(self._path("study.db")))
        study.optimize(_objective, n_trials=2)
        study, _ = provide_study(self._folder, "y", 2, 0.2)
        self.assertEqual(2, len(study.trials))

    def test_negative_n_jobs_counts_as_parallel(self):
        _, journal_file = provide_study(self._folder, "y", -1, 0.2)
        if (os.cpu_count() or 1) > 1:
            self.assertIsNotNone(journal_file)

    def test_parallel_migrates_sqlite_study(self):
        study, _ = provide_study(self._folder, "y", 1, 0.2)
        study.optimize(_objective, n_trials=4)
        with open(self._path("sampler.pkl"), "wb") as handle:
            pickle.dump(study.sampler, handle)
        before = [(t.number, t.state, t.value, t.params, t.user_attrs) for t in study.trials]

        migrated, journal_file = provide_study(self._folder, "y", 2, 0.2)
        self.assertEqual(self._path("study.journal"), journal_file)
        after = [(t.number, t.state, t.value, t.params, t.user_attrs) for t in migrated.trials]
        self.assertEqual(before, after)
        # The database and its sampler are kept as a backup.
        self.assertTrue(os.path.exists(self._path("study.db")))
        self.assertTrue(os.path.exists(self._path("sampler.pkl.sqlite")))
        self.assertFalse(os.path.exists(self._path("sampler.pkl")))
        self.assertFalse(os.path.exists(self._path("study.journal.migrating")))
        # The migrated study can carry on sampling.
        migrated.optimize(_objective, n_trials=2)
        self.assertEqual(6, len(migrated.trials))

    def test_journal_is_kept_for_later_sequential_runs(self):
        study, _ = provide_study(self._folder, "y", 1, 0.2)
        study.optimize(_objective, n_trials=1)
        parallel, _ = provide_study(self._folder, "y", 2, 0.2)
        parallel.optimize(_objective, n_trials=2)

        sequential, journal_file = provide_study(self._folder, "y", 1, 0.2)
        self.assertEqual(self._path("study.journal"), journal_file)
        self.assertEqual(3, len(sequential.trials))

    def test_interrupted_migration_is_redone(self):
        study, _ = provide_study(self._folder, "y", 1, 0.2)
        study.optimize(_objective, n_trials=2)
        with open(self._path("study.journal.migrating"), "w", encoding="utf8") as handle:
            handle.write("partial")

        migrated, _ = provide_study(self._folder, "y", 2, 0.2)
        self.assertEqual(2, len(migrated.trials))
        self.assertFalse(os.path.exists(self._path("study.journal.migrating")))

    def test_saved_sampler_is_restored(self):
        provide_study(self._folder, "y", 1, 0.2)
        with open(self._path("sampler.pkl"), "wb") as handle:
            pickle.dump(DriftMixedSampler(random_mix_ratio=0.5), handle)
        study, _ = provide_study(self._folder, "y", 1, 0.2)
        self.assertEqual(0.5, study.sampler._random_mix_ratio)

    def test_new_sampler_uses_mix_ratio(self):
        study, _ = provide_study(self._folder, "y", 1, 0.4)
        self.assertEqual(0.4, study.sampler._random_mix_ratio)
