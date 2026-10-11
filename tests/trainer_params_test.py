"""Tests for the persisted trainer params."""
import datetime
import json
import os
import tempfile
import unittest

from wavetrainer.trainer import Trainer
from wavetrainer.trainer_params import TrainerParams, load_params, save_params


def _params(**overrides):
    values = dict(
        walkforward_timedelta=datetime.timedelta(days=7, seconds=30),
        trials=12,
        test_size=0.2,
        validation_size=datetime.timedelta(days=14),
        dt_column="dt",
        max_false_positive_reduction_steps=3,
        correlation_chunk_size=500,
        random_mix_ratio=0.3,
    )
    values.update(overrides)
    return TrainerParams(**values)


def _unset(**overrides):
    # What a Trainer passes when reloading: no walk-forward timedelta,
    # defaults for the sizes, and nothing for the optional settings.
    values = dict(
        walkforward_timedelta=None,
        trials=100,
        test_size=0.15,
        validation_size=0.15,
        dt_column=None,
        max_false_positive_reduction_steps=None,
        correlation_chunk_size=None,
        random_mix_ratio=0.2,
    )
    values.update(overrides)
    return TrainerParams(**values)


class TestTrainerParams(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = _params()
            save_params(tmpdir, saved)
            self.assertEqual(saved, load_params(tmpdir, _unset()))

    def test_round_trip_float_validation_and_timedelta_test_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = _params(test_size=datetime.timedelta(days=3), validation_size=0.1)
            save_params(tmpdir, saved)
            self.assertEqual(saved, load_params(tmpdir, _unset()))

    def test_saved_values_win_over_given_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_params(tmpdir, _params())
            loaded = load_params(
                tmpdir,
                _unset(trials=1, test_size=0.5, validation_size=0.5, random_mix_ratio=0.9),
            )
            self.assertEqual(12, loaded.trials)
            self.assertEqual(0.2, loaded.test_size)
            self.assertEqual(datetime.timedelta(days=14), loaded.validation_size)
            self.assertEqual(0.3, loaded.random_mix_ratio)

    def test_given_optional_settings_win_over_saved_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_params(tmpdir, _params())
            loaded = load_params(
                tmpdir,
                _unset(
                    dt_column="other",
                    max_false_positive_reduction_steps=9,
                    correlation_chunk_size=7,
                ),
            )
            self.assertEqual("other", loaded.dt_column)
            self.assertEqual(9, loaded.max_false_positive_reduction_steps)
            self.assertEqual(7, loaded.correlation_chunk_size)

    def test_unsaved_sizes_keep_the_given_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_params(tmpdir, _params(test_size=None, validation_size=None))
            loaded = load_params(tmpdir, _unset())
            self.assertEqual(0.15, loaded.test_size)
            self.assertEqual(0.15, loaded.validation_size)

    def test_old_params_file_without_optional_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "params.json"), "w", encoding="utf8") as handle:
                json.dump(
                    {
                        "trials": 5,
                        "walkforward_timedelta": {"days": 7, "seconds": 0},
                        "test_size": None,
                        "validation_size": None,
                        "dt_column": None,
                    },
                    handle,
                )
            loaded = load_params(tmpdir, _unset())
            self.assertEqual(datetime.timedelta(days=7), loaded.walkforward_timedelta)
            self.assertIsNone(loaded.max_false_positive_reduction_steps)
            self.assertIsNone(loaded.correlation_chunk_size)
            self.assertEqual(0.20, loaded.random_mix_ratio)

    def test_load_without_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                load_params(tmpdir, _unset())

    def test_save_without_walkforward_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_params(tmpdir, _unset())

    def test_trainer_reloads_saved_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Trainer(
                tmpdir,
                walkforward_timedelta=datetime.timedelta(days=7),
                trials=9,
                test_size=datetime.timedelta(days=10),
                dt_column="dt",
                random_mix_ratio=0.5,
            )
            trainer = Trainer(tmpdir)
            self.assertEqual(datetime.timedelta(days=7), trainer._walkforward_timedelta)
            self.assertEqual(9, trainer._trials)
            self.assertEqual(datetime.timedelta(days=10), trainer._test_size)
            self.assertEqual(0.15, trainer._validation_size)
            self.assertEqual("dt", trainer._dt_column)
            self.assertEqual(0.5, trainer._random_mix_ratio)

    def test_trainer_without_params_or_walkforward_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                Trainer(tmpdir)
