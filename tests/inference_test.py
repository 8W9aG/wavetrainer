"""Tests for inference from a trained Trainer."""
import datetime
import inspect
import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from wavetrainer.trainer import Trainer


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        handle.write("{}")


class TestFoldDates(unittest.TestCase):
    """The fold listing transform() reads, without training anything."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)
        self.trainer = Trainer(self.folder, walkforward_timedelta=datetime.timedelta(days=7))
        self.column_path = os.path.join(self.folder, "y")
        os.makedirs(self.column_path)
        self.df = pd.DataFrame(
            {"column1": [1.0, 2.0]},
            index=[datetime.datetime(2022, 3, 1), datetime.datetime(2022, 3, 2)],
        )

    def test_lists_non_empty_fold_folders_in_date_order(self):
        _touch(os.path.join(self.column_path, "2022-01-08T00:00:00", "trial.json"))
        _touch(os.path.join(self.column_path, "2022-01-01T00:00:00", "trial.json"))
        os.makedirs(os.path.join(self.column_path, "2022-01-15T00:00:00"))
        _touch(os.path.join(self.column_path, "2022-01-08T00:00:00.lock"))
        _touch(os.path.join(self.column_path, "study.db"))

        self.assertEqual(
            self.trainer._provide_fold_dates(self.column_path),
            [datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 8)],
        )

    def test_listing_is_cached_until_dropped(self):
        _touch(os.path.join(self.column_path, "2022-01-01T00:00:00", "trial.json"))
        first = self.trainer._provide_fold_dates(self.column_path)

        _touch(os.path.join(self.column_path, "2022-01-08T00:00:00", "trial.json"))
        with mock.patch("wavetrainer.inference.os.listdir") as listdir:
            self.assertEqual(self.trainer._provide_fold_dates(self.column_path), first)
        listdir.assert_not_called()

        # fit_trial drops the entry when it saves a fold.
        self.trainer._cached_fold_dates.pop(self.column_path)
        self.assertEqual(
            self.trainer._provide_fold_dates(self.column_path),
            [datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 8)],
        )

    # An empty listing is cached too, so the error and ignore_no_dates must
    # behave the same on a repeat call as on the first.
    def test_no_dates_raises_on_every_call(self):
        os.makedirs(os.path.join(self.column_path, "2022-01-01T00:00:00"))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "no dates found for y"):
                self.trainer.transform(self.df)
        self.assertEqual(self.trainer._cached_fold_dates[self.column_path], [])

    def test_no_dates_ignored_on_every_call(self):
        os.makedirs(os.path.join(self.column_path, "2022-01-01T00:00:00"))
        for _ in range(2):
            out = self.trainer.transform(self.df, ignore_no_dates=True)
            pd.testing.assert_frame_equal(out, self.df)


class TestTransform(unittest.TestCase):
    """transform() against one small trained Trainer."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp()
        x_data = list(range(101))
        x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in x_data]
        # Deliberately not in sorted column order, so the test of the
        # caller's frame shows transform() sorts a copy, not the original.
        cls.df = pd.DataFrame(
            data={
                "column2": [(x * random.random()) + random.random() for x in x_data],
                "column1": [float(x) for x in x_data],
            },
            index=x_index,
        )
        y = pd.DataFrame(
            data={
                "y": [x % 2 == 0 for x in x_data],
                "y2": [float(x) + random.random() for x in x_data],
            },
            index=cls.df.index,
        )
        cls.kwargs = {
            "walkforward_timedelta": datetime.timedelta(days=7),
            "trials": 3,
            "allowed_models": {"catboost"},
            "use_power_transformer": True,
        }
        Trainer(cls.folder, **cls.kwargs).fit(cls.df, y=y)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)

    def _trainer(self):
        return Trainer(self.folder, **self.kwargs)

    # Callers that already run in parallel (one simulation per core) would
    # otherwise start a process per column inside each of theirs.
    def test_n_jobs_defaults_to_one(self):
        default = inspect.signature(Trainer.transform).parameters["n_jobs"].default
        self.assertEqual(default, 1)

    def test_default_keeps_models_cached_across_calls(self):
        trainer = self._trainer()
        trainer.transform(self.df.iloc[[-1]])
        # Run in this process, so the models stay loaded on the Trainer and
        # the next call reuses them; worker processes would load and drop
        # their own copies every call.
        models = dict(trainer._cached_models)
        self.assertTrue(models)
        trainer.transform(self.df.iloc[[-1]])
        self.assertEqual(trainer._cached_models, models)
        for folder, model in models.items():
            self.assertIs(trainer._cached_models[folder], model)

    # Column parallelism is still available by asking for it.
    def test_parallel_columns_match_in_process(self):
        trainer = self._trainer()
        pd.testing.assert_frame_equal(
            trainer.transform(self.df, n_jobs=-1),
            trainer.transform(self.df),
            check_exact=True,
        )

    # transform() no longer copies its input up front, so check it still
    # never writes to the caller's frame.
    def test_callers_frame_is_left_alone(self):
        df = self.df.copy()
        out = self._trainer().transform(df)
        pd.testing.assert_frame_equal(df, self.df, check_exact=True)
        # Every input column comes back, unchanged, in sorted order and
        # ahead of the predictions.
        self.assertEqual(list(out.columns[:2]), ["column1", "column2"])
        pd.testing.assert_frame_equal(out[df.columns], df, check_exact=True)
        self.assertTrue({"y_prediction", "y2_prediction"} <= set(out.columns))
