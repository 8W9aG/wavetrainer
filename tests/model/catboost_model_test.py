"""Tests for the catboost model class."""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import optuna
import pandas as pd

from wavetrainer.model.catboost.catboost_model import CatboostModel


class TestCatboostModel(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = pd.DataFrame(data={
            "a": rng.random(50),
            "b": rng.random(50),
        })
        self.y = pd.Series(data=self.x["a"] * 2.0 + self.x["b"], name="y")
        self.model = CatboostModel()
        self.model.set_options(optuna.trial.FixedTrial({
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 2,
            "l2_leaf_reg": 3.0,
            "boosting_type": "Plain",
            "early_stopping_rounds": 10,
        }), self.x)
        self.model.fit(self.x, y=self.y)
        self._cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()

    def test_transform_does_not_dump_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != "WAVETRAINER_DUMP_INFERENCE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.model.transform(self.x.copy())
        self.assertEqual(os.listdir(self._tmpdir.name), [])

    def test_transform_dumps_to_env_path(self):
        dump_path = os.path.join(self._tmpdir.name, "dump", "inference.parquet")
        os.makedirs(os.path.dirname(dump_path))
        with mock.patch.dict(os.environ, {"WAVETRAINER_DUMP_INFERENCE": dump_path}):
            self.model.transform(self.x.copy())
        self.assertEqual(os.listdir(os.path.dirname(dump_path)), ["inference.parquet"])
        pd.testing.assert_frame_equal(pd.read_parquet(dump_path), self.x)

    def test_transform_dumps_to_default_path(self):
        with mock.patch.dict(os.environ, {"WAVETRAINER_DUMP_INFERENCE": "1"}):
            self.model.transform(self.x.copy())
        pd.testing.assert_frame_equal(
            pd.read_parquet("last_catboost_inference.parquet"), self.x
        )

    def test_transform_predictions_unchanged_by_dump(self):
        with mock.patch.dict(os.environ, {"WAVETRAINER_DUMP_INFERENCE": ""}):
            pred_off = self.model.transform(self.x.copy())
        with mock.patch.dict(os.environ, {"WAVETRAINER_DUMP_INFERENCE": "1"}):
            pred_on = self.model.transform(self.x.copy())
        pd.testing.assert_frame_equal(pred_off, pred_on, check_exact=True)
