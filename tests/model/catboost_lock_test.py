"""Tests that concurrent catboost calls are serialised by CATBOOST_LOCK."""
import contextlib
import pickle
import threading
import time
import unittest
from unittest import mock

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

from wavetrainer.model.catboost.catboost_lock import CATBOOST_LOCK
from wavetrainer.model.catboost.catboost_model import CatboostModel
from wavetrainer.model.catboost.catboost_regressor_wrap import CatBoostRegressorWrapper

_N_THREADS = 4


def _make_data(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(data={"a": rng.random(50), "b": rng.random(50)})
    y = pd.Series(data=x["a"] * 2.0 + x["b"], name="y")
    return x, y


def _make_model(x: pd.DataFrame) -> CatboostModel:
    model = CatboostModel()
    model.set_options(optuna.trial.FixedTrial({
        "iterations": 50,
        "learning_rate": 0.1,
        "depth": 2,
        "l2_leaf_reg": 3.0,
        "boosting_type": "Plain",
        "early_stopping_rounds": 10,
    }), x)
    return model


def _run_threads(target) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(_N_THREADS)

    def run(i: int) -> None:
        try:
            barrier.wait()
            target(i)
        except BaseException as exc:  # pylint: disable=broad-except
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(_N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "fit thread hung"
    return errors


def _recording_fit(intervals: list[tuple[float, float]], original=None):
    def fit(self, *args, **kwargs):
        start = time.perf_counter()
        # Widen the window so overlapping fits would be caught.
        time.sleep(0.05)
        result = original(self, *args, **kwargs) if original is not None else self
        intervals.append((start, time.perf_counter()))
        return result
    return fit


def _overlaps(intervals: list[tuple[float, float]]) -> bool:
    ordered = sorted(intervals)
    return any(nxt[0] < prev[1] for prev, nxt in zip(ordered, ordered[1:]))


class TestCatboostLock(unittest.TestCase):

    def test_concurrent_fits_complete_and_never_overlap(self):
        intervals: list[tuple[float, float]] = []
        data = [_make_data(seed) for seed in range(_N_THREADS)]
        preds: dict[int, pd.DataFrame] = {}

        def fit_and_predict(i: int) -> None:
            x, y = data[i]
            model = _make_model(x)
            model.fit(x, y=y, eval_x=x, eval_y=y)
            preds[i] = model.transform(x.copy())

        recorder = _recording_fit(intervals, CatBoostRegressor.fit)
        with mock.patch.object(CatBoostRegressor, "fit", recorder):
            errors = _run_threads(fit_and_predict)

        self.assertEqual(errors, [])
        self.assertEqual(len(intervals), _N_THREADS)
        self.assertFalse(_overlaps(intervals), intervals)
        for i in range(_N_THREADS):
            self.assertEqual(len(preds[i]), len(data[i][0]))

    def test_overlap_is_detected_without_lock(self):
        # Sanity check that the overlap assertion above can actually fail.
        intervals: list[tuple[float, float]] = []
        x, y = _make_data(0)
        with mock.patch.object(CatBoostRegressor, "fit", _recording_fit(intervals)), \
                mock.patch(
                    "wavetrainer.model.catboost.catboost_regressor_wrap.CATBOOST_LOCK",
                    contextlib.nullcontext(),
                ):
            errors = _run_threads(lambda _: CatBoostRegressorWrapper().fit(x, y))
        self.assertEqual(errors, [])
        self.assertTrue(_overlaps(intervals), intervals)

    def test_transform_waits_for_lock(self):
        x, y = _make_data(0)
        model = _make_model(x)
        model.fit(x, y=y)
        done = threading.Event()

        def transform() -> None:
            model.transform(x.copy())
            done.set()

        with CATBOOST_LOCK:
            thread = threading.Thread(target=transform)
            thread.start()
            self.assertFalse(done.wait(0.2))
        thread.join(timeout=30)
        self.assertTrue(done.is_set())

    def test_model_is_picklable(self):
        x, y = _make_data(0)
        model = _make_model(x)
        model.fit(x, y=y)
        restored = pickle.loads(pickle.dumps(model))
        pd.testing.assert_frame_equal(
            restored.transform(x.copy()), model.transform(x.copy())
        )
