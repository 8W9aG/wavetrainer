"""Tests for the trainer class."""
import datetime
import os
import random
import tempfile
import threading
import unittest

import pandas as pd

from wavetrainer.trainer import Trainer, _fold_lock
from wavetrainer.model_type import QUANTILE_KEY


class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, allowed_models={"catboost"})
            x_data = [i for i in range(101)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                    "column2": [(x * random.random()) + random.random() for x in x_data],
                    "column3": [int(((x / random.random()) - random.random()) * 1000.0) for x in x_data],
                },
                index=x_index,
            )
            df["column3"] = df["column3"].astype('category')
            y = pd.DataFrame(
                data={
                    "y": [x % 2 == 0 for x in x_data],
                    "y2": [(x + 2) % 3 == 0 for x in x_data],
                    "y3": [float(x) + 2.0 for x in x_data],
                },
                index=df.index,
            )
            trainer.fit(df, y=y)
            df = trainer.transform(df)
            print("df:")
            print(df)

    def test_trainer_dt_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, dt_column="dt_column", allowed_models={"catboost"})
            x_data = [i for i in range(100)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                    "dt_column": x_index,
                },
            )
            y = pd.DataFrame(
                data={
                    "y": [x % 2 == 0 for x in x_data],
                },
                index=df.index,
            )
            y["y"] = y["y"].astype(bool)
            trainer.fit(df, y=y)
            df = trainer.transform(df)
            print("df:")
            print(df)

    def test_concurrent_fold_folder_creation(self):
        # Regression test: optuna runs trials for the same walk-forward
        # fold concurrently across threads (study.optimize(n_jobs>1)), and
        # those trials share the same fold folder path. Creating
        # (os.makedirs) and tearing down (os.removedirs) that folder used
        # to race, raising FileExistsError/FileNotFoundError even though
        # os.makedirs was called with exist_ok=True, because one thread
        # could remove the directory out from under another thread that
        # had just created or was still using it.
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = os.path.join(tmpdir, "some_column", "2022-01-01T00:00:00")

            errors = []

            def worker():
                for _ in range(200):
                    try:
                        fold_lock = _fold_lock(folder)
                        with fold_lock:
                            new_folder = not os.path.exists(folder)
                            os.makedirs(folder, exist_ok=True)
                        if new_folder:
                            with fold_lock:
                                try:
                                    os.removedirs(folder)
                                except OSError:
                                    pass
                    except OSError as exc:  # pragma: no cover - failure path
                        errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual([], errors)

    def test_quantile_trainer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, allowed_models={"catboost"})
            x_data = [i for i in range(101)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                },
                index=x_index,
            )
            y = pd.DataFrame(
                data={
                    "y": [float(x + 1) for x in x_data],
                },
                index=df.index,
            )
            y.attrs = {QUANTILE_KEY: True}
            trainer.fit(df, y=y)
            df = trainer.transform(df)
            print("df:")
            print(df)

    def test_trainer_with_power_transformer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                tmpdir,
                walkforward_timedelta=datetime.timedelta(days=7),
                trials=5,
                allowed_models={"catboost"},
                use_power_transformer=True,
            )
            x_data = [i for i in range(101)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                    "column2": [(x * random.random()) + random.random() for x in x_data],
                    "column3": [int(((x / random.random()) - random.random()) * 1000.0) for x in x_data],
                },
                index=x_index,
            )
            df["column3"] = df["column3"].astype('category')
            y = pd.DataFrame(
                data={
                    "y": [x % 2 == 0 for x in x_data],
                    "y2": [(x + 2) % 3 == 0 for x in x_data],
                    "y3": [float(x) + 2.0 for x in x_data],
                },
                index=df.index,
            )
            trainer.fit(df, y=y)
            df = trainer.transform(df)
            print("df:")
            print(df)