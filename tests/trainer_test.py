"""Tests for the trainer class."""
import datetime
import json
import os
import random
import signal
import tempfile
import threading
import unittest
from unittest import mock

import optuna
import pandas as pd

from wavetrainer.trainer import Trainer
from wavetrainer.trial import TrialObjective, fold_lock
from wavetrainer.model_type import QUANTILE_KEY
from wavetrainer.trial_pool import journal_storage


class _CrashingObjective(TrialObjective):
    """Kills its worker process on trial 2, the way a segfault would."""

    def __call__(self, trial):
        if trial.number == 2:
            os._exit(139)
        return super().__call__(trial)


def _parallel_data(days=101):
    x_data = list(range(days))
    x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in x_data]
    df = pd.DataFrame(
        data={
            "column1": x_data,
            "column2": [(x * random.random()) + random.random() for x in x_data],
            # Categorical, so workers see codes backed by a memory map.
            "column3": pd.Categorical([str(x % 4) for x in x_data]),
        },
        index=x_index,
    )
    y = pd.DataFrame(data={"y": [x % 2 == 0 for x in x_data]}, index=df.index)
    return df, y


def _load_study(folder, column="y"):
    return optuna.load_study(
        study_name="wavetrain",
        storage=journal_storage(os.path.join(folder, column, "study.journal")),
    )


def _fit_with_deadline(trainer, df, y, seconds=900):
    # Fit stays on the main thread (it plots with matplotlib), so a hang is
    # turned into a failure with an alarm rather than a watchdog thread.
    def _hung(signum, frame):  # pragma: no cover - failure path
        raise AssertionError("Trainer.fit hung")

    previous = signal.signal(signal.SIGALRM, _hung)
    signal.alarm(seconds)
    try:
        trainer.fit(df, y=y)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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
            # transform() runs in-process by default, so ask for workers here
            # to keep the per-column parallel path tested.
            df = trainer.transform(df, n_jobs=-1)
            print("df:")
            print(df)

    def test_transform_columns_restricts_to_the_given_trained_targets(self):
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
                },
                index=df.index,
            )
            trainer.fit(df, y=y)

            restricted = trainer.transform(df, columns=["y"])
            new_cols = [c for c in restricted.columns if c not in df.columns]
            self.assertTrue(any(c.startswith("y_") for c in new_cols))
            self.assertFalse(any(c.startswith("y2_") for c in new_cols))

            # An unrecognised name is silently ignored, not an error --
            # same tolerance ignore_no_dates already gives a listed
            # folder with no trained dates.
            unknown_only = trainer.transform(df, columns=["not_a_real_column"])
            self.assertEqual(
                sorted(unknown_only.columns), sorted(df.columns)
            )

            # columns=None (the default) still processes everything.
            full = trainer.transform(df)
            full_new_cols = [c for c in full.columns if c not in df.columns]
            self.assertTrue(any(c.startswith("y_") for c in full_new_cols))
            self.assertTrue(any(c.startswith("y2_") for c in full_new_cols))

    def test_transform_picks_up_a_newly_trained_fold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=3, allowed_models={"catboost"})
            x_data = [i for i in range(101)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                    "column2": [(x * random.random()) + random.random() for x in x_data],
                },
                index=x_index,
            )
            y = pd.DataFrame(data={"y": [x % 2 == 0 for x in x_data]}, index=df.index)
            column_path = os.path.join(tmpdir, "y")

            def folds_on_disk():
                return sorted(
                    x for x in os.listdir(column_path)
                    if os.path.isdir(os.path.join(column_path, x))
                    and os.listdir(os.path.join(column_path, x))
                )

            def used_folds(frame):
                used = []
                provide_model = trainer._provide_model

                def spy(folder):
                    used.append(os.path.basename(folder))
                    return provide_model(folder)

                with mock.patch.object(trainer, "_provide_model", side_effect=spy):
                    out = trainer.transform(frame)
                self.assertIn("y_prediction", out.columns)
                return sorted(set(used))

            trainer.fit(df.iloc[:60], y=y.iloc[:60])
            first_folds = folds_on_disk()
            self.assertTrue(first_folds)
            self.assertEqual(used_folds(df.iloc[[-1]]), [first_folds[-1]])
            self.assertEqual(
                [d.isoformat() for d in trainer._cached_fold_dates[column_path]],
                first_folds,
            )

            # A warm call reads the cached listing rather than the fold folders.
            with mock.patch("wavetrainer.inference.os.listdir", wraps=os.listdir) as listdir:
                self.assertEqual(used_folds(df.iloc[[-1]]), [first_folds[-1]])
            self.assertEqual([c.args for c in listdir.call_args_list], [(tmpdir,)])

            # Training more data on the same Trainer saves later folds, and
            # transform() has to use them rather than a stale listing.
            trainer.fit(df, y=y)
            second_folds = folds_on_disk()
            self.assertGreater(second_folds[-1], first_folds[-1])
            self.assertEqual(used_folds(df.iloc[[-1]]), [second_folds[-1]])

            # A fresh Trainer on the same folder agrees.
            fresh = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=3, allowed_models={"catboost"})
            pd.testing.assert_frame_equal(
                fresh.transform(df.iloc[[-1]]),
                trainer.transform(df.iloc[[-1]]),
                check_exact=True,
            )

    def test_regression_trials_score_by_r2(self):
        # Regression trials used to all return a loss of 0.0, so the
        # optimiser had nothing to tell them apart by.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, allowed_models={"catboost"})
            x_data = [i for i in range(101)]
            x_index = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=i) for i in range(len(x_data))]
            df = pd.DataFrame(
                data={
                    "column1": x_data,
                    "column2": [(x * random.random()) + random.random() for x in x_data],
                },
                index=x_index,
            )
            y = pd.DataFrame(
                data={"y": [float(x) + 2.0 + random.random() for x in x_data]},
                index=df.index,
            )
            trainer.fit(df, y=y)

            column_path = os.path.join(tmpdir, "y")
            folds = [
                os.path.join(column_path, x) for x in os.listdir(column_path)
                if os.path.exists(os.path.join(column_path, x, "trial.json"))
            ]
            self.assertTrue(folds)
            for fold in folds:
                with open(os.path.join(fold, "trial.json"), encoding="utf8") as handle:
                    r2, loss = json.load(handle)["output"]
                with open(os.path.join(fold, "stats.json"), encoding="utf8") as handle:
                    self.assertEqual(json.load(handle)["R2"], r2)
                self.assertEqual(loss, -r2)

            # With n_jobs=1 the study is kept in SQLite rather than a journal.
            study = optuna.load_study(
                study_name="wavetrain",
                storage="sqlite:///" + os.path.join(column_path, "study.db"),
            )
            values = [
                t.value for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            self.assertGreater(len(set(values)), 1)
            self.assertNotIn(0.0, values)

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
                        lock = fold_lock(folder)
                        with lock:
                            new_folder = not os.path.exists(folder)
                            os.makedirs(folder, exist_ok=True)
                        if new_folder:
                            with lock:
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

    def test_trainer_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, allowed_models={"catboost"}, n_jobs=2)
            df, y = _parallel_data()
            _fit_with_deadline(trainer, df, y)

            study = _load_study(tmpdir)
            first_idx = min(t.user_attrs["idx"] for t in study.trials)
            initial = [t for t in study.trials if t.user_attrs["idx"] == first_idx]
            # The first batch of trials runs in workers and is split across
            # them, but the total is still the requested number of trials.
            self.assertEqual(5, len(initial))
            self.assertNotIn(optuna.trial.TrialState.RUNNING, [t.state for t in study.trials])
            # The shared copy of the frame is cleaned up.
            self.assertEqual([], [f for f in os.listdir(tmpdir) if f.startswith(".wavetrainer-frame")])

            out = trainer.transform(df)
            self.assertIn("y_prediction", out.columns)

    def test_trainer_parallel_resumes_sequential_study(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df, y = _parallel_data()
            cutoff = datetime.datetime(2022, 3, 25)
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=4, allowed_models={"catboost"}, cutoff_dt=cutoff)
            trainer.fit(df, y=y)
            sqlite_study = optuna.load_study(
                study_name="wavetrain",
                storage=f"sqlite:///{tmpdir}/y/study.db",
            )
            before = [(t.number, t.value, t.user_attrs.get("idx")) for t in sqlite_study.trials]
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "y", "study.journal")))

            # Resume in parallel on more data, which adds new folds.
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=4, allowed_models={"catboost"}, n_jobs=2)
            _fit_with_deadline(trainer, df, y)
            study = _load_study(tmpdir)
            after = [(t.number, t.value, t.user_attrs.get("idx")) for t in study.trials]
            self.assertEqual(before, after[: len(before)])
            self.assertGreater(len(after), len(before))

            # Once a column has a journal, sequential runs keep using it.
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=4, allowed_models={"catboost"})
            study, journal_file = trainer._provide_study("y")
            self.assertIsNotNone(journal_file)
            self.assertEqual(len(after), len(study.trials))
            trainer.fit(df, y=y)
            self.assertEqual(len(after), len(_load_study(tmpdir).trials))
            self.assertIn("y_prediction", trainer.transform(df).columns)

    def test_trainer_parallel_survives_worker_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(tmpdir, walkforward_timedelta=datetime.timedelta(days=7), trials=5, allowed_models={"catboost"}, n_jobs=2)
            df, y = _parallel_data()
            with mock.patch("wavetrainer.trainer.TrialObjective", _CrashingObjective):
                with self.assertLogs(level="ERROR") as logs:
                    _fit_with_deadline(trainer, df, y)
            self.assertTrue(any("exit code 139" in line for line in logs.output))

            study = _load_study(tmpdir)
            states = [t.state for t in study.trials]
            self.assertEqual(optuna.trial.TrialState.FAIL, study.trials[2].state)
            self.assertNotIn(optuna.trial.TrialState.RUNNING, states)
            # The crashed trial's share was rerun, so the first batch still
            # produced the requested number of trials.
            self.assertEqual(5, sum(1 for t in study.trials[:5]))
            self.assertGreaterEqual(states.count(optuna.trial.TrialState.COMPLETE), 4)
            self.assertIn("y_prediction", trainer.transform(df).columns)
