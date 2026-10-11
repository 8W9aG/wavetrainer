"""Tests for the trial worker pool."""
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import pickle
import unittest
from unittest import mock

import numpy as np
import optuna
import pandas as pd

from wavetrainer import shared_frame, trial_pool
from wavetrainer.drift_mixed_sampler import DriftMixedSampler
from wavetrainer.threshold_callback import ThresholdCallback
from wavetrainer.shared_frame import SharedFrame
from wavetrainer.trial_pool import (TrialWorkerPool, exit_on_sigterm,
                                    journal_storage, worker_count)

# Long enough for a broken pool to show up as a test failure rather than
# hanging the suite, short enough for a slow machine spawning workers.
_HANG_SECONDS = 300


def _quadratic(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    return x * x


def _crash_on_trial_one(trial: optuna.Trial) -> float:
    if trial.number == 1:
        os._exit(139)
    return _quadratic(trial)


def _always_crash(trial: optuna.Trial) -> float:
    os._exit(139)


def _raise_value_error(trial: optuna.Trial) -> float:
    raise ValueError("bad trial")


def _first_trial_best(trial: optuna.Trial) -> float:
    trial.suggest_float("x", -10.0, 10.0)
    if trial.number == 0:
        return 0.0
    time.sleep(0.5)
    return 1.0


def _run_with_deadline(target):
    errors = []

    def _target():
        try:
            target()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=_HANG_SECONDS)
    if thread.is_alive():  # pragma: no cover - failure path
        raise AssertionError("TrialWorkerPool.optimize hung")
    if errors:
        raise errors[0]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestTrialWorkerPool(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmpdir.name, "study.journal")
        self._study = optuna.create_study(
            study_name="wavetrain",
            storage=journal_storage(self._path),
            sampler=DriftMixedSampler(seed=0),
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def _states(self):
        return [t.state for t in self._study.get_trials(deepcopy=False)]

    def test_total_trials_match_n_trials(self):
        with TrialWorkerPool(3) as pool:
            _run_with_deadline(
                lambda: pool.optimize(self._study, self._path, _quadratic, 7)
            )
        trials = self._study.get_trials(deepcopy=False)
        self.assertEqual(7, len(trials))
        self.assertTrue(all(t.state == optuna.trial.TrialState.COMPLETE for t in trials))
        # Each worker reseeds its copy of the sampler; with the parent's
        # seed shared unchanged they would propose the same parameters.
        self.assertEqual(7, len({t.params["x"] for t in trials}))

    def test_crashed_worker_does_not_hang_and_its_trials_are_rerun(self):
        with TrialWorkerPool(2) as pool:
            with self.assertLogs(level="ERROR") as logs:
                _run_with_deadline(
                    lambda: pool.optimize(self._study, self._path, _crash_on_trial_one, 6)
                )
            # The pool keeps working after the crash.
            _run_with_deadline(
                lambda: pool.optimize(self._study, self._path, _quadratic, 2)
            )
        self.assertTrue(any("exit code 139" in line for line in logs.output))
        states = self._states()
        self.assertEqual(8, len(states))
        self.assertEqual(1, states.count(optuna.trial.TrialState.FAIL))
        self.assertEqual(7, states.count(optuna.trial.TrialState.COMPLETE))

    def test_worker_crashing_every_time_gives_up(self):
        with TrialWorkerPool(2) as pool:
            with self.assertLogs(level="ERROR"):
                _run_with_deadline(
                    lambda: pool.optimize(self._study, self._path, _always_crash, 20)
                )
        states = self._states()
        self.assertTrue(states)
        self.assertLess(len(states), 20)
        self.assertTrue(all(s == optuna.trial.TrialState.FAIL for s in states))

    def test_trial_exception_is_reraised(self):
        with TrialWorkerPool(2) as pool:
            with self.assertRaises(ValueError):
                pool.optimize(self._study, self._path, _raise_value_error, 4)
        self.assertNotIn(optuna.trial.TrialState.RUNNING, self._states())

    def test_threshold_callback_stops_every_worker(self):
        with TrialWorkerPool(2) as pool:
            _run_with_deadline(
                lambda: pool.optimize(
                    self._study,
                    self._path,
                    _first_trial_best,
                    40,
                    callbacks=[ThresholdCallback(0.5)],
                )
            )
        self.assertLess(len(self._states()), 10)

    def test_threshold_callback_sees_trials_from_other_processes(self):
        other = optuna.load_study(study_name="wavetrain", storage=journal_storage(self._path))
        other.enqueue_trial({"x": 0.0})
        other.optimize(_quadratic, n_trials=1)
        stopped = []
        self._study.stop = lambda: stopped.append(True)
        trial = self._study.get_trials(deepcopy=False)[0]
        callback = ThresholdCallback(1.0)
        callback(self._study, optuna.trial.create_trial(value=5.0, params={}, distributions={}))
        self.assertEqual([True], stopped)
        self.assertEqual(0.0, trial.value)


_SIGNAL_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    import optuna
    from wavetrainer.trial_pool import TrialWorkerPool, exit_on_sigterm, journal_storage

    def sleepy(trial):
        with open(os.path.join(sys.argv[1], f"{os.getpid()}.pid"), "w") as handle:
            handle.write("")
        time.sleep(120)
        return 0.0

    if __name__ == "__main__":
        path = os.path.join(sys.argv[1], "study.journal")
        study = optuna.create_study(study_name="wavetrain", storage=journal_storage(path))
        with exit_on_sigterm(), TrialWorkerPool(2) as pool:
            pool.optimize(study, path, sleepy, 2)
    """
)


class TestTrialWorkerPoolSignals(unittest.TestCase):

    def _run_and_signal(self, sig):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w", encoding="utf8") as handle:
                handle.write(_SIGNAL_SCRIPT)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), env.get("PYTHONPATH", "")])
            proc = subprocess.Popen([sys.executable, script, tmpdir], env=env)
            try:
                deadline = time.monotonic() + _HANG_SECONDS
                pids: list[int] = []
                while len(pids) < 2 and time.monotonic() < deadline:
                    time.sleep(0.5)
                    pids = [int(f[:-4]) for f in os.listdir(tmpdir) if f.endswith(".pid")]
                self.assertEqual(2, len(pids))
                proc.send_signal(sig)
                proc.wait(timeout=60)
                deadline = time.monotonic() + 30
                while any(_pid_alive(pid) for pid in pids) and time.monotonic() < deadline:
                    time.sleep(0.2)
                self.assertFalse([pid for pid in pids if _pid_alive(pid)])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            # Every trial was either finished or marked failed, unless the
            # parent was killed outright and could not clean up.
            if sig != signal.SIGKILL:
                study = optuna.load_study(
                    study_name="wavetrain",
                    storage=journal_storage(os.path.join(tmpdir, "study.journal")),
                )
                self.assertNotIn(
                    optuna.trial.TrialState.RUNNING, [t.state for t in study.trials]
                )

    def test_ctrl_c_stops_workers(self):
        self._run_and_signal(signal.SIGINT)

    def test_sigterm_stops_workers(self):
        self._run_and_signal(signal.SIGTERM)

    def test_parent_killed_stops_workers(self):
        self._run_and_signal(signal.SIGKILL)


class TestWorkerCount(unittest.TestCase):

    def test_positive_is_used_as_is(self):
        self.assertEqual(1, worker_count(1))
        self.assertEqual(3, worker_count(3))

    def test_zero_is_one(self):
        self.assertEqual(1, worker_count(0))

    def test_negative_counts_back_from_cpus(self):
        cpus = os.cpu_count() or 1
        self.assertEqual(cpus, worker_count(-1))
        self.assertEqual(max(1, cpus - 1), worker_count(-2))
        self.assertEqual(1, worker_count(-(cpus + 10)))


class TestSharedFrame(unittest.TestCase):

    def test_round_trip_is_read_only_and_cached(self):
        df = pd.DataFrame(
            {"a": np.arange(10.0), "b": pd.Categorical(list("xyxyxyxyxy"))},
            index=pd.date_range("2022-01-01", periods=10),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = SharedFrame.write(df, tmpdir)
            self.assertTrue(os.path.basename(shared.path).startswith(".wavetrainer-frame-"))
            loaded = shared.load()
            # The categorical codes come back as np.memmap rather than
            # ndarray, which assert_frame_equal treats as a difference.
            pd.testing.assert_frame_equal(df, loaded, check_categorical=False)
            self.assertEqual(list(df["b"].cat.categories), list(loaded["b"].cat.categories))
            self.assertEqual(df["b"].cat.codes.tolist(), loaded["b"].cat.codes.tolist())
            self.assertIsInstance(loaded["b"].array.codes, np.memmap)
            self.assertFalse(loaded["a"].to_numpy().flags.writeable)
            self.assertIs(loaded, shared.load())
            # Survives pickling as just its path.
            self.assertIs(loaded, pickle.loads(pickle.dumps(shared)).load())
            shared.remove()
            self.assertFalse(os.path.exists(shared.path))
            shared.remove()

    def test_loading_another_frame_drops_the_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = SharedFrame.write(pd.DataFrame({"a": [1.0]}), tmpdir)
            second = SharedFrame.write(pd.DataFrame({"a": [2.0]}), tmpdir)
            first.load()
            second.load()
            self.assertEqual([second.path], list(shared_frame._SHARED_FRAMES))


class TestExitOnSigterm(unittest.TestCase):

    def test_sigterm_raises_system_exit_and_restores_default(self):
        with self.assertRaises(SystemExit) as raised:
            with exit_on_sigterm():
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
        self.assertEqual(128 + signal.SIGTERM, raised.exception.code)
        self.assertIs(signal.SIG_DFL, signal.getsignal(signal.SIGTERM))

    def test_existing_handler_is_left_alone(self):
        received = []
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: received.append(signum))
        try:
            with exit_on_sigterm():
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.2)
            self.assertEqual([signal.SIGTERM], received)
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_off_main_thread_is_a_no_op(self):
        errors = []

        def _target():
            try:
                with exit_on_sigterm():
                    pass
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=_target)
        thread.start()
        thread.join()
        self.assertEqual([], errors)
        self.assertIs(signal.SIG_DFL, signal.getsignal(signal.SIGTERM))


class TestWorkerEnvironment(unittest.TestCase):

    def _pool(self, n_workers, gpus, visible=None):
        env = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
        with mock.patch.dict(os.environ, env), \
                mock.patch("torch.cuda.is_available", return_value=gpus > 0), \
                mock.patch("torch.cuda.device_count", return_value=gpus):
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            return TrialWorkerPool(n_workers)

    def test_no_gpu_leaves_environment_alone(self):
        pool = self._pool(2, 0)
        self.assertEqual({}, pool._worker_env(0))

    def test_too_few_gpus_runs_workers_on_cpu(self):
        pool = self._pool(3, 1)
        for index in range(3):
            self.assertEqual({"CUDA_VISIBLE_DEVICES": ""}, pool._worker_env(index))

    def test_enough_gpus_gives_each_worker_its_own(self):
        pool = self._pool(2, 4)
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "0"}, pool._worker_env(0))
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "1"}, pool._worker_env(1))

    def test_respects_already_visible_devices(self):
        pool = self._pool(2, 2, visible="5,7")
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "5"}, pool._worker_env(0))
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "7"}, pool._worker_env(1))

    def test_environ_overrides_are_restored(self):
        with mock.patch.dict(os.environ, {"WAVETRAINER_TEST_SET": "before"}):
            os.environ.pop("WAVETRAINER_TEST_UNSET", None)
            with trial_pool._environ({"WAVETRAINER_TEST_SET": "x", "WAVETRAINER_TEST_UNSET": "y"}):
                self.assertEqual("x", os.environ["WAVETRAINER_TEST_SET"])
                self.assertEqual("y", os.environ["WAVETRAINER_TEST_UNSET"])
            self.assertEqual("before", os.environ["WAVETRAINER_TEST_SET"])
            self.assertNotIn("WAVETRAINER_TEST_UNSET", os.environ)


class TestPoolHelpers(unittest.TestCase):

    def test_fail_orphaned_trials_only_touches_this_batch(self):
        study = optuna.create_study()
        old = study.ask()
        study.tell(study.ask(), 1.0)
        orphan = study.ask()
        with self.assertLogs(level="WARNING"):
            trial_pool._fail_orphaned_trials(study, first_number=2)
        states = {t.number: t.state for t in study.trials}
        self.assertEqual(optuna.trial.TrialState.RUNNING, states[old.number])
        self.assertEqual(optuna.trial.TrialState.COMPLETE, states[1])
        self.assertEqual(optuna.trial.TrialState.FAIL, states[orphan.number])

    def test_unpicklable_exception_is_replaced(self):
        class _Local(Exception):
            pass

        replaced = trial_pool._picklable_exception(_Local("nope"))
        self.assertIsInstance(replaced, RuntimeError)
        self.assertIn("_Local: nope", str(replaced))
        original = ValueError("fine")
        self.assertIs(original, trial_pool._picklable_exception(original))

    def test_zero_trials_starts_no_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "study.journal")
            study = optuna.create_study(study_name="wavetrain", storage=journal_storage(path))
            pool = TrialWorkerPool(2)
            pool.optimize(study, path, _quadratic, 0)
            self.assertEqual([None, None], pool._workers)
            pool.close()

    def test_expired_timeout_starts_no_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "study.journal")
            study = optuna.create_study(study_name="wavetrain", storage=journal_storage(path))
            pool = TrialWorkerPool(2)
            pool.optimize(study, path, _quadratic, 4, timeout=0.0)
            self.assertEqual([None, None], pool._workers)
            self.assertEqual(0, len(study.trials))
            pool.close()
