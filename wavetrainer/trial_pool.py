"""Run Optuna trials in separate worker processes sharing one journal storage.

The parent splits a batch of trials into one task per worker. Each worker
loads the study from the shared journal file, runs its share with its own
copy of the sampler, and reports back how many trials it ran. Trial results
reach the parent through the journal, not through the task's reply.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments
import contextlib
import dataclasses
import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import threading
import time
import traceback
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Iterator, Sequence

import optuna
import torch
import tqdm
from optuna.storages.journal import JournalFileBackend, JournalStorage

# How often the parent wakes while waiting on workers, to refresh progress.
_POLL_SECONDS = 1.0
# How long close() waits for idle workers to exit before terminating them.
_CLOSE_GRACE_SECONDS = 5.0
# How many times a batch hands a crashed worker's unused trials to a fresh
# worker before giving up on them. Bounds the cost of a worker that crashes
# every time, e.g. while importing a native library.
_MAX_CRASH_RETRIES = 3
# Stands in for a result from a worker that died before replying.
_CRASHED = object()


def journal_storage(path: str) -> JournalStorage:
    """Open the Optuna journal file storage at the given path."""
    return JournalStorage(JournalFileBackend(path))


def worker_count(n_jobs: int) -> int:
    """Convert an n_jobs value to a worker count, using joblib's convention."""
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


@dataclasses.dataclass
class _OptimizeTask:
    """One worker's share of a batch: run n_trials of objective on the study."""

    storage_path: str
    study_name: str
    sampler: optuna.samplers.BaseSampler
    objective: Callable[[optuna.Trial], float]
    n_trials: int
    timeout: float | None
    callbacks: Sequence[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]


def _run_task(task: _OptimizeTask) -> int:
    """Run a task in this worker and return how many trials it ran."""
    # The sampler arrives as a copy of the parent's, so give each worker its
    # own random stream or they would all propose the same parameters.
    task.sampler.reseed_rng()
    study = optuna.load_study(
        study_name=task.study_name,
        storage=journal_storage(task.storage_path),
        sampler=task.sampler,
    )
    # Counted so the parent can tell a stopped worker (fewer than its share)
    # from one that ran everything it was given.
    ran = 0

    def _count(_study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        nonlocal ran
        ran += 1

    study.optimize(
        task.objective,
        n_trials=task.n_trials,
        timeout=task.timeout,
        callbacks=[_count, *task.callbacks],
        n_jobs=1,
        show_progress_bar=False,
    )
    return ran


def _exit_with_parent(parent: BaseProcess) -> None:
    """Wait for the parent process to end, then end this worker too."""
    parent.join()
    os._exit(1)


def _picklable_exception(exc: BaseException) -> BaseException:
    """Return exc if it can be sent to the parent, else a RuntimeError copy."""
    try:
        pickle.dumps(exc)
        return exc
    except Exception:  # pylint: disable=broad-exception-caught
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def _worker_main(conn: multiprocessing.connection.Connection) -> None:
    """A worker's loop: receive a task, run it, reply, until told to stop.

    Replies are ("ok", trials_run) or ("error", exception, traceback). A None
    task, or the parent closing its end of the pipe, ends the loop.
    """
    # Ctrl-C reaches the whole process group; let the parent decide what to
    # do with it, rather than every worker dying mid-write on its own.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Exit if the parent dies (even by SIGKILL) so no worker is orphaned.
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            return
        if task is None:
            return
        try:
            result: tuple[Any, ...] = ("ok", _run_task(task))
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            result = ("error", _picklable_exception(exc), traceback.format_exc())
        try:
            conn.send(result)
        except OSError:
            return


@contextlib.contextmanager
def _environ(overrides: dict[str, str]) -> Iterator[None]:
    """Set environment variables temporarily, e.g. for a process to inherit."""
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def exit_on_sigterm() -> Iterator[None]:
    """Turn SIGTERM into SystemExit while active, so cleanup code runs.

    Only takes effect on the main thread, and only when SIGTERM still has its
    default handler, so an application's own handler is left alone.
    """
    if (
        threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    ):
        yield
        return

    def _raise(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


@dataclasses.dataclass
class _Worker:
    """A worker process and the parent's end of the pipe to it."""

    process: BaseProcess
    conn: multiprocessing.connection.Connection


def _fail_orphaned_trials(study: optuna.Study, first_number: int) -> None:
    # Called only when no worker is running a trial for this study, so any of
    # this batch's trials still RUNNING belonged to a worker that died or was
    # terminated.
    for trial in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        if trial.number < first_number:
            continue
        logging.warning(
            "Marking trial %d from a stopped worker as failed.", trial.number
        )
        with contextlib.suppress(RuntimeError):
            study._storage.set_trial_state_values(  # pylint: disable=protected-access
                trial._trial_id,  # pylint: disable=protected-access
                state=optuna.trial.TrialState.FAIL,
            )


class TrialWorkerPool:
    """A fixed set of spawned worker processes that run Optuna trials.

    Unlike a ProcessPoolExecutor (or loky), one worker dying, e.g. from a
    segfault in native code, does not break the pool or kill its siblings:
    the death is logged, its unused trials go to a replacement worker, and
    the other workers carry on. Workers are long-lived so that importing the
    native libraries and loading the shared frame happen once per worker
    rather than once per batch of trials.

    GPUs: a worker gets a GPU of its own (via CUDA_VISIBLE_DEVICES) when
    there are at least as many visible GPUs as workers. Otherwise workers run
    on the CPU, as several processes sharing one GPU contend for its memory.
    The parent process keeps whatever GPU it already sees.
    """

    def __init__(self, n_workers: int):
        self._ctx = multiprocessing.get_context("spawn")
        self._workers: list[_Worker | None] = [None] * n_workers
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        # None: no GPUs, leave workers' environment alone. Empty: hide the
        # GPUs from workers. Otherwise: the device id for each worker.
        self._gpu_ids: list[str] | None = None
        if gpus >= n_workers:
            self._gpu_ids = (
                visible.split(",")[:gpus] if visible else [str(i) for i in range(gpus)]
            )
        elif gpus > 0:
            self._gpu_ids = []

    @property
    def size(self) -> int:
        """The number of workers in the pool."""
        return len(self._workers)

    def _worker_env(self, index: int) -> dict[str, str]:
        """The environment overrides for the worker in this slot."""
        if self._gpu_ids is None:
            return {}
        if not self._gpu_ids:
            return {"CUDA_VISIBLE_DEVICES": ""}
        return {"CUDA_VISIBLE_DEVICES": self._gpu_ids[index]}

    def _worker(self, index: int) -> _Worker:
        """The worker in this slot, spawning one if it is empty or dead."""
        worker = self._workers[index]
        if worker is not None and worker.process.is_alive():
            return worker
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn,),
            name=f"wavetrainer-trial-worker-{index}",
        )
        # A spawned process takes its environment from the parent at start.
        with _environ(self._worker_env(index)):
            process.start()
        # Only the worker holds the child end, so the parent sees EOF if it dies.
        child_conn.close()
        worker = _Worker(process=process, conn=parent_conn)
        self._workers[index] = worker
        return worker

    def _crashed(self, index: int, worker: _Worker) -> object:
        """Log a dead worker and empty its slot, so it is respawned when needed."""
        worker.process.join(timeout=_CLOSE_GRACE_SECONDS)
        logging.error(
            "Trial worker %d (pid %s) died with exit code %s; continuing with the "
            "remaining workers.",
            index,
            worker.process.pid,
            worker.process.exitcode,
        )
        worker.conn.close()
        self._workers[index] = None
        return _CRASHED

    def _run_round(
        self, tasks: list[_OptimizeTask], on_poll: Callable[[], None]
    ) -> list[Any]:
        """Send task i to worker i and wait for every reply or death.

        Returns each worker's reply, or _CRASHED for one that died. A trial's
        exception is re-raised here, with the worker's traceback attached.
        """
        results: list[Any] = [None] * len(tasks)
        active: dict[int, _Worker] = {}
        for index, task in enumerate(tasks):
            worker = self._worker(index)
            try:
                worker.conn.send(task)
            except OSError:
                results[index] = self._crashed(index, worker)
                continue
            active[index] = worker
        while active:
            # Wake on a reply, a worker exiting, or the poll interval.
            multiprocessing.connection.wait(
                [w.conn for w in active.values()]
                + [w.process.sentinel for w in active.values()],
                timeout=_POLL_SECONDS,
            )
            on_poll()
            for index, worker in list(active.items()):
                result: Any
                if worker.conn.poll():
                    # Readable means a reply, or EOF if the worker died.
                    try:
                        result = worker.conn.recv()
                    except (EOFError, OSError):
                        result = self._crashed(index, worker)
                elif not worker.process.is_alive():
                    result = self._crashed(index, worker)
                else:
                    continue
                del active[index]
                if result is not _CRASHED and result[0] == "error":
                    exc = result[1]
                    exc.add_note(f"Raised in trial worker {index}:\n{result[2]}")
                    raise exc
                results[index] = result
        return results

    def optimize(
        self,
        study: optuna.Study,
        storage_path: str,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: Sequence[
            Callable[[optuna.Study, optuna.trial.FrozenTrial], None]
        ] = (),
        show_progress_bar: bool = False,
    ) -> None:
        """Run n_trials of the objective across the workers.

        Mirrors study.optimize: the trials are split across the workers so
        the total stays n_trials, timeout bounds the whole batch, and a
        callback stopping the study stops every worker at its next trial
        boundary (each callback sees the shared storage, see
        ThresholdCallback). An exception raised by a trial is re-raised here.
        """
        # pylint: disable=too-many-locals
        deadline = None if timeout is None else time.monotonic() + timeout
        # Trial numbers are sequential, so this batch's trials are the ones
        # numbered from here on.
        first_number = len(study.get_trials(deepcopy=False))
        progress = (
            tqdm.tqdm(total=n_trials, desc="Trials") if show_progress_bar else None
        )

        def _done() -> int:
            return len(study.get_trials(deepcopy=False)) - first_number

        def _on_poll() -> None:
            if progress is not None:
                progress.update(max(_done() - progress.n, 0))

        retries = 0
        remaining = n_trials
        try:
            while remaining > 0:
                time_left = None
                if deadline is not None:
                    time_left = deadline - time.monotonic()
                    if time_left <= 0:
                        break
                # Split the remaining trials as evenly as possible.
                n_workers = min(self.size, remaining)
                shares = [
                    remaining // n_workers + (1 if i < remaining % n_workers else 0)
                    for i in range(n_workers)
                ]
                tasks = [
                    _OptimizeTask(
                        storage_path=storage_path,
                        study_name=study.study_name,
                        sampler=study.sampler,
                        objective=objective,
                        n_trials=share,
                        timeout=time_left,
                        callbacks=list(callbacks),
                    )
                    for share in shares
                ]
                try:
                    results = self._run_round(tasks, _on_poll)
                except BaseException:
                    self.terminate()
                    raise
                finally:
                    # No worker is mid-trial now, whether the round ended
                    # normally or the workers were just terminated.
                    _fail_orphaned_trials(study, first_number)
                crashed = sum(1 for r in results if r is _CRASHED)
                # A worker that returned having run fewer trials than its
                # share was stopped (by a callback or the timeout), which
                # ends the batch, crashes or not.
                stopped = any(
                    r is not _CRASHED and r[1] < share
                    for r, share in zip(results, shares)
                )
                if not crashed or stopped:
                    break
                # Rerun what the crashed workers did not get to. The trial a
                # worker died in counts as run (it is marked failed).
                remaining = n_trials - _done()
                if remaining <= 0:
                    break
                retries += 1
                if retries > _MAX_CRASH_RETRIES:
                    logging.error(
                        "Giving up on %d trial(s) after %d worker crashes.",
                        remaining,
                        retries,
                    )
                    break
                logging.warning(
                    "%d worker(s) crashed; rerunning their %d unused trial(s).",
                    crashed,
                    remaining,
                )
        finally:
            _on_poll()
            if progress is not None:
                progress.close()

    def terminate(self) -> None:
        """Kill every worker immediately."""
        for index, worker in enumerate(self._workers):
            if worker is None:
                continue
            if worker.process.is_alive():
                worker.process.terminate()
            worker.process.join(timeout=_CLOSE_GRACE_SECONDS)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
            worker.conn.close()
            self._workers[index] = None

    def close(self) -> None:
        """Ask idle workers to exit, then terminate any that do not."""
        for worker in self._workers:
            if worker is not None:
                with contextlib.suppress(OSError):
                    worker.conn.send(None)
        deadline = time.monotonic() + _CLOSE_GRACE_SECONDS
        for worker in self._workers:
            if worker is not None:
                worker.process.join(timeout=max(deadline - time.monotonic(), 0))
        self.terminate()

    def __enter__(self) -> "TrialWorkerPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
