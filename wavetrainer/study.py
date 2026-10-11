"""Create or resume the Optuna study for one target column."""

import logging
import os
import pickle

import optuna

from .drift_mixed_sampler import DriftMixedSampler
from .trial_pool import journal_storage, worker_count

# The pickled sampler, saved after training so a resumed study keeps it.
SAMPLER_FILENAME = "sampler.pkl"
_STUDYDB_FILENAME = "study.db"
_STUDYJOURNAL_FILENAME = "study.journal"
_STUDY_NAME = "wavetrain"


def _provide_storage(
    folder: str, column: str, n_jobs: int
) -> tuple[str | optuna.storages.BaseStorage, str | None]:
    """Find the study storage for a column, and its journal path if it has one.

    Sequential training keeps the SQLite database it always used. SQLite
    fails with "database is locked" when several processes write to it, so
    parallel training uses a journal file instead, copying an existing
    SQLite study into it the first time. Once a column has a journal it
    is always used, so a later sequential run never resumes from a stale
    SQLite database. The SQLite file and its sampler are kept as a backup.
    """
    storage_name = f"sqlite:///{folder}/{column}/{_STUDYDB_FILENAME}"
    journal_file = os.path.join(folder, column, _STUDYJOURNAL_FILENAME)
    # Sequential, and never run in parallel: SQLite, exactly as before.
    if worker_count(n_jobs) <= 1 and not os.path.exists(journal_file):
        return storage_name, None
    # First parallel run on a column with an SQLite study: copy it over.
    if not os.path.exists(journal_file) and os.path.exists(
        os.path.join(folder, column, _STUDYDB_FILENAME)
    ):
        logging.info("Copying %s into %s", storage_name, journal_file)
        # Copy into a temporary file and rename it into place at the end, so
        # an interrupted copy never leaves a partial journal that looks done.
        migrating_file = journal_file + ".migrating"
        if os.path.exists(migrating_file):
            os.remove(migrating_file)
        optuna.copy_study(
            from_study_name=_STUDY_NAME,
            from_storage=storage_name,
            to_storage=journal_storage(migrating_file),
        )
        # The saved sampler caches the SQLite study's id (in TPE's search
        # space) and fails on the copy, which has a different id. Set it
        # aside with the database; the copy starts from a fresh sampler.
        sampler_file = os.path.join(folder, column, SAMPLER_FILENAME)
        if os.path.exists(sampler_file):
            os.replace(sampler_file, sampler_file + ".sqlite")
        os.replace(migrating_file, journal_file)
    return journal_storage(journal_file), journal_file


def provide_study(
    folder: str, column: str, n_jobs: int, random_mix_ratio: float
) -> tuple[optuna.Study, str | None]:
    """Create or load a column's study, and return its journal path if it has one.

    The journal path is what worker processes need to open the same study;
    None means the study is in SQLite and its trials must run in this process.
    """
    storage_name, journal_file = _provide_storage(folder, column, n_jobs)
    sampler_file = os.path.join(folder, column, SAMPLER_FILENAME)
    restored_sampler = None
    if os.path.exists(sampler_file):
        with open(sampler_file, "rb") as handle:
            restored_sampler = pickle.load(handle)

    # A new study, or one just migrated to a journal, starts a fresh sampler.
    if restored_sampler is None:
        restored_sampler = DriftMixedSampler(random_mix_ratio=random_mix_ratio)

    study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=storage_name,
        load_if_exists=True,
        sampler=restored_sampler,
        directions=[
            # optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ],
    )
    return study, journal_file
