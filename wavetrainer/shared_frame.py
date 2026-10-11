"""A DataFrame shared with worker processes through a memory-mapped file."""

import contextlib
import os
import tempfile

import joblib  # type: ignore
import pandas as pd

# The SharedFrame loaded in this process, keyed by its file path.
_SHARED_FRAMES: dict[str, pd.DataFrame] = {}


class SharedFrame:
    """A DataFrame written to disk once and memory-mapped by each worker.

    joblib stores the frame's numpy blocks so that loading with mmap_mode="r"
    maps them instead of copying them, so the workers share the page cache
    rather than each holding a private copy of a very wide frame. Only the
    path is pickled into tasks.
    """

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def write(cls, df: pd.DataFrame, folder: str) -> "SharedFrame":
        """Dump the frame to a hidden file in the folder."""
        fd, path = tempfile.mkstemp(
            dir=folder, prefix=".wavetrainer-frame-", suffix=".joblib"
        )
        os.close(fd)
        joblib.dump(df, path)
        return cls(path)

    def load(self) -> pd.DataFrame:
        """Load the frame, memory-mapped and read-only, caching it per process.

        The arrays are read-only (np.memmap), so copy rows before changing them.
        """
        df = _SHARED_FRAMES.get(self.path)
        if df is None:
            # A worker outlives one fit, so drop any frame from an earlier one.
            _SHARED_FRAMES.clear()
            df = joblib.load(self.path, mmap_mode="r")
            _SHARED_FRAMES[self.path] = df
        return df

    def remove(self) -> None:
        """Delete the file backing the frame."""
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.path)
