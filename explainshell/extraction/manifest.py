"""Structured manifest for batch extraction runs.

Written atomically after each batch completes, so it survives crashes.
Thread-safe for parallel batch mode (jobs > 1).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Literal, Protocol

from pydantic import BaseModel


class BatchManifestEntry(BaseModel):
    """One batch's record in the manifest."""

    batch_idx: int
    batch_id: str | None
    status: Literal["submitted", "completed", "failed"]
    error: str | None
    files: list[str]


class BatchManifest(BaseModel):
    """Top-level manifest schema — used for reading and validation."""

    version: Literal[1]
    model: str
    batch_size: int
    total_batches: int | None
    batches: list[BatchManifestEntry]


class BatchManifestWriter(Protocol):
    """Protocol for batch manifest writers used by the runner."""

    def set_total_batches(self, n: int) -> None: ...

    def record_batch(
        self,
        batch_idx: int,
        batch_id: str | None,
        status: Literal["submitted", "completed", "failed"],
        files: list[str],
        error: str | None = None,
    ) -> None: ...


class FileBatchManifestWriter:
    """Thread-safe manifest writer for batch extraction runs.

    Records batch outcomes incrementally and flushes to disk after each update.
    """

    def __init__(self, path: str, model: str, batch_size: int) -> None:
        self._path = path
        self._data = BatchManifest(
            version=1,
            model=model,
            batch_size=batch_size,
            total_batches=None,
            batches=[],
        )
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def set_total_batches(self, n: int) -> None:
        """Set the total number of batches (called once after grouping)."""
        self._data.total_batches = n

    def record_batch(
        self,
        batch_idx: int,
        batch_id: str | None,
        status: Literal["submitted", "completed", "failed"],
        files: list[str],
        error: str | None = None,
    ) -> None:
        """Record a batch outcome and flush to disk.

        If an entry with the same ``batch_idx`` already exists (e.g. from an
        earlier "submitted" record), it is replaced in-place.
        """
        entry = BatchManifestEntry(
            batch_idx=batch_idx,
            batch_id=batch_id,
            status=status,
            error=error,
            files=files,
        )
        with self._lock:
            # Replace existing entry for same batch_idx, or append.
            for i, existing in enumerate(self._data.batches):
                if existing.batch_idx == batch_idx:
                    self._data.batches[i] = entry
                    break
            else:
                self._data.batches.append(entry)
            self._flush()

    def to_dict(self) -> dict:
        """Return manifest data as a plain dict for embedding in reports."""
        with self._lock:
            return self._data.model_dump()

    def _flush(self) -> None:
        """Atomic write: dump to .tmp, then os.replace."""
        data = self._data.model_dump()
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, self._path)
