"""Atomic file writes: results are written per block and must never be seen
half-written after a Kaggle session is killed (README 3.1, checkpointing)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_json(path: str | os.PathLike, obj) -> Path:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    return atomic_write_bytes(path, text.encode("utf-8"))
