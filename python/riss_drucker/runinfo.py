"""Run records (README section 3, reproducibility): every run records the
model-spec hash, git commit, compiler flags, thread count, RNG seeds, SIMD
level, linalg backend and the environment probe."""
from __future__ import annotations

import datetime as _dt
import platform
import sys
from pathlib import Path

from . import _core
from ._version import __version__
from .config import runtime_config
from .io_utils import atomic_write_json

SCHEMA_VERSION = 1


def run_record(
    spec_hash: str | None = None,
    seeds: dict | None = None,
    extra: dict | None = None,
    env_probe: dict | None = None,
) -> dict:
    cfg = runtime_config()
    rec = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "package_version": __version__,
        "build": _core.build_info(),
        "cpu_features": _core.cpu_features(),
        "simd_level_active": _core.simd_level_active(),
        "linalg_backend_default": _core.default_linalg_backend(),
        "runtime_config": cfg.as_dict(),
        "spec_hash": spec_hash,
        "seeds": dict(seeds or {}),
        "python": sys.version,
        "platform": platform.platform(),
        # TODO(riss-drucker-v1): time-indexed parameters (Phase D) attach here.
        "extra": dict(extra or {}),
    }
    if env_probe is not None:
        rec["env_probe"] = env_probe
    return rec


def write_run_record(path: str | Path, **kwargs) -> Path:
    return atomic_write_json(path, run_record(**kwargs))
