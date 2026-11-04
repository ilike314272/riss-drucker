"""riss_drucker: effective-Hamiltonian pipeline (Python orchestration layer).

The numerical core is the C++ extension ``riss_drucker._core``; this package
adds configuration, run records and (in later milestones) orchestration.
"""
from __future__ import annotations

from . import _core
from ._core import (
    axpy,
    build_info,
    cpu_features,
    csr_matvec,
    default_linalg_backend,
    dense_eigh_bytes,
    dot,
    eigh,
    linalg_backends,
    simd_level_active,
    simd_levels_available,
)
from ._version import __version__
from .config import RuntimeConfig, runtime_config
from .runinfo import run_record, write_run_record

if _core.__version__ != __version__:  # pragma: no cover - packaging error
    raise ImportError(
        f"riss_drucker version mismatch: python package {__version__}, "
        f"compiled core {_core.__version__}. Rebuild the wheel."
    )

__all__ = [
    "__version__",
    "axpy",
    "build_info",
    "cpu_features",
    "csr_matvec",
    "default_linalg_backend",
    "dense_eigh_bytes",
    "dot",
    "eigh",
    "linalg_backends",
    "simd_level_active",
    "simd_levels_available",
    "RuntimeConfig",
    "runtime_config",
    "run_record",
    "write_run_record",
]
