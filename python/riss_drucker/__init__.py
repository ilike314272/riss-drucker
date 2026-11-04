"""riss_drucker: effective-Hamiltonian pipeline (Python orchestration layer).

The numerical core is the C++ extension ``riss_drucker._core``; this package
adds configuration, run records and (in later milestones) orchestration.
"""
from __future__ import annotations

from . import _core
from ._core import (
    BudgetError,
    CompiledModel,
    axpy,
    build_info,
    compile_asym_rotor,
    compile_torsion_rotor,
    cpu_features,
    csr_matvec,
    default_linalg_backend,
    dense_eigh_bytes,
    dot,
    eigh,
    inertia,
    linalg_backends,
    simd_level_active,
    simd_levels_available,
    sturm_count,
)
from ._version import __version__
from .config import RuntimeConfig, runtime_config
from .runinfo import run_record, write_run_record
from .solve import Solution, solve, solve_csr, solve_options
from . import angular, assign, peaks, posterior, robust, rotor, spectra, stats, synth
from .cache import CacheError, compile_cached
from .fit import FitResult, Observation, Prior, fit
from .results import fit_result_record, validate_result
from .spec import ModelSpec

if _core.__version__ != __version__:  # pragma: no cover - packaging error
    raise ImportError(
        f"riss_drucker version mismatch: python package {__version__}, "
        f"compiled core {_core.__version__}. Rebuild the wheel."
    )

__all__ = [
    "__version__",
    "BudgetError",
    "CacheError",
    "FitResult",
    "ModelSpec",
    "Observation",
    "Prior",
    "angular",
    "assign",
    "peaks",
    "posterior",
    "robust",
    "spectra",
    "stats",
    "compile_asym_rotor",
    "compile_cached",
    "fit",
    "fit_result_record",
    "rotor",
    "synth",
    "validate_result",
    "CompiledModel",
    "Solution",
    "compile_torsion_rotor",
    "inertia",
    "solve",
    "solve_csr",
    "solve_options",
    "sturm_count",
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
