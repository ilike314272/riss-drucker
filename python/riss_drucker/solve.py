"""Certified block solves from Python (README 5.3).

    model = rd.compile_torsion_rotor(n_fold=3, v_orders=[1], m_max=40)
    sol = rd.solve(model, {"F": 5.3, "V3": 350.0}, nlowest=6)
    sol.certified, sol.levels()[:5]

Every block result carries its certificates. A failed certificate never raises:
it marks the block (and the solution) uncertified and is listed by
``Solution.failed_certificates()`` (README 10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import _core
from .config import runtime_config


def _theta_vector(model, theta) -> list[float]:
    names = list(model.param_names)
    if isinstance(theta, Mapping):
        missing = [n for n in names if n not in theta]
        extra = [k for k in theta if k not in names]
        if missing or extra:
            raise ValueError(f"theta must give exactly {names}; missing {missing}, unknown {extra}")
        return [float(theta[n]) for n in names]
    vec = [float(x) for x in theta]
    if len(vec) != len(names):
        raise ValueError(f"theta must have {len(names)} entries {names}, got {len(vec)}")
    return vec


def solve_options(
    *,
    nlowest: int | None = None,
    window: tuple[float, float] | None = None,
    method: str = "auto",
    backend: str | None = None,
    memory_budget: float | None = None,
    keep_vectors: bool = False,
    lanczos_max_basis: int | None = None,
    lanczos_max_runs: int | None = None,
    lanczos_tol: float | None = None,
    dense_max_dim: int | None = None,
    **extra,
) -> "_core.SolveOptions":
    if (nlowest is None) == (window is None):
        raise ValueError("give exactly one of nlowest=k or window=(lo, hi)")
    o = _core.SolveOptions()
    if nlowest is not None:
        if nlowest < 0:
            raise ValueError("nlowest must be >= 0")
        o.nlowest = int(nlowest)
    else:
        lo, hi = window
        if not lo < hi:
            raise ValueError("window must satisfy lo < hi")
        o.window_lo, o.window_hi = float(lo), float(hi)
    o.method = method
    o.backend = backend or _core.default_linalg_backend()
    budget = runtime_config().memory_budget_bytes if memory_budget is None else memory_budget
    o.memory_budget_bytes = float(budget or 0.0)
    o.keep_vectors = bool(keep_vectors)
    if dense_max_dim is not None:
        o.dense_max_dim = int(dense_max_dim)
    lz = o.lanczos
    if lanczos_max_basis is not None:
        lz.max_basis = int(lanczos_max_basis)
    if lanczos_max_runs is not None:
        lz.max_runs = int(lanczos_max_runs)
    if lanczos_tol is not None:
        lz.tol = float(lanczos_tol)
    o.lanczos = lz
    for k, v in extra.items():
        if not hasattr(o, k):
            raise TypeError(f"unknown solve option {k!r}")
        setattr(o, k, v)
    return o


@dataclass
class Solution:
    model_name: str
    spec_json: str
    param_names: list[str]
    theta: list[float]
    seed: int
    threads: int
    blocks: list[dict]
    results: list[dict]
    options: dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        return all(r["certified"] for r in self.results)

    def failed_certificates(self) -> list[dict]:
        out = []
        for b, r in zip(self.blocks, self.results):
            for c in r["certificates"]:
                if not c["passed"]:
                    out.append({"block": b["key"], **c})
        return out

    def levels(self) -> list[dict]:
        """All returned levels, sorted by energy, with block and assignment."""
        rows = []
        for b, r in zip(self.blocks, self.results):
            assign = r.get("assignment", [None] * len(r["eigenvalues"]))
            weight = r.get("assignment_weight", [None] * len(r["eigenvalues"]))
            for i, e in enumerate(r["eigenvalues"]):
                rows.append({
                    "energy": float(e), "block": b["key"], "J": b["two_j"] / 2, "irrep": b["irrep"],
                    "sub": b["sub"], "index": i, "assignment": assign[i], "weight": weight[i],
                    "residual": float(r["residuals"][i]), "certified": r["certified"],
                })
        rows.sort(key=lambda row: row["energy"])
        return rows

    def energies(self) -> np.ndarray:
        return np.array([row["energy"] for row in self.levels()])

    def summary(self) -> dict:
        """Compact, JSON-serializable record of the solve (no vectors)."""
        return {
            "model": self.model_name, "spec_json": self.spec_json,
            "theta": dict(zip(self.param_names, self.theta)), "seed": self.seed,
            "threads": self.threads, "options": self.options, "certified": self.certified,
            "failed_certificates": self.failed_certificates(),
            "blocks": [
                {"key": b["key"], "dim": b["dim"], "method": r["method"],
                 "eigenvalues": [float(x) for x in r["eigenvalues"]],
                 "certified": r["certified"], "certificates": r["certificates"],
                 "window": list(r["window"]), "gap_ratio_mean": r["gap_ratio_mean"]}
                for b, r in zip(self.blocks, self.results)
            ],
        }


def solve(model, theta, *, threads: int | None = None, seed: int = 0, warm_start: "Solution | None" = None,
          **options) -> Solution:
    """Solve every block of a compiled model at parameters ``theta``.

    theta: mapping name -> value (exactly ``model.param_names``) or a sequence.
    options: see ``solve_options`` (nlowest= or window=, method=, backend=, ...).
    warm_start: a previous Solution of the same model solved with keep_vectors=True.
    """
    vec = _theta_vector(model, theta)
    opts = solve_options(**options)
    nthreads = runtime_config().threads if threads is None else int(threads)
    warm = None
    if warm_start is not None:
        if warm_start.spec_json != model.spec_json:
            raise ValueError("warm_start comes from a different model structure")
        if any("eigenvectors" not in r for r in warm_start.results):
            raise ValueError("warm_start needs a Solution computed with keep_vectors=True")
        warm = [np.asarray(r["eigenvectors"]) for r in warm_start.results]
    results = model.solve(vec, opts, nthreads, int(seed), warm)
    recorded = {k: v for k, v in options.items()}
    recorded.setdefault("method", "auto")
    return Solution(model.name, model.spec_json, list(model.param_names), vec, int(seed), nthreads,
                    model.blocks, results, recorded)


def solve_csr(indptr, indices, data, *, warm: np.ndarray | None = None, **options) -> dict:
    """Certified solve of one symmetric CSR block (scipy-compatible arrays)."""
    return _core.solve_csr(np.asarray(indptr), np.asarray(indices), np.asarray(data),
                           solve_options(**options), warm)
