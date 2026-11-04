"""Synthetic data (README 5.7), M2 part: assigned levels from a known theta,
with Gaussian noise and ground truth kept with every observation. Selection
rules, intensities and level-sequence generators come in M3."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .fit import Observation
from .solve import solve


def synthesize_levels(model, theta: Mapping[str, float], *, noise: float = 0.0, seed: int = 0,
                      nlowest: int | None = None, block_filter: Callable[[dict], bool] | None = None,
                      labels: Mapping[str, list[str]] | None = None, threads: int | None = None) -> list[Observation]:
    """All (or the lowest `nlowest`) levels of the selected blocks as observations.

    `energy` = true level + N(0, noise^2); `sigma` = noise (or 1e-12 if noise == 0).
    """
    rng = np.random.default_rng(seed)
    n = nlowest if nlowest is not None else max(b["dim"] for b in model.blocks)
    sol = solve(model, theta, nlowest=n, method="dense", threads=threads)
    if not sol.certified:
        raise RuntimeError(f"synthetic solve not certified: {sol.failed_certificates()[:3]}")
    obs = []
    for b, r in zip(sol.blocks, sol.results):
        if block_filter is not None and not block_filter(b):
            continue
        labs = labels.get(b["key"]) if labels else None
        for i, e in enumerate(r["eigenvalues"]):
            obs.append(Observation(block=b["key"], index=i, energy=float(e + noise * rng.standard_normal()),
                                   sigma=noise if noise > 0 else 1e-12,
                                   label=labs[i] if labs else None, true_energy=float(e)))
    return obs


def perturb_theta(theta: Mapping[str, float], rel: float, seed: int = 0) -> dict[str, float]:
    """theta_k * (1 + rel * N(0,1)), a starting point for recovery tests."""
    rng = np.random.default_rng(seed)
    return {k: float(v * (1 + rel * rng.standard_normal())) for k, v in theta.items()}
