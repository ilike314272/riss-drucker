"""From a torsion potential V(phi) (with uncertainty) to torsional levels.

V(phi) = sum_L V_L (1 - cos L phi) / 2 is fitted by least squares on a grid
(cosine series: potentials symmetric under phi -> -phi). Posterior samples of
the surface give samples of V_L and hence of the levels (propagated variance,
README M5)."""
from __future__ import annotations

import numpy as np

from ..solve import solve


def fit_cosine_series(phi, V, orders) -> dict:
    """Least-squares V_L with a free constant (the energy zero does not matter;
    (1 - cos L phi)/2 terms all vanish at phi = 0, where V need not be minimal)."""
    phi = np.asarray(phi, float)
    A = np.column_stack([np.ones_like(phi)] + [(1 - np.cos(L * phi)) / 2 for L in orders])
    coef, *_ = np.linalg.lstsq(A, np.asarray(V, float), rcond=None)
    return {f"V{L}": float(c) for L, c in zip(orders, coef[1:])}


def levels(model, F: float, vcoef: dict, nlowest: int = 4):
    th = {"F": F, **{k: vcoef.get(k, 0.0) for k in model.param_names if k.startswith("V")}}
    sol = solve(model, th, nlowest=nlowest, method="dense")
    if not sol.certified:
        raise RuntimeError("torsion solve not certified")
    return np.sort(sol.energies())


def propagate(model, F: float, phi, V_samples, orders, nlowest: int = 4, relative: bool = True) -> dict:
    """Levels for each potential sample; returns mean, sd and all samples (relative to the ground level)."""
    out = []
    for V in V_samples:
        c = fit_cosine_series(phi, V - np.min(V), orders)
        e = levels(model, F, c, nlowest)
        out.append(e - e[0] if relative else e)
    out = np.array(out)
    return {"mean": out.mean(0), "sd": out.std(0, ddof=1) if len(out) > 1 else np.zeros(out.shape[1]), "samples": out}
