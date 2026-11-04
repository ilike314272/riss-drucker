"""Calculated-vs-empirical error prior (README 6.4).

Where a database provides both computed and empirical (e.g. MARVEL) energies
for the same states, the differences E_emp - E_calc, grouped by band, J bin and
symmetry, give an empirical distribution of the computed error. A smooth model
sigma(J) = sqrt(a^2 + (b * J(J+1))^2) fitted to the robust spread is the default
model-error hyperparameter prior for ab initio inputs (M5).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _robust_std(x):
    x = np.asarray(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def summarize(e_emp, e_calc, J, *, band: Sequence | None = None, symmetry: Sequence | None = None,
              j_bins: Sequence[float] = (0, 5, 10, 20, 40, 80, 1e9)) -> dict:
    e_emp, e_calc, J = map(lambda a: np.asarray(a, dtype=float), (e_emp, e_calc, J))
    d = e_emp - e_calc
    band = np.asarray(band if band is not None else ["all"] * len(d), dtype=object)
    sym = np.asarray(symmetry if symmetry is not None else ["all"] * len(d), dtype=object)
    groups = {}
    jb = np.digitize(J, j_bins) - 1
    for key in sorted(set(zip(band, sym, jb))):
        m = (band == key[0]) & (sym == key[1]) & (jb == key[2])
        x = d[m]
        groups[f"{key[0]}|{key[1]}|J[{j_bins[key[2]]:g},{j_bins[key[2] + 1]:g})"] = {
            "n": int(m.sum()), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "robust_std": _robust_std(x), "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
        }
    return {"n": int(len(d)), "mean": float(d.mean()), "robust_std": _robust_std(d), "groups": groups,
            "sigma_model": fit_sigma_model(d, J)}


def fit_sigma_model(diff, J) -> dict:
    """sigma(J) = sqrt(a^2 + (b J(J+1))^2) by least squares on |diff| (robust scale per J bin)."""
    diff, J = np.asarray(diff, dtype=float), np.asarray(J, dtype=float)
    x = J * (J + 1)
    uj, counts = np.unique(J, return_counts=True)
    if len(uj) >= 3 and counts.min() >= 10:
        # One robust spread per J value: no mixing of different sigmas.
        xm = uj * (uj + 1)
        sm = np.array([_robust_std(diff[J == j]) for j in uj])
    else:
        order = np.argsort(x)
        xs, ds = x[order], diff[order]
        chunks = np.array_split(np.arange(len(ds)), max(1, min(10, len(ds) // 20)))
        xm = np.array([xs[c].mean() for c in chunks])
        sm = np.array([_robust_std(ds[c]) for c in chunks])
    # Relative (1/sigma^2-weighted) least squares, so small-J spreads count.
    w = 1.0 / np.maximum(sm, 1e-300) ** 2
    A = np.vstack([np.ones_like(xm), xm**2]).T * w[:, None]
    coef, *_ = np.linalg.lstsq(A, sm**2 * w, rcond=None)
    a2, b2 = max(coef[0], 0.0), max(coef[1], 0.0)
    return {"a": math.sqrt(a2), "b": math.sqrt(b2), "form": "sqrt(a^2 + (b*J(J+1))^2)"}


def sigma_at(model: dict, J) -> np.ndarray:
    x = np.asarray(J, dtype=float)
    return np.sqrt(model["a"] ** 2 + (model["b"] * x * (x + 1)) ** 2)
