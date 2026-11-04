"""Rotational line strengths and intensities for the asymmetric rotor (M3).

Line strength (summed over M and space-fixed components) between levels
|f> = sum c'_K' |J' K'> and |i> = sum c_K |J K>:

    S(f, i) = (2J+1)(2J'+1) | sum_{q,K,K'} mu_q c'_K' c_K (-1)^K' (J' 1 J; -K' q K) |^2

with molecule-fixed spherical components mu_0 = mu_z, mu_{+-1} = -+(mu_x +- i mu_y)/sqrt 2
in the rotor's representation (x, y, z). Certificates: sum_f S(f, i) = (2J+1)|mu|^2
for every level i whose J+1 manifold is in the model, and the C2 selection rules
(a-, b-, c-type change the (Ka, Kc) parities by (0,1), (1,1), (1,0)).

Intensities (relative absorption):  I = g_ns S nu exp(-c2 E''/T)(1 - exp(-c2 nu/T)) / Q(T),
c2 = hc/k in cm K, energies relative to the lowest level of the model, Q from
the model's own levels (truncated at j_max; reported).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from . import _core
from .rotor import _AXES, asym_top_labels

C2 = 1.438776877  # cm K (second radiation constant, CODATA 2018)


@dataclass
class Line:
    lower: tuple[str, int]      # (block key, index)
    upper: tuple[str, int]
    lower_label: str
    upper_label: str
    lower_energy: float
    upper_energy: float
    wavenumber: float
    strength: float             # S, units of |mu|^2
    j_lower: int
    j_upper: int
    intensity: float | None = None
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def _tables(Jp: int, J: int) -> np.ndarray:
    """T[q+1][K'+J', K+J] = (-1)^K' (J' 1 J; -K' q K) for q = -1, 0, 1."""
    T = np.zeros((3, 2 * Jp + 1, 2 * J + 1))
    for K in range(-J, J + 1):
        for q in (-1, 0, 1):
            Kp = K + q
            if abs(Kp) > Jp:
                continue
            T[q + 1, Kp + Jp, K + J] = (-1) ** (Kp % 2) * _core.wigner3j_2(2 * Jp, 2, 2 * J, -2 * Kp, 2 * q, 2 * K)
    return T


def _k_coefficients(labels: list[str], vecs: np.ndarray, J: int) -> np.ndarray:
    """Wang-basis eigenvectors -> |J K> coefficients, rows K = -J..J."""
    C = np.zeros((2 * J + 1, vecs.shape[1]))
    s = 1.0 / math.sqrt(2.0)
    for r, lab in enumerate(labels):
        body = lab.split("=")[1]
        if "," in body:
            K, g = int(body.split(",")[0]), (1 if body.endswith("+") else -1)
            C[K + J] += s * vecs[r]
            C[-K + J] += g * s * vecs[r]
        else:
            C[J] += vecs[r]
    return C


def dipole_spherical(axis: str, mu) -> np.ndarray:
    """(mu_a, mu_b, mu_c) -> molecule-fixed spherical components (mu_-1, mu_0, mu_+1)."""
    comp = dict(zip("abc", (float(m) for m in mu)))
    x, y, z = (comp[a] for a in _AXES[axis])
    return np.array([(x - 1j * y) / math.sqrt(2), z, -(x + 1j * y) / math.sqrt(2)])


def _levels_by_j(model, solution):
    labels = asym_top_labels(model)
    by_j: dict[int, list] = {}
    for b, r in zip(solution.blocks, solution.results):
        if "eigenvectors" not in r:
            raise ValueError("line strengths need a solution computed with keep_vectors=True")
        J = b["two_j"] // 2
        vecs = np.asarray(r["eigenvectors"])
        C = _k_coefficients(b["basis_labels"], vecs, J)
        for i, e in enumerate(r["eigenvalues"]):
            by_j.setdefault(J, []).append({"key": b["key"], "index": i, "energy": float(e),
                                           "label": labels[b["key"]][i], "c": C[:, i]})
    return by_j


def line_strengths(model, solution, mu, *, min_strength: float = 1e-12) -> list[Line]:
    """All dipole transitions between computed levels (Delta J = 0, +-1)."""
    if model.name != "asym_rotor":
        raise ValueError("line_strengths supports asym_rotor models")
    axis = json.loads(model.spec_json)["axis"]
    muq = dipole_spherical(axis, mu)
    mu2 = float(np.sum(np.abs(muq) ** 2))
    by_j = _levels_by_j(model, solution)
    lines = []
    for J in sorted(by_j):
        for Jp in (J, J + 1):
            if Jp not in by_j:
                continue
            T = _tables(Jp, J)
            M = np.tensordot(muq, T, axes=(0, 0))                    # (2J'+1, 2J+1), complex
            Cl = np.stack([lv["c"] for lv in by_j[J]], axis=1)
            Cu = np.stack([lv["c"] for lv in by_j[Jp]], axis=1)
            S = (2 * J + 1) * (2 * Jp + 1) * np.abs(Cu.T @ M @ Cl) ** 2  # rows: J' levels
            for a, up in enumerate(by_j[Jp]):
                for b, lo in enumerate(by_j[J]):
                    if S[a, b] < min_strength * mu2:
                        continue
                    if Jp == J and up["energy"] <= lo["energy"]:
                        continue  # count each Delta J = 0 pair once
                    u, l = (up, lo) if up["energy"] > lo["energy"] else (lo, up)
                    ju, jl = (Jp, J) if u is up else (J, Jp)
                    lines.append(Line((l["key"], l["index"]), (u["key"], u["index"]), l["label"], u["label"],
                                      l["energy"], u["energy"], u["energy"] - l["energy"], float(S[a, b]), jl, ju))
    lines.sort(key=lambda ln: ln.wavenumber)
    return lines


def strength_sum_rule_defect(model, solution, mu) -> float:
    """max_i |sum_f S(f, i) / ((2J+1)|mu|^2) - 1| over levels with J+1 in the model."""
    axis = json.loads(model.spec_json)["axis"]
    muq = dipole_spherical(axis, mu)
    mu2 = float(np.sum(np.abs(muq) ** 2))
    by_j = _levels_by_j(model, solution)
    worst = 0.0
    for J in sorted(by_j):
        if J + 1 not in by_j:
            continue
        Cl = np.stack([lv["c"] for lv in by_j[J]], axis=1)
        total = np.zeros(Cl.shape[1])
        for Jp in (J - 1, J, J + 1):
            if Jp < 0 or Jp not in by_j:
                continue
            M = np.tensordot(muq, _tables(Jp, J), axes=(0, 0))
            Cu = np.stack([lv["c"] for lv in by_j[Jp]], axis=1)
            if Cu.shape[1] != 2 * Jp + 1 or Cl.shape[1] != 2 * J + 1:
                raise ValueError("the sum rule needs complete J manifolds (solve with all levels)")
            total += ((2 * J + 1) * (2 * Jp + 1) * np.abs(Cu.T @ M @ Cl) ** 2).sum(axis=0)
        worst = max(worst, float(np.max(np.abs(total / ((2 * J + 1) * mu2) - 1.0))))
    return worst


def partition_function(solution, T: float, spin_weight: Callable[[str, str], float] | None = None,
                       labels: dict | None = None) -> float:
    e0 = min(float(min(r["eigenvalues"])) for r in solution.results if len(r["eigenvalues"]))
    q = 0.0
    for b, r in zip(solution.blocks, solution.results):
        J = b["two_j"] / 2
        for i, e in enumerate(r["eigenvalues"]):
            g = spin_weight(b["key"], labels[b["key"]][i] if labels else "") if spin_weight else 1.0
            q += g * (2 * J + 1) * math.exp(-C2 * (e - e0) / T)
    return q


def add_intensities(lines: list[Line], solution, T: float, *, spin_weight=None, labels=None) -> float:
    """Fill Line.intensity (relative absorption). Returns Q(T)."""
    e0 = min(float(min(r["eigenvalues"])) for r in solution.results if len(r["eigenvalues"]))
    Q = partition_function(solution, T, spin_weight, labels)
    for ln in lines:
        g = spin_weight(ln.lower[0], ln.lower_label) if spin_weight else 1.0
        nu = ln.wavenumber
        ln.intensity = (g * ln.strength * nu * math.exp(-C2 * (ln.lower_energy - e0) / T)
                        * (1 - math.exp(-C2 * nu / T)) / Q)
    return Q


def band_type(lower_label: str, upper_label: str) -> str:
    """'a', 'b' or 'c' from the change of (Ka, Kc) parities; '?' if none applies."""
    def par(lab):
        ka, kc = lab.split("(")[1].rstrip(")").split(",")
        return int(ka) % 2, int(kc) % 2
    (a1, c1), (a2, c2) = par(lower_label), par(upper_label)
    d = ((a1 + a2) % 2, (c1 + c2) % 2)
    return {(0, 1): "a", (1, 1): "b", (1, 0): "c"}.get(d, "?")
