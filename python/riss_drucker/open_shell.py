"""Open-shell level assignment (Hund's case (b) labels) for open_rotor models.

The C++ model works in the case (a) basis |J P>|S Sigma>; levels are labelled by
expectation values computed here with the same operator construction:
N from <N^2> = N(N+1), |Ka| from <N_a^2>, and the fine-structure component
F_i ordered by J = N + S, N + S - 1, ... (F1 has the largest J for given N).
`purity` is how close <N^2> is to an integer N(N+1) (1 = pure case (b)).
"""
from __future__ import annotations

import json
import math

import numpy as np

_AXES = {"Ir": ("b", "c", "a"), "IIr": ("c", "a", "b"), "IIIr": ("a", "b", "c")}


def _angmom(two_j: int):
    n = two_j + 1
    j = two_j / 2
    jp = np.zeros((n, n))
    for i in range(n - 1):
        m = -j + i
        jp[i + 1, i] = math.sqrt(j * (j + 1) - m * (m + 1))
    jm = jp.T
    return 0.5 * (jp + jm), -0.5j * (jp - jm), np.diag(-j + np.arange(n))


def operators(two_j: int, two_s: int) -> dict:
    """Molecule-fixed N components (x, y, z) and N^2 in the model's basis order."""
    Jx, Jy, Jz = _angmom(two_j)
    Sx, Sy, Sz = _angmom(two_s)
    IJ, IS = np.eye(two_j + 1), np.eye(two_s + 1)
    S = [np.kron(IJ, a) for a in (Sx, Sy, Sz)]
    N = [np.kron(a, IS) - s for a, s in zip((Jx, -Jy, Jz), S)]
    return {"N": N, "N2": sum(n @ n for n in N)}


def assign(model, solution) -> dict:
    """Block key -> list of {N, J, F, Ka, purity} per level (needs keep_vectors=True)."""
    spec = json.loads(model.spec_json)
    if spec.get("model") != "open_rotor":
        raise ValueError("assign needs an open_rotor model")
    two_s, axis = spec["two_s"], spec["axis"]
    a_idx = _AXES[axis].index("a")  # representation index (x=0, y=1, z=2) of the a axis
    out = {}
    for b, r in zip(solution.blocks, solution.results):
        if "eigenvectors" not in r:
            raise ValueError("assign needs a solution computed with keep_vectors=True")
        ops = operators(b["two_j"], two_s)
        V = np.asarray(r["eigenvectors"])
        n2 = np.real(np.einsum("ij,ik,kj->j", V.conj(), ops["N2"], V))
        na2 = np.real(np.einsum("ij,ik,kj->j", V.conj(), ops["N"][a_idx] @ ops["N"][a_idx], V))
        rows = []
        J = b["two_j"] / 2
        S = two_s / 2
        for x, ka2 in zip(n2, na2):
            Nf = (-1 + math.sqrt(1 + 4 * max(x, 0.0))) / 2
            N = int(round(Nf))
            F = int(round(N + S - J)) + 1  # F1: J = N + S
            rows.append({"N": N, "J": J, "F": F, "Ka": int(round(math.sqrt(max(ka2, 0.0)))),
                         "purity": float(1.0 - min(1.0, abs(x - N * (N + 1)) / max(1.0, 2 * N + 1)))})
        out[b["key"]] = rows
    return out
