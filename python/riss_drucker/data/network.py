"""MARVEL-style spectroscopic network inversion (README 5.4).

Levels are nodes, transitions are edges. With the signed incidence matrix D
(row t: +1 at the upper level, -1 at the lower) and W = diag(1/sigma_t^2), each
connected component is solved as (D^T W D) E = D^T W nu with its anchor level
fixed at 0 (or at a given value). Transitions with normalized residual above
`threshold` are flagged as outliers and down-weighted, and the solve repeats
(cycle-based validation: only transitions in cycles can be checked; levels
reached by a single transition are reported as unconfirmed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NetworkResult:
    levels: dict            # level key -> {"energy", "uncertainty", "component", "degree", "confirmed"}
    components: list        # list of {"anchor", "size", "anchored": bool}
    residuals: list         # per transition: normalized residual at the solution
    outliers: list          # indices of transitions flagged as outliers
    iterations: int
    settings: dict = field(default_factory=dict)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def invert(transitions, *, anchors: dict | None = None, threshold: float = 5.0, max_iter: int = 20,
           inflate: float = 10.0, dense_uncertainty_limit: int = 4000) -> NetworkResult:
    """transitions: iterable of objects/dicts with wavenumber, uncertainty, upper, lower.
    anchors: {level_key: energy}; components without an anchor are anchored at
    their first level (energies relative, reported as floating)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    tr = [(t["wavenumber"], t["uncertainty"], t["upper"], t["lower"]) if isinstance(t, dict)
          else (t.wavenumber, t.uncertainty, t.upper, t.lower) for t in transitions]
    if not tr:
        raise ValueError("no transitions")
    anchors = dict(anchors or {})
    uf = _UF()
    for _, _, u, l in tr:
        uf.union(u, l)
    comp_of: dict = {}
    members: dict = {}
    for _, _, u, l in tr:
        for x in (u, l):
            r = uf.find(x)
            if x not in comp_of:
                comp_of[x] = r
                members.setdefault(r, []).append(x)
    sig = np.array([s for _, s, _, _ in tr], dtype=float)
    nu = np.array([v for v, _, _, _ in tr], dtype=float)
    weight_scale = np.ones(len(tr))
    outliers: set = set()
    levels, comps = {}, []
    it = 0
    for it in range(1, max_iter + 1):
        levels.clear()
        comps.clear()
        resid = np.zeros(len(tr))
        for root, mem in members.items():
            idx = {k: n for n, k in enumerate(mem)}
            rows = [t for t, (_, _, u, _) in enumerate(tr) if comp_of[u] == root]
            anchor = next((k for k in mem if k in anchors), mem[0])
            e_anchor = float(anchors.get(anchor, 0.0))
            n = len(mem)
            free = [k for k in mem if k != anchor]
            fidx = {k: n_ for n_, k in enumerate(free)}
            data, ri, ci = [], [], []
            for r_, t in enumerate(rows):
                _, _, u, l = tr[t]
                for k, sgn in ((u, 1.0), (l, -1.0)):
                    if k != anchor:
                        data.append(sgn); ri.append(r_); ci.append(fidx[k])
            D = sp.csr_matrix((data, (ri, ci)), shape=(len(rows), len(free)))
            w = 1.0 / (sig[rows] * weight_scale[rows]) ** 2
            # Right-hand side with the anchor's contribution moved over.
            b = nu[rows].copy()
            for r_, t in enumerate(rows):
                _, _, u, l = tr[t]
                if u == anchor:
                    b[r_] -= e_anchor
                if l == anchor:
                    b[r_] += e_anchor
            if free:
                A = (D.T @ sp.diags(w) @ D).tocsc()
                E = spsolve(A, D.T @ (w * b)) if len(free) > 1 else np.array([(D.T @ (w * b))[0] / A[0, 0]])
                E = np.atleast_1d(E)
                if len(free) <= dense_uncertainty_limit:
                    var = np.diag(np.linalg.inv(A.toarray())) if len(free) > 1 else np.array([1.0 / A[0, 0]])
                else:
                    var = np.full(len(free), np.nan)
            else:
                E, var = np.zeros(0), np.zeros(0)
            calc = D @ E if free else np.zeros(len(rows))
            for r_, t in enumerate(rows):
                _, _, u, l = tr[t]
                c = calc[r_] + (e_anchor if u == anchor else 0.0) - (e_anchor if l == anchor else 0.0)
                resid[t] = (nu[t] - c) / sig[t]
            degree = {k: 0 for k in mem}
            for t in rows:
                if t in outliers:
                    continue
                _, _, u, l = tr[t]
                degree[u] += 1
                degree[l] += 1
            for k in mem:
                if k == anchor:
                    levels[k] = {"energy": e_anchor, "uncertainty": 0.0, "component": len(comps),
                                 "degree": degree[k], "confirmed": True}
                else:
                    j = fidx[k]
                    levels[k] = {"energy": float(E[j]), "uncertainty": float(np.sqrt(var[j])) if np.isfinite(var[j]) else None,
                                 "component": len(comps), "degree": degree[k], "confirmed": degree[k] >= 2}
            comps.append({"anchor": anchor, "size": n, "anchored": anchor in anchors})
        new_out = {t for t in range(len(tr)) if abs(resid[t]) > threshold * weight_scale[t]}
        if new_out <= outliers:
            break
        outliers |= new_out
        for t in new_out:
            weight_scale[t] = inflate
    return NetworkResult(levels, comps, resid.tolist(), sorted(outliers), it,
                         {"threshold": threshold, "inflate": inflate})
