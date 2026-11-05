"""MARVEL-style spectroscopic network inversion (README 5.4).

Levels are nodes, transitions are edges. With the signed incidence matrix D
(row t: +1 at the upper level, -1 at the lower) and weights w_t = 1/sigma_t^2,
each connected component is solved as (D^T W D) E = D^T W nu with an anchor
level fixed (MARVEL ground state at 0, or a given value).

Robustness: measured uncertainties span many decades (1e-6 .. 5e-2 cm^-1 in
the real H2S MARVEL input), so one mis-assigned line with a tiny claimed sigma
can dominate an ordinary least-squares solve and drag thousands of levels.
Huber weights alone do not help: they bound influence in units of each line's
own sigma. Outlier detection therefore uses capped precision,
sigma_eff = max(sigma, sigma_floor) with sigma_floor = median(sigma), and Huber
IRLS on that (influence bounded on an absolute scale). Lines whose residual
exceeds `threshold` x their *claimed* sigma at that consensus solution are
flagged; the clean set is solved with full 1/sigma^2 weights, re-checked, and
any newly discrepant lines are flagged until none remain (at most `rounds`). Only transitions in
cycles can be checked; a level reached by a single transition is reported as
unconfirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NetworkResult:
    levels: dict            # level key -> {"energy", "uncertainty", "component", "degree", "confirmed"}
    components: list        # list of {"anchor", "size", "anchored": bool, "n_transitions"}
    residuals: list         # per input transition: (nu - (E_u - E_l)) / sigma; NaN if ignored
    outliers: list          # indices into the input list of lines flagged as outliers
    iterations: int
    settings: dict = field(default_factory=dict)


def _components(tr):
    parent: dict = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, _, u, l in tr:
        ru, rl = find(u), find(l)
        if ru != rl:
            parent[ru] = rl
    members: dict = {}
    for _, _, u, l in tr:
        for x in (u, l):
            r = find(x)
            members.setdefault(r, {})[x] = None
    rows: dict = {}
    for t, (_, _, u, _) in enumerate(tr):
        rows.setdefault(find(u), []).append(t)
    return [(list(members[r].keys()), rows[r]) for r in members]


def _solve(mem, rows, tr, nu, w, anchor, e_anchor, want_var, dense_limit):
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    free = [k for k in mem if k != anchor]
    fidx = {k: i for i, k in enumerate(free)}
    data, ri, ci = [], [], []
    b = nu[rows].copy()
    for r_, t in enumerate(rows):
        _, _, u, l = tr[t]
        if u == anchor:
            b[r_] -= e_anchor
        else:
            data.append(1.0); ri.append(r_); ci.append(fidx[u])
        if l == anchor:
            b[r_] += e_anchor
        else:
            data.append(-1.0); ri.append(r_); ci.append(fidx[l])
    D = sp.csr_matrix((data, (ri, ci)), shape=(len(rows), len(free)))
    ww = w[rows]
    if not free:
        return free, np.zeros(0), np.zeros(0), np.zeros(len(rows)) + b, D
    A = (D.T @ sp.diags(ww) @ D).tocsc()
    lu = splu(A)
    E = lu.solve(D.T @ (ww * b))
    for _ in range(2):  # iterative refinement (weights span many decades)
        E = E + lu.solve(D.T @ (ww * (b - D @ E)))
    var = None
    if want_var:
        if len(free) <= dense_limit:
            var = np.diag(np.linalg.inv(A.toarray()))
        else:
            var = np.full(len(free), np.nan)
    return free, E, var, b - D @ E, D


def invert(transitions, *, anchors: dict | None = None, threshold: float = 5.0, huber: float = 2.0,
           max_iter: int = 50, tol: float = 1e-10, rounds: int = 30,
           dense_uncertainty_limit: int = 6000) -> NetworkResult:
    """transitions: iterable of objects/dicts with wavenumber, uncertainty, upper, lower.
    anchors: {level_key: energy}; components without an anchor are anchored at
    their first level (energies relative, reported as floating).
    Transitions marked excluded (MARVEL: negative wavenumber) or with uncertainty <= 0 are ignored."""
    def excluded(t):
        return t.get("excluded", False) if isinstance(t, dict) else getattr(t, "excluded", False)

    items = list(transitions)
    tr, orig = [], []
    for n, t in enumerate(items):
        x = (t["wavenumber"], t["uncertainty"], t["upper"], t["lower"]) if isinstance(t, dict) \
            else (t.wavenumber, t.uncertainty, t.upper, t.lower)
        if excluded(t) or not x[1] > 0 or x[2] == x[3]:
            continue
        tr.append(x)
        orig.append(n)
    if not tr:
        raise ValueError("no usable transitions")
    anchors = dict(anchors or {})
    sig = np.array([s for _, s, _, _ in tr], dtype=float)
    nu = np.array([v for v, _, _, _ in tr], dtype=float)
    w0 = 1.0 / sig**2
    resid = np.zeros(len(tr))
    outliers: set = set()
    levels, comps = {}, []
    total_iter = 0
    for mem, rows in _components(tr):
        rows = np.asarray(rows)
        anchor = next((k for k in mem if k in anchors), mem[0])
        e_anchor = float(anchors.get(anchor, 0.0))
        # 1) Consensus solve with capped precision + Huber IRLS.
        floor = float(np.median(sig[rows]))
        s_eff = np.maximum(sig, floor)
        hw = np.ones(len(tr))
        prev = None
        for it in range(max_iter):
            free, E, _, r, _ = _solve(mem, list(rows), tr, nu, hw / s_eff**2, anchor, e_anchor, False,
                                      dense_uncertainty_limit)
            ze = np.abs(r) / s_eff[rows]
            hw[rows] = np.where(ze <= huber, 1.0, huber / np.maximum(ze, 1e-300))
            total_iter += 1
            if prev is not None and len(E) and np.max(np.abs(E - prev)) <= tol * (1.0 + np.max(np.abs(E))):
                break
            prev = E
        z = np.abs(r) / sig[rows]  # against the claimed uncertainties
        bad_mask = z > threshold
        # 2) Full-precision solves on the clean set. Every line (flagged or not)
        #    is re-judged against each clean solution: lines that fit are
        #    un-flagged (the capped consensus is only good to ~sigma_floor, so
        #    precise lines can look discrepant at first); newly discrepant lines
        #    are flagged worst-first, since one bad line can push neighbours over.
        seen_states = set()
        for _ in range(rounds):
            keep = rows[~bad_mask]
            if len(keep) == 0:
                break
            sub = _components([tr[t] for t in keep])
            mm, rr = next(((m_, [keep[i] for i in r_]) for m_, r_ in sub if anchor in m_), ([anchor], []))
            if not rr:
                break
            fr, Ec, _, _, _ = _solve(mm, rr, tr, nu, w0, anchor, e_anchor, False, dense_uncertainty_limit)
            Em = {anchor: e_anchor, **{k: float(Ec[i]) for i, k in enumerate(fr)},
                  **{k: v for k, v in zip([anchor] + free, [e_anchor] + list(E)) if k not in mm}}
            z_all = np.array([abs(nu[t] - (Em[tr[t][2]] - Em[tr[t][3]])) / sig[t] for t in rows])
            now_bad = z_all > threshold
            add = now_bad & ~bad_mask
            new_mask = bad_mask & now_bad
            if add.any():
                idx = np.nonzero(add)[0]
                idx = idx[np.argsort(-z_all[idx])][: max(1, len(idx) // 4)]
                new_mask[idx] = True
            key = new_mask.tobytes()
            if np.array_equal(new_mask, bad_mask) or key in seen_states:
                bad_mask = new_mask
                break
            seen_states.add(key)
            bad_mask = new_mask
        bad = rows[bad_mask]
        outliers.update(bad.tolist())
        robust = {anchor: e_anchor, **{k: float(E[i]) for i, k in enumerate(free)}}
        clean = rows[~bad_mask]
        comp_id = len(comps)
        # 2) Final plain solve on the clean set; excluding outliers can split the
        #    component, so re-split and solve the part connected to the anchor.
        sub = _components([tr[t] for t in clean]) if len(clean) else []
        main_mem, main_rows = next(((m_, [clean[i] for i in r_]) for m_, r_ in sub if anchor in m_), ([anchor], []))
        free, E2, var, _, _ = _solve(main_mem, main_rows, tr, nu, w0, anchor, e_anchor, True,
                                     dense_uncertainty_limit)
        final = {anchor: e_anchor, **{k: float(E2[i]) for i, k in enumerate(free)}}
        fpos = {k: i for i, k in enumerate(free)}
        degree = {k: 0 for k in mem}
        for t in clean:
            _, _, u, l = tr[t]
            degree[u] += 1
            degree[l] += 1
        for k in mem:
            if k in final:
                v = 0.0 if k == anchor else var[fpos[k]]
                levels[k] = {"energy": final[k], "uncertainty": float(np.sqrt(v)) if np.isfinite(v) else None,
                             "component": comp_id, "degree": degree[k],
                             "confirmed": k == anchor or degree[k] >= 2, "isolated_by_outliers": False}
            else:
                levels[k] = {"energy": robust[k], "uncertainty": None, "component": comp_id,
                             "degree": degree[k], "confirmed": False, "isolated_by_outliers": True}
        Emap = {k: v["energy"] for k, v in levels.items() if v["component"] == comp_id}
        for t in rows:
            _, _, u, l = tr[t]
            resid[t] = (nu[t] - (Emap[u] - Emap[l])) / sig[t]
        comps.append({"anchor": anchor, "size": len(mem), "anchored": anchor in anchors,
                      "n_transitions": int(len(rows)), "n_outliers": int(len(bad)),
                      "n_isolated_by_outliers": int(len(mem) - len(final))})
    # Report per input transition (indices into the caller's list; NaN for ignored lines).
    full = np.full(len(items), np.nan)
    full[orig] = resid
    return NetworkResult(levels, comps, full.tolist(), sorted(orig[t] for t in outliers), total_iter,
                         {"threshold": threshold, "huber": huber, "rounds": rounds,
                          "method": "capped-precision Huber IRLS, then full-precision clean re-solves"})
