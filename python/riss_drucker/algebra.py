"""Operator dictionary and algebraic counts as certificates (README 7; conventions 18).

Characters and Molien/Hilbert series give counts, not operators. Here they check
the operator library built by the grammar:

* `enumerate_terms` builds every symmetry-allowed, Hermitian operator of a given
  multidegree (per space) from the grammar (sym(word) + h.c., i*sym(word) + h.c.),
  keeps a maximal independent set by the *leading symbol* (commutative image of
  each space's atoms; finite spaces - spin, electronic - exactly), and selects
  time-reversal-even terms (Hamiltonian) or odd ones (contact-transformation
  generators S = iK).
* `molien_count` predicts that number from characters alone:
      D  = 1/|G| sum_g prod_s chi_s(g),   T = 1/|G| sum_g prod_s tau_s(g),
      #even = (D + T)/2,   #odd = (D - T)/2,
  with chi_s the character of the group on degree-d operator symbols of space s and
  tau_s the trace of (dagger o Theta) o g on them. A mismatch is a compiler bug and
  aborts.
* `redundancy` counts what unitary (contact) transformations remove:
  dim(span{O_k} ∩ span{[S_j, H]}) at the current parameters (linearized,
  floating-point rank with stated tolerance). Identifiable parameters <= n_terms -
  that; the Fisher rank of a fit may not exceed it (certificate).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .build import (BuildSpec, BuiltModel, Context, GrammarError, Parser, _close, compile_build, dagger, make_group,
                    poly_add, runtime_spaces, theta_conj)
from .symmetry import complete_homogeneous, spin_character, su2_angle_axis


# ---- multidegrees -----------------------------------------------------------------------

def multidegrees(spec: BuildSpec, *, vib=(0, 0), rot=(0, 0), torsion=None, osc=None, total=None):
    """All multidegrees within limits. vib, rot: (min, max) degrees; torsion: {name: (kmax_steps, mmax)}
    with k in units of 1/kappa; osc: {name: (min, max)}. total caps vib + rot + m + osc degrees."""
    group = make_group(spec)
    spaces = runtime_spaces(spec, group)
    axes = []
    for s in spaces:
        if s.kind == "vib":
            axes.append([("vib", n) for n in range(vib[0], vib[1] + 1)])
        elif s.kind == "rot":
            axes.append([("rot", n) for n in range(rot[0], rot[1] + 1)])
        elif s.kind == "torsion":
            kmax, mmax = (torsion or {}).get(s.name, (0, 0))
            axes.append([("torsion", s.name, k, m) for k in range(0, kmax + 1) for m in range(0, mmax + 1)])
        elif s.kind == "osc":
            lo, hi = (osc or {}).get(s.name, (0, 0))
            axes.append([("osc", s.name, n) for n in range(lo, hi + 1)])
        else:
            axes.append([(s.kind,)])
    out = []
    for combo in itertools.product(*axes):
        deg = sum(d[1] if d[0] in ("vib", "rot") else d[3] if d[0] == "torsion" else d[2] if d[0] == "osc" else 0
                  for d in combo)
        if total is not None and deg > total:
            continue
        out.append(combo)
    return out


# ---- per-space monomials and symbols -------------------------------------------------------

def _space_words(ctx: Context, s, d):
    """Words (tuples of atoms) spanning degree d of space s."""
    if s.kind == "vib":
        n = d[1]
        atoms = [("b", "c", m, c) for m, c in s.osc] + [("b", "a", m, c) for m, c in s.osc]
        weights = {m.name: m.weight for m in s.modes}
        words = []
        for combo in itertools.combinations_with_replacement(range(len(atoms)), n):
            w = tuple(atoms[i] for i in combo)
            if s.decl.polyads is not None:
                q = sum((1 if a[1] == "c" else -1) * weights[a[2]] for a in w)
                if q != 0:
                    continue
            words.append(w)
        return words
    if s.kind == "rot":
        return [tuple(("J", a) for a in combo) for combo in itertools.combinations_with_replacement("xyz", d[1])]
    if s.kind == "spin":
        out = []
        for n in range(0, s.ts + 1):
            out += [tuple(("S", a) for a in combo) for combo in itertools.combinations_with_replacement("xyz", n)]
        return out
    if s.kind == "elec":
        return [(("e", a, b),) for a in s.comps for b in s.comps]
    if s.kind == "well":
        return [(("w", i + 1, j + 1),) for i in range(s.n) for j in range(s.n)]
    if s.kind == "torsion":
        _, name, k, m = d
        ps = tuple(("p", name) for _ in range(m))
        if k == 0:
            return [ps]
        kk = k / s.kappa
        return [(("f", "cos", kk, name),) + ps, (("f", "sin", kk, name),) + ps]
    if s.kind == "osc":
        _, name, n = d
        return [tuple(("q", name) for _ in range(a)) + tuple(("p", name) for _ in range(n - a)) for a in range(n + 1)]
    raise GrammarError(f"no words for space {s.kind}")


def _symbol(ctx: Context, poly: dict) -> dict:
    """Leading symbol: commutative image per infinite space; exact matrix units for
    finite spaces (spin, electronic)."""
    out = {}
    for w, c in poly.items():
        per = [[] for _ in ctx.spaces]
        for a in w:
            per[ctx.space_of(a)].append(a)
        finite = []
        keys = []
        for si, s in enumerate(ctx.spaces):
            atoms = per[si]
            if s.kind in ("spin", "elec", "well"):
                M = None
                for a in atoms:
                    A = s.atom(a, 0).toarray()
                    M = A if M is None else M @ A
                if M is None:
                    M = np.eye(s.dim(0))
                finite.append(M)
                keys.append(None)
            else:
                keys.append(tuple(sorted(atoms)))
        if finite:
            F = finite[0]
            for x in finite[1:]:
                F = np.kron(F, x)
            for (r, cc) in zip(*np.nonzero(np.abs(F) > 1e-14)):
                k = (tuple(keys), int(r), int(cc))
                out[k] = out.get(k, 0) + c * F[r, cc]
        else:
            k = (tuple(keys), 0, 0)
            out[k] = out.get(k, 0) + c
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def _word_text(w) -> str:
    parts = []
    for a in w:
        t = a[0]
        if t == "b":
            parts.append(f"a{a[2]}{a[3]}{'+' if a[1] == 'c' else ''}")
        elif t in ("J", "S"):
            parts.append(f"{t}{a[1]}")
        elif t == "e":
            parts.append(f"e[{a[1]},{a[2]}]")
        elif t == "w":
            parts.append(f"w[{a[1]},{a[2]}]")
        elif t == "f":
            k = a[2]
            ks = f"{k:g}" if abs(k - round(k)) > 1e-12 else str(int(round(k)))
            parts.append(f"{a[1]}{ks}[{a[3]}]")
        elif t in ("p", "q"):
            parts.append(f"{t}[{a[1]}]")
    return " ".join(parts) if parts else "I"


@dataclass
class Enumeration:
    terms: list               # term strings (with parameter names)
    by_degree: list           # [{degree, count, predicted, terms}]
    certificates: list


def enumerate_terms(spec: BuildSpec, degrees: list, *, parity: int = 1, prefix: str = "c",
                    exclude: list | None = None, check: bool = True) -> Enumeration:
    """Independent symmetry-allowed Hermitian operators of each multidegree.

    parity=+1: time-reversal even (Hamiltonian terms); -1: odd (generators K, S = iK).
    exclude: operator texts (or terms) already in the model; their symbols are
    projected out first so the result completes them.
    """
    group = make_group(spec)
    spaces = runtime_spaces(spec, group)
    ctx = Context(group, spaces)
    parser = Parser(ctx)
    certs, out_terms, by_deg = [], [], []
    ex_polys = []
    for t in exclude or []:
        txt = t.split(":", 1)[1] if ":" in t else t
        ex_polys.append(ctx.canon(parser.parse_expr(txt)))
    counter = 0
    for md in degrees:
        per_words = [_space_words(ctx, s, d) for s, d in zip(spaces, md)]
        Q = []          # orthonormal basis of accepted symbols (dicts)

        def add(v, commit=True):
            x = dict(v)
            for q in Q:
                d = sum(np.conj(q[k]) * x[k] for k in q if k in x)
                if abs(d) > 0:
                    for k, c in q.items():
                        x[k] = x.get(k, 0) - d * c
            nr = math.sqrt(sum(abs(c) ** 2 for c in x.values()))
            ref = max(1e-300, max(abs(c) for c in v.values()) if v else 1.0)
            if nr > 1e-8 * ref:
                if commit:
                    Q.append({k: c / nr for k, c in x.items() if abs(c) > 1e-15 * nr})
                return True
            return False

        def vec(sym):
            return sym
        for p in ex_polys:
            sym = _symbol(ctx, p)
            if sym and all(_degree_of_key(ctx, k) == _md_key(md) for k in sym):
                add(vec(sym))
        n_ex = len(Q)
        found = []
        for combo in itertools.product(*per_words):
            w = tuple(a for part in combo for a in part)
            P = ctx.sym({w: 1.0})
            if not P:
                continue
            for phase, ph_txt in ((1.0, ""), (1j, "i*")):
                Pp = {k: phase * v for k, v in P.items()}
                O = ctx.canon(poly_add(Pp, dagger(Pp)))
                if not O:
                    continue
                th = ctx.canon(theta_conj(O))
                if parity == 1 and not _close(O, th):
                    continue
                if parity == -1 and not _close({k: -v for k, v in O.items()}, th):
                    continue
                sym = _symbol(ctx, O)
                if not sym:
                    continue
                if add(vec(sym)):
                    counter += 1
                    name = f"{prefix}{counter}"
                    found.append(f"{name}: sym({ph_txt}{_word_text(w)}) + h.c.")
        pred = molien_count(group, spaces, md, parity=parity)
        rank = len(Q)
        if check:
            ok = rank == pred
            certs.append({"name": f"molien_{_md_label(md)}", "passed": ok, "value": rank, "bound": pred,
                          "detail": "independent symmetry-allowed Hermitian operators (leading symbols) equal the "
                                    "character (Molien-type) count"})
            if not ok:
                raise GrammarError(f"operator count {rank} != character count {pred} at {_md_label(md)}: "
                                   "compiler bug")
        by_deg.append({"degree": _md_label(md), "count": rank - n_ex, "predicted": pred, "existing": n_ex,
                       "terms": found})
        out_terms += found
    return Enumeration(out_terms, by_deg, certs)


def _md_key(md):
    return tuple(md)


def _degree_of_key(ctx, key):
    keys = key[0]
    out = []
    for s, atoms in zip(ctx.spaces, keys):
        if s.kind == "vib":
            out.append(("vib", len(atoms)))
        elif s.kind == "rot":
            out.append(("rot", len(atoms)))
        elif s.kind == "torsion":
            f = [a for a in atoms if a[0] == "f"]
            k = round(f[0][2] * s.kappa) if f else 0
            out.append(("torsion", s.name, k, sum(1 for a in atoms if a[0] == "p")))
        elif s.kind == "osc":
            out.append(("osc", s.name, len(atoms)))
        else:
            out.append((s.kind,))
    return tuple(out)


def _md_label(md) -> str:
    parts = []
    for d in md:
        if d[0] in ("vib", "rot"):
            parts.append(f"{d[0]}{d[1]}")
        elif d[0] == "torsion":
            parts.append(f"{d[1]}:k{d[2]}p{d[3]}")
        elif d[0] == "osc":
            parts.append(f"{d[1]}:{d[2]}")
        else:
            parts.append(d[0])
    return ",".join(parts)


# ---- character counts -------------------------------------------------------------------------

def _laurent_h(lams, ws, n):
    """[t^n z^0] prod_j 1/((1 - t z^w_j lam_j)(1 - t z^-w_j lam_j))."""
    W = max([abs(w) for w in ws] + [1]) * n
    size = 2 * W + 1
    # series[t][z+W]
    ser = np.zeros((n + 1, size), complex)
    ser[0, W] = 1.0
    for lam, w in zip(lams, ws):
        for sign in (1, -1):
            new = np.zeros_like(ser)
            for k in range(0, n + 1):
                # multiply by (t z^{sign w} lam)^k
                shift = sign * w * k
                fac = lam**k
                for tdeg in range(0, n + 1 - k):
                    row = ser[tdeg]
                    if not np.any(row):
                        continue
                    if shift >= 0:
                        new[tdeg + k, shift:] += fac * row[:size - shift] if shift else fac * row
                    else:
                        new[tdeg + k, :size + shift] += fac * row[-shift:]
            ser = new
    return ser[n, W]


def _space_chars(group, s, d, gi):
    el = group.elements[gi]
    if s.kind == "vib":
        n = d[1]
        lams, ws = [], []
        for m in s.modes:
            ev = np.linalg.eigvals(group.irreps[m.irrep][gi])
            lams += list(ev)
            ws += [m.weight] * len(ev)
        if s.decl.polyads is not None:
            chi = _laurent_h(lams, ws, n)
        else:
            chi = complete_homogeneous(lams + lams, n)
        tau = complete_homogeneous([l * l for l in lams], n // 2) if n % 2 == 0 else 0.0
        return chi, tau
    if s.kind == "rot":
        n = d[1]
        ev = np.linalg.eigvals(el.R)
        h = complete_homogeneous(ev, n)
        return h, (-1) ** n * h
    if s.kind == "spin":
        c = spin_character(s.ts, el.u)
        c2 = spin_character(s.ts, el.u @ el.u)
        return abs(c) ** 2, (-1) ** s.ts * c2
    if s.kind in ("elec", "well"):
        c = s.char(gi, 0)
        g2 = group.table[gi, gi]
        c2 = s.char(int(g2), 0)
        return abs(c) ** 2, c2
    if s.kind == "torsion":
        _, name, k, m = d
        sg, c = el.coord(name)
        kk = k / s.kappa
        f = 1.0 if k == 0 else (1 + sg) * math.cos(kk * c)
        chi = f * sg**m
        return chi, (-1) ** m * chi
    if s.kind == "osc":
        _, name, n = d
        sg, _ = el.coord(name)
        return (n + 1) * sg**n, (sg**n if n % 2 == 0 else 0.0)
    raise GrammarError(f"no character for space {s.kind}")


def molien_count(group, spaces, md, *, parity: int = 1) -> int:
    D = T = 0.0
    for gi in range(group.order):
        chi, tau = 1.0, 1.0
        for s, d in zip(spaces, md):
            c, t = _space_chars(group, s, d, gi)
            chi *= c
            tau *= t
        D += chi
        T += tau
    D, T = D / group.order, T / group.order
    v = (D + parity * T) / 2
    r = round(v.real)
    if abs(v - r) > 1e-6:
        raise GrammarError(f"non-integer Molien count {v}")
    return int(r)


def molien_series(group, spaces, md_list, parity: int = 1) -> dict:
    return {_md_label(md): molien_count(group, spaces, md, parity=parity) for md in md_list}


# ---- redundancy (contact transformations) -----------------------------------------------------

def redundancy(bm: BuiltModel, theta, generator_degrees: list, *, h0: list | None = None,
               tol: float = 1e-9) -> dict:
    """Order-by-order unitary-equivalence analysis.

    Generators K (time-reversal-odd Hermitian invariants of the given degrees),
    S = iK; redundant directions = span{O_k} ∩ span{[S_j, H0]} over all blocks, with
    H0 the zeroth-order Hamiltonian: the parameters in `h0` (default: terms with at
    most two atoms per word) at their values in theta, others zero.
    """
    gens = enumerate_terms(bm.spec, generator_degrees, parity=-1, prefix="K", check=True).terms
    if h0 is None:
        h0 = [t.param for t in bm.terms if t.order <= 2]
    th0 = {n: (float(theta[n]) if n in h0 else 0.0) for n in bm.param_names}
    H = bm.block_matrices(th0)
    O = []
    for k in range(len(bm.param_names)):
        O.append(np.concatenate([_op_dense(bm, b, k).ravel() for b in range(len(bm.blocks))]))
    O = np.array(O).T if O else np.zeros((0, 0))
    C = []
    for g in gens:
        txt = g.split(":", 1)[1]
        Kb = bm.operator_blocks(txt)
        cols = []
        for Hb, K in zip(H, Kb):
            S = (1j * K).real
            cols.append((S @ Hb - Hb @ S).ravel())
        C.append(np.concatenate(cols))
    C = np.array(C).T if C else np.zeros((O.shape[0], 0))

    gaps = []

    def rank(A):
        r, gap = certified_rank(A, tol)
        gaps.append(gap)
        return r
    rO, rC, rOC = rank(O), rank(C), rank(np.hstack([O, C])) if C.size else rank(O)
    inter = rO + rC - rOC
    directions = []
    if inter > 0:
        M = np.hstack([O, -C])
        _, s, Vt = np.linalg.svd(M)
        null = Vt[np.sum(s > tol * s[0]):]
        for v in null[:inter]:
            a = v[:O.shape[1]]
            if np.linalg.norm(a) > 1e-8:
                a = a / np.max(np.abs(a))
                directions.append({n: float(x) for n, x in zip(bm.param_names, a) if abs(x) > 1e-6})
    return {"n_terms": len(bm.param_names), "n_generators": len(gens), "redundant": int(inter),
            "identifiable_max": int(rO - inter), "operator_rank": rO, "h0": list(h0),
            "min_rank_gap": float(min(gaps)) if gaps else float("inf"), "directions": directions, "generators": gens}


def certified_rank(A, tol: float = 1e-9, min_gap: float = 1e4):
    """Numerical rank with a certificate: the singular values must show a gap of at least
    `min_gap` at the cut (relative tolerance `tol`), otherwise the rank is not decidable in
    floating point and GrammarError is raised. Returns (rank, gap)."""
    if A.size == 0:
        return 0, float("inf")
    s = np.linalg.svd(np.asarray(A), compute_uv=False)
    if s[0] == 0:
        return 0, float("inf")
    r = int(np.sum(s > tol * s[0]))
    if r == len(s):
        return r, float("inf")
    gap = float(s[r - 1] / max(s[r], 1e-300)) if r > 0 else float(s[0] / max(s[0], 1e-300))
    if r > 0 and gap < min_gap:
        raise GrammarError(f"rank not certifiable: singular-value gap {gap:.1e} < {min_gap:.0e} at rank {r}")
    return r, gap


def _op_dense(bm, b, k):
    import scipy.sparse as sp
    ip, ix, dt, shape = bm.model.operator(b, k)
    return sp.csr_matrix((dt, ix, ip), shape=tuple(shape)).toarray()


def fisher_rank_certificate(fitres, red: dict) -> dict:
    ok = fitres.rank <= red["identifiable_max"]
    return {"name": "fisher_rank_vs_redundancy", "passed": bool(ok), "value": float(fitres.rank),
            "bound": float(red["identifiable_max"]),
            "detail": "Fisher rank of the fit does not exceed terms minus contact-transformation redundancies"}
