"""Dynamic effective-Hamiltonian builder over product spaces (README 7; M7/M8).

A model is not chosen from a fixed list (rigid rotor, Watson, Hund's case ...):
it is *built* from
* spaces   - rotation |J k> (molecule-fixed, anomalous J), electron spin |S s>,
             electronic states |i> (with irreps), vibrational Fock spaces (modes with
             irreps, polyad weights), torsion-like periodic coordinates (period 2pi or
             4pi), and harmonic-oscillator LAM coordinates;
* a group  - point, permutation-inversion or double group (rd.symmetry), possibly
             with gauge elements (e.g. the H2O2 half-angle frame);
* terms    - text in the operator grammar, e.g.
                 "B: J2"   "zeta: sym(i*Jz a3x+ a3y) + h.c."   "F1: p[t] cos1[t] p[t]"
                 "k: sym(e[1x,1y] q2x)"   "gamma: Nx Sx + Ny Sy + Nz Sz"
The compiler builds every term in the product basis, certifies it (Hermitian,
time-reversal even, invariant under every group element, conserving the block
quantum numbers), projects onto irreps, and returns real symmetric blocks
(J, polyad, Gamma) in a time-reversal-adapted basis. The independent counts per
block from characters must equal the projected dimensions (certificate).

Grammar atoms (word order is operator order; different spaces commute):
  a<mode><comp>[+]  q<mode><comp>  p<mode><comp>  n<mode>      vibrational ladders
  Jx Jy Jz J2  |  Sx Sy Sz S2  |  Nx Ny Nz N2 (N = J - S)        angular momenta
  e[<i>,<j>]  Pe[<state>]                                         electronic |i><j|, projector
  cos<k>[<c>] sin<k>[<c>] p[<c>] q[<c>]                           LAM coordinate c (k may be 0.5)
  sym( ... )   + h.c.   i   I   numbers   ( )   * + -
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from . import _core
from .grammar import GrammarError
from .symmetry import (Group, SymmetryError, angular_momentum_matrices, complete_homogeneous, get_group,
                       spin_character, su2_from_axis_angle, wigner_D)

COMP_LETTERS = "xyzwuv"
_C2Y = su2_from_axis_angle([0, 1, 0], math.pi)


# ---- space declarations (serializable) -------------------------------------------------

@dataclass
class Mode:
    name: str
    irrep: str
    weight: int = 1


@dataclass
class VibSpace:
    modes: list
    polyads: list | None = None      # conserved polyad numbers kept as blocks
    nmax: int | None = None          # total-quanta truncation when no polyad is conserved
    kind: str = "vib"


@dataclass
class RotSpace:
    kind: str = "rot"


@dataclass
class SpinSpace:
    two_s: int
    kind: str = "spin"


@dataclass
class ElecSpace:
    states: list                     # [(name, irrep), ...]
    kind: str = "elec"


@dataclass
class TorsionSpace:
    name: str
    m_max: int
    period_pi: int = 2               # 2 -> period 2pi, 4 -> period 4pi (half-angle frames)
    kind: str = "torsion"


@dataclass
class WellSpace:
    """Localized minima ("wells") of a large-amplitude / Jahn-Teller problem. Each well carries an
    adiabatic electronic function given as a vector in the carrier space of `irrep`; the group
    permutes the wells with signs (rho(g) v_n = s v_m), which encodes the geometric phase."""
    name: str
    irrep: str
    vectors: list
    geometric_phase: bool = True     # False: wells permuted without signs (phase-free model)
    kind: str = "well"


@dataclass
class OscSpace:
    name: str
    nmax: int
    kind: str = "osc"


_SPACE_TYPES = {"vib": VibSpace, "rot": RotSpace, "spin": SpinSpace, "elec": ElecSpace, "torsion": TorsionSpace,
                "osc": OscSpace, "well": WellSpace}


def _space_dict(s) -> dict:
    d = dict(s.__dict__)
    if s.kind == "vib":
        d["modes"] = [[m.name, m.irrep, m.weight] for m in s.modes]
    if s.kind == "elec":
        d["states"] = [list(x) for x in s.states]
    return d


def _space_from_dict(d: dict):
    d = dict(d)
    k = d["kind"]
    if k == "vib":
        d["modes"] = [Mode(*m) for m in d["modes"]]
    if k == "elec":
        d["states"] = [tuple(x) for x in d["states"]]
    return _SPACE_TYPES[k](**d)


@dataclass
class BuildSpec:
    spaces: list
    group: str = "C1"
    terms: list = field(default_factory=list)
    J: list = field(default_factory=lambda: [0])
    max_order: int = 10
    name: str = "built"

    def to_dict(self) -> dict:
        return {"model": "built", "name": self.name,
                "group": self.group if isinstance(self.group, str) else group_to_dict(self.group),
                "spaces": [_space_dict(s) for s in self.spaces], "terms": list(self.terms),
                "J": [float(j) for j in self.J], "max_order": self.max_order}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "BuildSpec":
        grp = d["group"] if isinstance(d["group"], str) else group_from_dict(d["group"])
        return cls([_space_from_dict(s) for s in d["spaces"]], grp, list(d["terms"]),
                   [j if j % 1 else int(j) for j in d["J"]], d.get("max_order", 10), d.get("name", "built"))

    def with_terms(self, terms) -> "BuildSpec":
        return BuildSpec(self.spaces, self.group, list(terms), self.J, self.max_order, self.name)


def group_to_dict(g: Group) -> dict:
    """Serializable description: generators and irrep names keyed by characters on them."""
    gi = [g.find(x) for x in g.generators]
    def c(z):
        return [round(float(np.real(z)), 10), round(float(np.imag(z)), 10)]
    return {"name": g.name, "double": g.double, "periods": g.periods,
            "generators": [{"R": np.round(x.R, 12).tolist(), "parity": x.parity, "perm": list(x.perm),
                            "coords": [list(t) for t in x.coords], "gauge": x.gauge} for x in g.generators],
            "names": {nm: [round(float(np.real(ch[0]))), [c(ch[i]) for i in gi]] for nm, ch in g.chars.items()}}


def group_from_dict(d: dict) -> Group:
    from .symmetry import Element, su2_lift
    gens = []
    for x in d["generators"]:
        R = np.array(x["R"], float)
        gens.append(Element(R, su2_lift(R), int(x["parity"]), tuple(x["perm"]),
                            tuple((t[0], int(t[1]), float(t[2])) for t in x["coords"]), bool(x["gauge"])))
    names = {nm: (dim, [complex(a, b) for a, b in ch]) for nm, (dim, ch) in d["names"].items()}
    return Group(d["name"], gens, double=d["double"], periods=d.get("periods") or {}, names=names)


# ---- runtime spaces ------------------------------------------------------------------------

def _csr(M) -> sp.csr_matrix:
    return sp.csr_matrix(M, dtype=complex)


class _Space:
    kind = ""

    def owns(self, atom) -> bool:
        raise NotImplementedError


class _Rot(_Space):
    kind = "rot"

    def __init__(self, group: Group):
        self.g = group
        self._cache = {}

    def owns(self, atom):
        return atom[0] == "J"

    def dim(self, tj):
        return tj + 1

    def labels(self, tj):
        return [f"k={(-tj + 2 * i) / 2:g}" for i in range(tj + 1)]

    def mats(self, tj):
        if tj not in self._cache:
            jx, jy, jz = angular_momentum_matrices(tj)
            self._cache[tj] = {"x": jx, "y": -jy, "z": jz}       # molecule-fixed, anomalous
        return self._cache[tj]

    def atom(self, a, tj):
        return _csr(self.mats(tj)[a[1]])

    def U(self, gi, tj):
        u = self.g.elements[gi].u
        return _csr(wigner_D(tj, _C2Y @ u @ _C2Y.conj().T))

    def theta(self, tj):
        from scipy.linalg import expm
        _, jy, _ = angular_momentum_matrices(tj)
        return _csr(expm(1j * math.pi * jy))

    def char(self, gi, tj, sector=None):
        return spin_character(tj, self.g.elements[gi].u)


class _Spin(_Space):
    kind = "spin"

    def __init__(self, group, two_s):
        self.g, self.ts = group, int(two_s)
        jx, jy, jz = angular_momentum_matrices(self.ts)
        self.m = {"x": jx, "y": jy, "z": jz}

    def owns(self, atom):
        return atom[0] == "S"

    def dim(self, tj):
        return self.ts + 1

    def labels(self, tj):
        return [f"s={(-self.ts + 2 * i) / 2:g}" for i in range(self.ts + 1)]

    def atom(self, a, tj):
        return _csr(self.m[a[1]])

    def U(self, gi, tj):
        return _csr(wigner_D(self.ts, self.g.elements[gi].u))

    def theta(self, tj):
        from scipy.linalg import expm
        return _csr(expm(1j * math.pi * self.m["y"]))

    def char(self, gi, tj, sector=None):
        return spin_character(self.ts, self.g.elements[gi].u)


class _Elec(_Space):
    kind = "elec"

    def __init__(self, group, states):
        self.g = group
        self.comps, self.state_of, self.offset = [], {}, {}
        for name, irrep in states:
            if irrep not in group.irreps:
                raise GrammarError(f"electronic irrep {irrep!r} not in group {group.name}")
            d = group.irreps[irrep][0].shape[0]
            self.offset[name] = len(self.comps)
            for c in range(d):
                lab = f"{name}{COMP_LETTERS[c]}" if d > 1 else str(name)
                self.state_of[lab] = (name, irrep, c)
                self.comps.append(lab)
        self.states = list(states)
        self.index = {c: i for i, c in enumerate(self.comps)}

    def owns(self, atom):
        return atom[0] == "e"

    def dim(self, tj):
        return len(self.comps)

    def labels(self, tj):
        return [f"el={c}" for c in self.comps]

    def atom(self, a, tj):
        M = sp.lil_matrix((len(self.comps), len(self.comps)), dtype=complex)
        M[self.index[a[1]], self.index[a[2]]] = 1.0
        return M.tocsr()

    def U(self, gi, tj):
        n = len(self.comps)
        M = np.zeros((n, n), complex)
        for name, irrep in self.states:
            o = self.offset[name]
            r = self.g.irreps[irrep][gi]
            M[o:o + r.shape[0], o:o + r.shape[0]] = r
        return _csr(M)

    def theta(self, tj):
        return sp.identity(len(self.comps), dtype=complex, format="csr")

    def char(self, gi, tj, sector=None):
        return sum(self.g.chars[irrep][gi] for _, irrep in self.states)


class _Torsion(_Space):
    kind = "torsion"

    def __init__(self, group, decl: TorsionSpace):
        self.g, self.name, self.M = group, decl.name, int(decl.m_max)
        self.kappa = 2 if decl.period_pi == 4 else 1
        self.m = np.arange(-self.M, self.M + 1)

    def owns(self, atom):
        return atom[0] in ("p", "f", "q") and atom[-1] == self.name

    def dim(self, tj):
        return len(self.m)

    def labels(self, tj):
        return [f"m{self.name}={mm / self.kappa:g}" for mm in self.m]

    def atom(self, a, tj):
        n = len(self.m)
        if a[0] == "p":
            return _csr(np.diag(self.m / self.kappa))
        if a[0] == "q":
            raise GrammarError(f"q[{self.name}] is not defined on a periodic coordinate; use cos/sin")
        _, fn, k, _ = a
        s = k * self.kappa
        if abs(s - round(s)) > 1e-12:
            raise GrammarError(f"frequency {k} not representable on coordinate {self.name} (period "
                               f"{2 * self.kappa}pi)")
        s = int(round(s))
        shift = sp.eye(n, n, -s, dtype=complex, format="csr")     # |m+s><m|
        if fn == "cos":
            return (shift + shift.T) / 2
        return (shift - shift.T) / 2j

    def U(self, gi, tj):
        s, c = self.g.elements[gi].coord(self.name)
        n = len(self.m)
        rows = (s * self.m) + self.M
        vals = np.exp(1j * (self.m / self.kappa) * c)
        return sp.csr_matrix((vals, (rows, np.arange(n))), shape=(n, n))

    def theta(self, tj):
        n = len(self.m)
        return sp.csr_matrix((np.ones(n, complex), (n - 1 - np.arange(n), np.arange(n))), shape=(n, n))

    def char(self, gi, tj, sector=None):
        return complex(self.U(gi, tj).diagonal().sum())


class _Osc(_Space):
    kind = "osc"

    def __init__(self, group, decl: OscSpace):
        self.g, self.name, self.n = group, decl.name, int(decl.nmax)
        a = np.diag(np.sqrt(np.arange(1, self.n + 1)), 1)
        self.q = (a + a.T) / math.sqrt(2)
        self.p = 1j * (a.T - a) / math.sqrt(2)

    def owns(self, atom):
        return atom[0] in ("p", "q", "f") and atom[-1] == self.name

    def dim(self, tj):
        return self.n + 1

    def labels(self, tj):
        return [f"v{self.name}={v}" for v in range(self.n + 1)]

    def atom(self, a, tj):
        if a[0] == "q":
            return _csr(self.q)
        if a[0] == "p":
            return _csr(self.p)
        raise GrammarError(f"cos/sin not defined on oscillator coordinate {self.name}")

    def U(self, gi, tj):
        s, c = self.g.elements[gi].coord(self.name)
        if abs(c) > 1e-12:
            raise SymmetryError(f"oscillator coordinate {self.name} can only be reflected")
        return _csr(np.diag([float(s) ** v for v in range(self.n + 1)]))

    def theta(self, tj):
        return sp.identity(self.n + 1, dtype=complex, format="csr")

    def char(self, gi, tj, sector=None):
        s, _ = self.g.elements[gi].coord(self.name)
        return float(sum(float(s) ** v for v in range(self.n + 1)))


class _Well(_Space):
    kind = "well"

    def __init__(self, group, decl: WellSpace):
        self.g, self.name = group, decl.name
        V = np.array(decl.vectors, float)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        self.n = len(V)
        self.perm, self.sign = [], []
        for gi in range(group.order):
            rho = np.real(group.rep(decl.irrep)[gi])
            pm, sg = [], []
            for v in V:
                w = rho @ v
                d = V @ w
                m = int(np.argmax(np.abs(d)))
                if abs(abs(d[m]) - 1) > 1e-6:
                    raise SymmetryError(f"well vectors are not permuted by {decl.irrep} (element {gi})")
                pm.append(m)
                sg.append(float(np.sign(d[m])) if decl.geometric_phase else 1.0)
            self.perm.append(pm)
            self.sign.append(sg)

    def owns(self, atom):
        return atom[0] == "w"

    def dim(self, tj):
        return self.n

    def labels(self, tj):
        return [f"well={k + 1}" for k in range(self.n)]

    def atom(self, a, tj):
        M = sp.lil_matrix((self.n, self.n), dtype=complex)
        M[a[1] - 1, a[2] - 1] = 1.0
        return M.tocsr()

    def U(self, gi, tj):
        return sp.csr_matrix((np.array(self.sign[gi], complex), (self.perm[gi], np.arange(self.n))),
                             shape=(self.n, self.n))

    def theta(self, tj):
        return sp.identity(self.n, dtype=complex, format="csr")

    def char(self, gi, tj, sector=None):
        return float(sum(s for k, (m, s) in enumerate(zip(self.perm[gi], self.sign[gi])) if m == k))


def _sym_power_columns(rho: np.ndarray, n: int, basis: list[tuple]) -> np.ndarray:
    """Matrix of Sym^n(rho) on orthonormal occupation states |nu> = prod (a_c^+)^nu_c / sqrt(nu_c!) |0>,
    with a_c^+ -> sum_c' rho_c'c a_c'^+."""
    d = rho.shape[0]
    idx = {b: i for i, b in enumerate(basis)}
    M = np.zeros((len(basis), len(basis)), complex)
    for j, nu in enumerate(basis):
        poly = {tuple([0] * d): 1.0 + 0j}
        norm = 1.0
        for c in range(d):
            norm *= math.factorial(nu[c])
            for _ in range(nu[c]):
                new = {}
                for mono, coef in poly.items():
                    for cp in range(d):
                        if rho[cp, c] == 0:
                            continue
                        m2 = list(mono)
                        m2[cp] += 1
                        m2 = tuple(m2)
                        new[m2] = new.get(m2, 0) + coef * rho[cp, c]
                poly = new
        for mono, coef in poly.items():
            if abs(coef) > 1e-15:
                M[idx[mono], j] += coef * math.sqrt(np.prod([math.factorial(k) for k in mono]) / norm)
    return M


class _Vib(_Space):
    kind = "vib"

    def __init__(self, group, decl: VibSpace):
        self.g, self.decl = group, decl
        if (decl.polyads is None) == (decl.nmax is None):
            raise GrammarError("VibSpace needs exactly one of polyads= or nmax=")
        self.modes = decl.modes
        self.osc, self.mode_of_osc, self.comps = [], [], {}
        for mi, m in enumerate(self.modes):
            if m.irrep not in group.irreps:
                raise GrammarError(f"mode {m.name}: irrep {m.irrep!r} not in group {group.name}")
            d = group.irreps[m.irrep][0].shape[0]
            letters = [COMP_LETTERS[c] for c in range(d)] if d > 1 else [""]
            self.comps[m.name] = letters
            for c in range(d):
                self.osc.append((m.name, letters[c]))
                self.mode_of_osc.append(mi)
        w = [self.modes[mi].weight for mi in self.mode_of_osc]
        cap = max(decl.polyads) if decl.polyads is not None else decl.nmax
        weights = w if decl.polyads is not None else [1] * len(w)
        states = []

        def rec(i, acc, used):
            if i == len(self.osc):
                states.append(tuple(acc))
                return
            k = 0
            while used + k * weights[i] <= cap:
                rec(i + 1, acc + [k], used + k * weights[i])
                k += 1
        rec(0, [], 0)
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}
        self.P = np.array([sum(wi * ni for wi, ni in zip(w, s)) for s in states])
        self._ucache = {}

    def owns(self, atom):
        return atom[0] == "b"

    def dim(self, tj):
        return len(self.states)

    def labels(self, tj):
        return ["n=" + ",".join(f"{m}{c}:{k}" for (m, c), k in zip(self.osc, s) if k) or "n=0"
                for s in self.states]

    def atom(self, a, tj):
        _, kind, mode, comp = a
        try:
            o = self.osc.index((mode, comp))
        except ValueError:
            raise GrammarError(f"unknown oscillator {mode}{comp}") from None
        rows, cols, vals = [], [], []
        for j, s in enumerate(self.states):
            if s[o] == 0:
                continue
            t = list(s)
            t[o] -= 1
            i = self.index.get(tuple(t))
            if i is not None:
                rows.append(i)
                cols.append(j)
                vals.append(math.sqrt(s[o]))
        n = len(self.states)
        A = sp.csr_matrix((np.array(vals, complex), (rows, cols)), shape=(n, n))
        return A if kind == "a" else A.T.tocsr()

    def _mode_slices(self):
        out, k = [], 0
        for m in self.modes:
            d = len(self.comps[m.name])
            out.append(slice(k, k + d))
            k += d
        return out

    def U(self, gi, tj):
        key = gi
        if key in self._ucache:
            return self._ucache[key]
        slices = self._mode_slices()
        per_mode = {}
        rows, cols, vals = [], [], []
        for j, s in enumerate(self.states):
            parts = []
            for mi, sl in enumerate(slices):
                nu = s[sl]
                n = sum(nu)
                k = (mi, n)
                if k not in per_mode:
                    d = sl.stop - sl.start
                    basis = [b for b in itertools.product(range(n + 1), repeat=d) if sum(b) == n]
                    rho = self.g.irreps[self.modes[mi].irrep][gi]
                    per_mode[k] = (basis, {b: i for i, b in enumerate(basis)}, _sym_power_columns(rho, n, basis))
                basis, bidx, M = per_mode[k]
                col = M[:, bidx[tuple(nu)]]
                parts.append([(basis[i], col[i]) for i in np.nonzero(np.abs(col) > 1e-14)[0]])
            for combo in itertools.product(*parts):
                t = tuple(x for b, _ in combo for x in b)
                amp = np.prod([c for _, c in combo])
                rows.append(self.index[t])
                cols.append(j)
                vals.append(amp)
        n = len(self.states)
        U = sp.csr_matrix((np.array(vals, complex), (rows, cols)), shape=(n, n))
        self._ucache[key] = U
        return U

    def theta(self, tj):
        return sp.identity(len(self.states), dtype=complex, format="csr")

    def char(self, gi, tj, sector=None):
        """Independent formula: sum over per-mode quanta n_m (with the sector constraint) of
        prod_m h_{n_m}(eigenvalues of rho_m(g))."""
        lams = [np.linalg.eigvals(self.g.irreps[m.irrep][gi]) for m in self.modes]
        ws = [m.weight for m in self.modes]
        cap = max(self.decl.polyads) if self.decl.polyads is not None else self.decl.nmax
        poly = self.decl.polyads is not None
        total = 0
        ranges = [range(cap // (w if poly else 1) + 1) for w in ws]
        for ns in itertools.product(*ranges):
            P = sum(w * n for w, n in zip(ws, ns)) if poly else sum(ns)
            if P > cap:
                continue
            if sector is not None and P != sector:
                continue
            total += np.prod([complete_homogeneous(l, n) for l, n in zip(lams, ns)])
        return total


# ---- grammar --------------------------------------------------------------------------------

_TOK = re.compile(r"\s*(sym\(|h\.c\.|w\[\d+,\d+\]|e\[[^\]]*\]|Pe\[[^\]]*\]|(?:cos|sin)\d+(?:\.\d+)?\[\w+\]|[pq]\[\w+\]|"
                  r"[JSN][xyz2]|a\d+[a-z]?\+?|[qp]\d+[a-z]?|n\d+|\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|"
                  r"I|i|[()*+\-/])")


@dataclass
class BTerm:
    param: str
    text: str
    poly: dict
    hermitian: bool
    theta_even: bool
    order: int


def poly_add(a: dict, b: dict, s: complex = 1.0) -> dict:
    out = dict(a)
    for w, c in b.items():
        out[w] = out.get(w, 0) + s * c
    return {w: c for w, c in out.items() if abs(c) > 1e-13}


def poly_mul(a: dict, b: dict) -> dict:
    out = {}
    for w1, c1 in a.items():
        for w2, c2 in b.items():
            w = w1 + w2
            out[w] = out.get(w, 0) + c1 * c2
    return {w: c for w, c in out.items() if abs(c) > 1e-13}


def dagger_atom(a):
    if a[0] == "b":
        return ("b", "a" if a[1] == "c" else "c", a[2], a[3])
    if a[0] in ("e", "w"):
        return (a[0], a[2], a[1])
    return a


def dagger(p: dict) -> dict:
    out = {}
    for w, c in p.items():
        w2 = tuple(dagger_atom(a) for a in reversed(w))
        out[w2] = out.get(w2, 0) + np.conj(c)
    return out


def theta_conj(p: dict) -> dict:
    """Theta O Theta^-1 symbolically: J, S, p odd; complex coefficients conjugated."""
    out = {}
    for w, c in p.items():
        sgn = (-1) ** sum(1 for a in w if a[0] in ("J", "S", "p"))
        out[w] = np.conj(c) * sgn
    return out


def _normal_order_bosons(word: tuple) -> dict:
    """Normal order a word of boson atoms; returns {word: coef} with sorted creators then
    sorted annihilators."""
    for k in range(len(word) - 1):
        x, y = word[k], word[k + 1]
        if x[1] == "a" and y[1] == "c":
            swapped = word[:k] + (y, x) + word[k + 2:]
            out = _normal_order_bosons(swapped)
            if x[2:] == y[2:]:
                for w2, c2 in _normal_order_bosons(word[:k] + word[k + 2:]).items():
                    out[w2] = out.get(w2, 0) + c2
            return out
    cs = tuple(sorted(a for a in word if a[1] == "c"))
    an = tuple(sorted(a for a in word if a[1] == "a"))
    return {cs + an: 1.0}


def _close(a: dict, b: dict, tol=1e-10) -> bool:
    keys = set(a) | set(b)
    scale = max([abs(v) for v in a.values()] + [1e-300])
    return all(abs(a.get(k, 0) - b.get(k, 0)) <= tol * scale for k in keys)


class Context:
    """Atom vocabulary and group action for a list of runtime spaces."""

    def __init__(self, group: Group, spaces: list):
        self.g, self.spaces = group, spaces
        self.vib = next((s for s in spaces if s.kind == "vib"), None)
        self.rot = any(s.kind == "rot" for s in spaces)
        self.spin = any(s.kind == "spin" for s in spaces)
        self.elec = next((s for s in spaces if s.kind == "elec"), None)
        self.well = next((s for s in spaces if s.kind == "well"), None)
        self.coords = {s.name: s.kind for s in spaces if s.kind in ("torsion", "osc")}
        self.mode_irrep = {m.name: m.irrep for m in self.vib.modes} if self.vib else {}
        self._act = {}

    def space_of(self, atom) -> int:
        for i, s in enumerate(self.spaces):
            if s.owns(atom):
                return i
        raise GrammarError(f"no space carries atom {atom}")

    # group action on a single atom -> [(atom, coef)]
    def act(self, gi: int, atom):
        key = (gi, atom)
        if key in self._act:
            return self._act[key]
        el = self.g.elements[gi]
        t = atom[0]
        if t == "b":
            letters = self.vib.comps[atom[2]]
            rho = self.g.irreps[self.mode_irrep[atom[2]]][gi]
            c = letters.index(atom[3])
            out = [(("b", atom[1], atom[2], letters[cp]), rho[cp, c]) for cp in range(len(letters))
                   if abs(rho[cp, c]) > 1e-14]
        elif t in ("J", "S"):
            a = "xyz".index(atom[1])
            out = [((t, "xyz"[b]), el.R[b, a]) for b in range(3) if abs(el.R[b, a]) > 1e-14]
        elif t == "e":
            out = []
            n1, ir1, c1 = self.elec.state_of[atom[1]]
            n2, ir2, c2 = self.elec.state_of[atom[2]]
            r1, r2 = self.g.irreps[ir1][gi], self.g.irreps[ir2][gi]
            labs1 = [l for l in self.elec.comps if self.elec.state_of[l][0] == n1]
            labs2 = [l for l in self.elec.comps if self.elec.state_of[l][0] == n2]
            for i1, l1 in enumerate(labs1):
                for i2, l2 in enumerate(labs2):
                    v = r1[i1, c1] * np.conj(r2[i2, c2])
                    if abs(v) > 1e-14:
                        out.append((("e", l1, l2), v))
        elif t == "w":
            wl = self.well
            i, j = atom[1] - 1, atom[2] - 1
            out = [(("w", wl.perm[gi][i] + 1, wl.perm[gi][j] + 1), wl.sign[gi][i] * wl.sign[gi][j])]
        elif t == "f":
            _, fn, k, name = atom
            s, c = el.coord(name)
            ck, sk = math.cos(k * c), math.sin(k * c)
            if fn == "cos":
                out = [(("f", "cos", k, name), ck), (("f", "sin", k, name), -s * sk)]
            else:
                out = [(("f", "sin", k, name), s * ck), (("f", "cos", k, name), sk)]
            out = [(a, v) for a, v in out if abs(v) > 1e-14]
        elif t in ("p", "q"):
            s, _ = el.coord(atom[1])
            out = [(atom, float(s))]
        else:
            raise GrammarError(f"unknown atom {atom}")
        self._act[key] = out
        return out

    def canon(self, p: dict) -> dict:
        """Canonical form: atoms of different spaces commute (stable sort by space); boson
        words normal ordered ([a_i, a_j^+] = delta_ij, sorted); electronic products reduced."""
        out = {}
        for w, c in p.items():
            segs = {}
            for a in w:
                segs.setdefault(self.space_of(a), []).append(a)
            parts = []
            for si in sorted(segs):
                seg = segs[si]
                if seg[0][0] == "b":
                    parts.append(_normal_order_bosons(tuple(seg)))
                elif seg[0][0] in ("e", "w"):
                    cur = seg[0]
                    ok = True
                    for nx in seg[1:]:
                        if cur[2] != nx[1]:
                            ok = False
                            break
                        cur = (cur[0], cur[1], nx[2])
                    parts.append({(cur,): 1.0} if ok else {})
                else:
                    parts.append({tuple(seg): 1.0})
            acc = {(): c}
            for part in parts:
                acc = poly_mul(acc, part)
            for w2, c2 in acc.items():
                out[w2] = out.get(w2, 0) + c2
        return {w: c for w, c in out.items() if abs(c) > 1e-13}

    def act_poly(self, gi: int, p: dict) -> dict:
        out = {}
        for w, c in p.items():
            for combo in itertools.product(*[self.act(gi, a) for a in w]):
                ww = tuple(a for a, _ in combo)
                v = c * np.prod([x for _, x in combo]) if combo else c
                out[ww] = out.get(ww, 0) + v
        return {w: c for w, c in out.items() if abs(c) > 1e-13}

    def sym(self, p: dict) -> dict:
        out = {}
        for gi in range(self.g.order):
            out = poly_add(out, self.act_poly(gi, p))
        out = self.canon(out)
        return {w: c / self.g.order for w, c in out.items() if abs(c / self.g.order) > 1e-12}

    def invariance_defect(self, p: dict) -> float:
        scale = max([abs(c) for c in p.values()] + [1e-300])
        worst = 0.0
        for gen in self.g.generators:
            gi = self.g.find(gen)
            q = self.canon(self.act_poly(gi, p))
            keys = set(p) | set(q)
            worst = max(worst, max(abs(p.get(k, 0) - q.get(k, 0)) for k in keys) / scale if keys else 0)
        return worst


class Parser:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def parse_term(self, text: str) -> BTerm:
        if ":" not in text:
            raise GrammarError(f"term {text!r}: expected '<param>: <expr>'")
        param, expr = text.split(":", 1)
        param = param.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", param):
            raise GrammarError(f"bad parameter name {param!r}")
        poly = self.ctx.canon(self.parse_expr(expr))
        if not poly:
            raise GrammarError(f"term {param}: operator vanishes")
        herm = _close(poly, self.ctx.canon(dagger(poly)))
        teven = _close(poly, self.ctx.canon(theta_conj(poly)))
        order = max(len(w) for w in poly)
        return BTerm(param, text.strip(), poly, herm, teven, order)

    def parse_expr(self, expr: str) -> dict:
        toks, pos = [], 0
        s = expr.strip()
        while pos < len(s):
            m = _TOK.match(s, pos)
            if not m or m.end() == pos:
                raise GrammarError(f"cannot parse {s[pos:]!r}")
            toks.append(m.group(1))
            pos = m.end()
        self.toks, self.i = toks, 0
        p = self._expr(top=True)
        if self.i != len(self.toks):
            raise GrammarError(f"unexpected {' '.join(self.toks[self.i:])!r}")
        return p

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _expr(self, top=False):
        sgn0 = 1
        while self._peek() in ("+", "-"):         # leading sign: "-sym(...)", "(-Jx + Jy)"
            sgn0 *= -1 if self.toks[self.i] == "-" else 1
            self.i += 1
        p = self._product()
        if sgn0 == -1:
            p = poly_add({}, p, -1)
        while self._peek() in ("+", "-"):
            sgn = 1 if self.toks[self.i] == "+" else -1
            self.i += 1
            if self._peek() == "h.c.":
                if not top or sgn != 1:
                    raise GrammarError("'+ h.c.' only at the end of a term")
                self.i += 1
                if self.i != len(self.toks):
                    raise GrammarError("'+ h.c.' must end the term")
                return poly_add(p, dagger(p))
            if self._peek() is None:
                raise GrammarError("dangling operator")
            p = poly_add(p, self._product(), sgn)
        return p

    def _product(self):
        factors = []
        while True:
            t = self._peek()
            if t is None or t in ("+", "-", ")", "h.c."):
                break
            if t == "*":
                self.i += 1
                continue
            if t == "/":
                self.i += 1
                d = self._peek()
                if d is None or not re.fullmatch(r"[\d.eE+-]+", d):
                    raise GrammarError("division only by a number")
                self.i += 1
                factors.append({(): 1.0 / float(d)})
                continue
            factors.append(self._factor())
        if not factors:
            here = " ".join(self.toks[max(0, self.i - 3):self.i + 3])
            raise GrammarError(f"empty product near {here!r} (a dangling '*', an empty '()' or an operator "
                               "with nothing after it)")
        p = factors[0]
        for f in factors[1:]:
            p = poly_mul(p, f)
        return p

    def _factor(self):
        t = self.toks[self.i]
        self.i += 1
        ctx = self.ctx
        if t == "sym(":
            p = self._expr()
            self._close_paren()
            return ctx.sym(p)
        if t == "(":
            p = self._expr()
            self._close_paren()
            return p
        if t == "I":
            return {(): 1.0}
        if t == "i":
            return {(): 1j}
        if re.fullmatch(r"\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?", t):
            return {(): float(t)}
        if t[0] in "JSN" and len(t) == 2:
            if t[0] == "J" and not ctx.rot or t[0] == "S" and not ctx.spin:
                raise GrammarError(f"{t}: no {'rotation' if t[0] == 'J' else 'spin'} space")
            if t[0] == "N" and not (ctx.rot and ctx.spin):
                raise GrammarError(f"{t}: N = J - S needs rotation and spin spaces")

            def comp(L, a):
                if L == "N":
                    return poly_add({(("J", a),): 1.0}, {(("S", a),): 1.0}, -1)
                return {((L, a),): 1.0}
            if t[1] == "2":
                out = {}
                for a in "xyz":
                    c = comp(t[0], a)
                    out = poly_add(out, poly_mul(c, c))
                return out
            return comp(t[0], t[1])
        if t.startswith("e["):
            if ctx.elec is None:
                raise GrammarError(f"{t}: no electronic space")
            a, b = [x.strip() for x in t[2:-1].split(",")]
            for x in (a, b):
                if x not in ctx.elec.index:
                    raise GrammarError(f"unknown electronic component {x!r} (have {ctx.elec.comps})")
            return {(("e", a, b),): 1.0}
        if t.startswith("w["):
            if ctx.well is None:
                raise GrammarError(f"{t}: no well space")
            i, j = (int(x) for x in t[2:-1].split(","))
            if not (1 <= i <= ctx.well.n and 1 <= j <= ctx.well.n):
                raise GrammarError(f"{t}: wells are 1..{ctx.well.n}")
            return {(("w", i, j),): 1.0}
        if t.startswith("Pe["):
            st = t[3:-1].strip()
            labs = [l for l in ctx.elec.comps if ctx.elec.state_of[l][0] == st] if ctx.elec else []
            if not labs:
                raise GrammarError(f"unknown electronic state {st!r}")
            return {(("e", l, l),): 1.0 for l in labs}
        m = re.fullmatch(r"(cos|sin)(\d+(?:\.\d+)?)\[(\w+)\]", t)
        if m:
            fn, k, name = m.group(1), float(m.group(2)), m.group(3)
            if ctx.coords.get(name) != "torsion":
                raise GrammarError(f"{t}: no periodic coordinate {name!r}")
            if k == 0:
                return {(): 1.0} if fn == "cos" else {}
            return {(("f", fn, k, name),): 1.0}
        m = re.fullmatch(r"([pq])\[(\w+)\]", t)
        if m:
            if m.group(2) not in ctx.coords:
                raise GrammarError(f"{t}: no coordinate {m.group(2)!r}")
            return {((m.group(1), m.group(2)),): 1.0}
        m = re.fullmatch(r"a(\d+)([a-z]?)(\+?)", t)
        if m:
            return {(("b", "c" if m.group(3) else "a", m.group(1), self._comp(m.group(1), m.group(2))),): 1.0}
        m = re.fullmatch(r"([qp])(\d+)([a-z]?)", t)
        if m:
            mode, c = m.group(2), self._comp(m.group(2), m.group(3))
            cr, an = {(("b", "c", mode, c),): 1.0}, {(("b", "a", mode, c),): 1.0}
            r2 = 1 / math.sqrt(2)
            if m.group(1) == "q":
                return poly_add({w: v * r2 for w, v in cr.items()}, an, r2)
            return poly_add({w: 1j * r2 * v for w, v in cr.items()}, an, -1j * r2)
        m = re.fullmatch(r"n(\d+)", t)
        if m:
            mode = m.group(1)
            if mode not in ctx.mode_irrep:
                raise GrammarError(f"unknown mode {mode}")
            return {(("b", "c", mode, c), ("b", "a", mode, c)): 1.0 for c in ctx.vib.comps[mode]}
        raise GrammarError(f"unexpected token {t!r}")

    def _comp(self, mode, c):
        if mode not in self.ctx.mode_irrep:
            raise GrammarError(f"unknown mode {mode}")
        letters = self.ctx.vib.comps[mode]
        if c == "" and letters == [""]:
            return ""
        if c not in letters:
            raise GrammarError(f"mode {mode} has components {letters}, not {c!r}")
        return c

    def _close_paren(self):
        if self._peek() != ")":
            raise GrammarError("missing ')'")
        self.i += 1


# ---- compile -----------------------------------------------------------------------------------

def _needs_double(spec: BuildSpec) -> bool:
    half_j = any(abs(j % 1 - 0.5) < 1e-9 for j in spec.J)
    half_s = any(s.kind == "spin" and s.two_s % 2 for s in spec.spaces)
    return half_j or half_s


def make_group(spec: BuildSpec) -> Group:
    if isinstance(spec.group, Group):
        return spec.group
    return get_group(spec.group, double=_needs_double(spec))


def runtime_spaces(spec: BuildSpec, group: Group) -> list:
    out = []
    decls = sorted(spec.spaces, key=lambda s: {"rot": 0, "spin": 1}.get(s.kind, 2))
    for d in decls:
        if d.kind == "rot":
            out.append(_Rot(group))
        elif d.kind == "spin":
            out.append(_Spin(group, d.two_s))
        elif d.kind == "elec":
            out.append(_Elec(group, d.states))
        elif d.kind == "vib":
            out.append(_Vib(group, d))
        elif d.kind == "torsion":
            out.append(_Torsion(group, d))
        elif d.kind == "osc":
            out.append(_Osc(group, d))
        elif d.kind == "well":
            out.append(_Well(group, d))
    kinds = [s.kind for s in out]
    for k in ("rot", "spin", "elec", "vib", "well"):
        if kinds.count(k) > 1:
            raise GrammarError(f"at most one {k} space")
    return out


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out.tocsr()


class Operators:
    """Full product-space matrices for one J."""

    def __init__(self, ctx: Context, tj: int):
        self.ctx, self.tj = ctx, tj
        self.dims = [s.dim(tj) for s in ctx.spaces]
        self.N = int(np.prod(self.dims))
        self._atoms = {}

    def atom(self, si, a):
        k = (si, a)
        if k not in self._atoms:
            self._atoms[k] = self.ctx.spaces[si].atom(a, self.tj)
        return self._atoms[k]

    def matrix(self, poly: dict) -> sp.csr_matrix:
        groups = {}
        for w, c in poly.items():
            per = [[] for _ in self.ctx.spaces]
            for a in w:
                per[self.ctx.space_of(a)].append(a)
            key = tuple(tuple(x) for x in per)
            groups[key] = groups.get(key, 0) + c
        total = sp.csr_matrix((self.N, self.N), dtype=complex)
        for key, c in groups.items():
            if abs(c) < 1e-14:
                continue
            mats = []
            for si, atoms in enumerate(key):
                if not atoms:
                    mats.append(sp.identity(self.dims[si], dtype=complex, format="csr"))
                else:
                    m = self.atom(si, atoms[0])
                    for a in atoms[1:]:
                        m = m @ self.atom(si, a)
                    mats.append(m)
            total = total + c * _kron_all(mats)
        return total.tocsr()

    def U(self, gi):
        return _kron_all([s.U(gi, self.tj) for s in self.ctx.spaces])

    def theta(self):
        return _kron_all([s.theta(self.tj) for s in self.ctx.spaces])

    def labels(self):
        per = [s.labels(self.tj) for s in self.ctx.spaces]
        return [" ".join(x) for x in itertools.product(*per)]

    def sector_values(self):
        vals = np.zeros(self.N, dtype=int)
        si = next((i for i, s in enumerate(self.ctx.spaces) if s.kind == "vib"), None)
        if si is None or self.ctx.spaces[si].decl.polyads is None:
            return None
        mats = [np.ones(d) for d in self.dims]
        mats[si] = self.ctx.spaces[si].P
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out.astype(int)


def _fro(M) -> float:
    return float(sp.linalg.norm(M)) if sp.issparse(M) else float(np.linalg.norm(M))


@dataclass
class BuiltBlock:
    two_j: int
    irrep: str
    polyad: int | None
    W: np.ndarray            # columns: real (time-reversal-fixed) basis in the full product basis
    labels: list
    kramers: bool = False    # Theta^2 = -1: complex block stored realified [[Re, -Im], [Im, Re]]


class BuiltModel:
    """A compiled CompiledModel plus what is needed to map back to the product basis."""

    def __init__(self, spec, group, ctx, terms, model, blocks, certs, counts):
        self.spec, self.group, self.ctx, self.terms = spec, group, ctx, terms
        self.model, self.blocks, self.certificates, self.counts = model, blocks, certs, counts

    @property
    def param_names(self):
        return list(self.model.param_names)

    def block_matrices(self, theta) -> list[np.ndarray]:
        vec = [float(theta[n]) for n in self.model.param_names] if isinstance(theta, dict) else \
            [float(x) for x in theta]
        out = []
        for b in range(len(self.blocks)):
            ip, ix, dt, shape = self.model.assemble(b, vec)
            out.append(sp.csr_matrix((dt, ix, ip), shape=tuple(shape)).toarray())
        return out

    def eig(self, theta):
        """Dense eigen-decomposition of every block: list of (energies, vectors in block basis)."""
        out = []
        for blk, H in zip(self.blocks, self.block_matrices(theta)):
            w, v = np.linalg.eigh((H + H.T) / 2)
            if blk.kramers:          # each level appears twice in the realified block
                n = v.shape[0] // 2
                w = w[::2]
                v = v[:n, ::2] + 1j * v[n:, ::2]
                v = v / np.linalg.norm(v, axis=0)
            out.append((w, v))
        return out

    def full_vectors(self, theta):
        return [(w, self.blocks[b].W @ v) for b, (w, v) in enumerate(self.eig(theta))]

    def operator_blocks(self, poly_or_text) -> list[np.ndarray]:
        """Matrix of any operator (text or poly) in every block basis (complex)."""
        poly = Parser(self.ctx).parse_expr(poly_or_text) if isinstance(poly_or_text, str) else poly_or_text
        cache, out = {}, []
        for b in self.blocks:
            if b.two_j not in cache:
                cache[b.two_j] = Operators(self.ctx, b.two_j).matrix(poly)
            M = cache[b.two_j]
            out.append(b.W.conj().T @ (M @ b.W))
        return out


def _dense_to_csr(A: np.ndarray, tol: float = 1e-13):
    A = np.where(np.abs(A) > tol * max(1.0, np.max(np.abs(A)) if A.size else 1.0), A, 0.0)
    S = sp.csr_matrix(A)
    S.sort_indices()
    return (S.indptr.astype(np.int64), S.indices.astype(np.int32), S.data.astype(float), A.shape)


def compile_build(spec: BuildSpec, *, check_reps: bool = True, seed: int = 7) -> BuiltModel:
    """Compile a BuildSpec. Every certificate failure raises GrammarError (a compiler bug
    or an invalid term, never a statistical event)."""
    group = make_group(spec)
    spaces = runtime_spaces(spec, group)
    ctx = Context(group, spaces)
    parser = Parser(ctx)
    terms = [parser.parse_term(t) for t in spec.terms]
    names = [t.param for t in terms]
    if len(set(names)) != len(names):
        raise GrammarError("duplicate parameter names")
    has_rot = ctx.rot
    Js = [round(2 * j) for j in spec.J] if has_rot else [0]
    # odd-electron systems without rotation have Theta^2 = -1: complex Kramers blocks, stored realified
    certs = []

    def require(name, ok, value, bound, detail):
        certs.append({"name": name, "passed": bool(ok), "value": float(value), "bound": float(bound),
                      "detail": detail})
        if not ok:
            raise GrammarError(f"compile certificate '{name}' failed: {detail}")

    require("hermiticity", all(t.hermitian for t in terms), sum(not t.hermitian for t in terms), 0,
            f"non-Hermitian terms: {[t.param for t in terms if not t.hermitian]} (use '+ h.c.')")
    require("time_reversal", all(t.theta_even for t in terms), sum(not t.theta_even for t in terms), 0,
            f"time-reversal-odd terms: {[t.param for t in terms if not t.theta_even]} "
            "(J, S, p are odd; use i for odd products)")
    defects = {t.param: ctx.invariance_defect(t.poly) for t in terms}
    bad = [k for k, v in defects.items() if v > 1e-9]
    require("total_symmetry", not bad, max(defects.values(), default=0), 1e-9,
            f"terms not invariant under {group.name}: {bad} (use sym(...))")
    worst = max((t.order for t in terms), default=0)
    require("order_bound", worst <= spec.max_order, worst, spec.max_order, "max atoms per word")

    rng = np.random.default_rng(seed)
    gen_idx = [group.find(g) for g in group.generators]
    phys = [nm for nm in group.irreps if group.physical(nm)]
    blocks_out, built_blocks, counts = [], [], []
    for tj in Js:
        ops = Operators(ctx, tj)
        mats = [ops.matrix(t.poly) for t in terms]
        Us = {gi: ops.U(gi) for gi in range(group.order)}
        TH = ops.theta()
        if check_reps:
            _check_representation(ctx, ops, Us, gen_idx, tj, require, rng)
        # Theta commutes with the group
        dmax = max((_fro(TH @ Us[gi].conj() - Us[gi] @ TH) for gi in gen_idx), default=0.0)
        require(f"theta_commutes_J2={tj}", dmax < 1e-8, dmax, 1e-8, "time reversal commutes with every generator")
        th2p = _fro(TH @ TH.conj() - sp.identity(ops.N, format="csr"))
        th2m = _fro(TH @ TH.conj() + sp.identity(ops.N, format="csr"))
        kramers = th2m < 1e-8
        require(f"theta_squared_J2={tj}", min(th2p, th2m) < 1e-8, min(th2p, th2m), 1e-8,
                "time reversal squares to +1 (real blocks) or -1 (Kramers: complex blocks, realified)")
        for t, M in zip(terms, mats):
            if _fro(M) < 1e-10 * (1 + tj) ** t.order:
                continue          # the operator vanishes at this J (e.g. a K+-3 term at J = 1)
            h = _fro(M - M.conj().T) / max(_fro(M), 1e-300)
            sym_def = max((_fro(Us[gi] @ M @ Us[gi].conj().T - M) for gi in gen_idx), default=0.0) / max(_fro(M), 1e-300)
            te = _fro(TH @ M.conj() @ TH.conj().T - M) / max(_fro(M), 1e-300)
            if h > 1e-9 or sym_def > 1e-9 or te > 1e-9:
                require(f"matrix_certificates_{t.param}_J2={tj}", False, max(h, sym_def, te), 1e-9,
                        f"term {t.param} at 2J={tj}: hermiticity {h:.1e}, symmetry {sym_def:.1e}, "
                        f"time reversal {te:.1e}")
        sectors = ops.sector_values()
        if sectors is not None:
            Pd = sp.diags(sectors.astype(complex))
            for t, M in zip(terms, mats):
                c = _fro(Pd @ M - M @ Pd)
                if c > 1e-9 * max(_fro(M), 1):
                    require("polyad_conservation", False, c, 1e-9, f"term {t.param} changes the polyad number")
            sector_list = sorted(set(sectors.tolist()) & set(spec.spaces[[s.kind for s in spec.spaces].index("vib")]
                                                              .polyads))
        else:
            sector_list = [None]
        labels_all = ops.labels()
        for P in sector_list:
            idx = np.arange(ops.N) if P is None else np.nonzero(sectors == P)[0]
            if len(idx) == 0:
                continue
            chi_sec = np.array([np.prod([s.char(gi, tj, P if s.kind == "vib" else None) for s in ctx.spaces])
                                for gi in range(group.order)])
            dec = group.decompose(chi_sec)
            total = sum(n * group.irreps[g][0].shape[0] for g, n in dec.items())
            require(f"sector_dimension_J2={tj}_P={P}", total == len(idx), total, len(idx),
                    "character of the product representation reproduces the sector dimension")
            done = set()
            for gname in phys:
                n = dec.get(gname, 0)
                if n == 0 or gname in done:
                    continue
                fs = group.frobenius_schur(gname)
                if fs == -1 and not kramers:
                    raise GrammarError(f"quaternionic irrep {gname} with Theta^2=+1 is not supported")
                partners = [gname] if (fs == 1 or kramers) else [gname, group.conjugate(gname)]
                if kramers and fs == 0:
                    done.add(group.conjugate(gname))     # Gamma* is the Kramers partner: same levels
                done.update(partners)
                Bs = []
                for pi, pn in enumerate(partners):
                    rho = group.irreps[gname] if pi == 0 else [np.conj(m) for m in group.irreps[gname]]
                    d = rho[0].shape[0]
                    Z = rng.standard_normal((len(idx), n + 6)) + 1j * rng.standard_normal((len(idx), n + 6))
                    Y = np.zeros_like(Z)
                    for gi in range(group.order):
                        c = np.conj(rho[gi][0, 0])
                        if abs(c) > 1e-14:
                            Y += c * (Us[gi][idx][:, idx] @ Z)
                    Y *= d / group.order
                    Uy, sv, _ = np.linalg.svd(Y, full_matrices=False)
                    rank = int(np.sum(sv > 1e-8 * sv[0]))
                    require(f"count_J2={tj}_P={P}_{pn}", rank == n, rank, n,
                            f"projected dimension of {pn} equals the character count {n}")
                    Bs.append(Uy[:, :n])
                B = np.hstack(Bs)
                Bf = np.zeros((ops.N, B.shape[1]), complex)
                Bf[idx] = B
                if kramers:
                    W = Bf
                    block_ops = []
                    for M in mats:
                        A = W.conj().T @ (M @ W)
                        A = (A + A.conj().T) / 2
                        R = np.block([[A.real, -A.imag], [A.imag, A.real]])
                        block_ops.append(_dense_to_csr(R))
                    lab = [labels_all[int(np.argmax(np.abs(W[:, k])))] for k in range(W.shape[1])]
                    lab = lab + [x + " (Im)" for x in lab]
                    blocks_out.append({"two_j": tj, "irrep": gname, "spin_species": "", "electronic_state": 0,
                                       "sub": ("" if P is None else f"P={P};") + "kramers=1", "basis_labels": lab, "ops": block_ops})
                    built_blocks.append(BuiltBlock(tj, gname, P, W, lab[:W.shape[1]], kramers=True))
                    counts.append({"two_j": tj, "polyad": P, "irrep": gname, "count": int(W.shape[1])})
                    continue
                TB = TH @ Bf.conj()
                C = np.hstack([Bf + TB, 1j * (Bf - TB)])
                G = (C.conj().T @ C).real
                w, E = np.linalg.eigh(G)
                keep = w > 1e-8 * w[-1]
                require(f"real_basis_J2={tj}_P={P}_{gname}", int(keep.sum()) == B.shape[1], int(keep.sum()),
                        B.shape[1], "time-reversal-fixed real basis has the block dimension")
                W = C @ (E[:, keep] / np.sqrt(w[keep]))
                block_ops, imag = [], 0.0
                for M in mats:
                    A = W.conj().T @ (M @ W)
                    imag = max(imag, float(np.max(np.abs(A.imag))) / max(1.0, float(np.max(np.abs(A)))))
                    block_ops.append(_dense_to_csr((A.real + A.real.T) / 2))
                require(f"real_blocks_J2={tj}_P={P}_{gname}", imag < 1e-9, imag, 1e-9,
                        "block matrices are real in the time-reversal basis")
                lab = [labels_all[int(np.argmax(np.abs(W[:, k])))] for k in range(W.shape[1])]
                irrep_label = gname if len(partners) == 1 else f"{partners[0]}+{partners[1]}"
                blocks_out.append({"two_j": tj, "irrep": irrep_label, "spin_species": "", "electronic_state": 0,
                                   "sub": "" if P is None else f"P={P}", "basis_labels": lab, "ops": block_ops})
                built_blocks.append(BuiltBlock(tj, irrep_label, P, W, lab))
                counts.append({"two_j": tj, "polyad": P, "irrep": irrep_label, "count": int(W.shape[1])})
    if not blocks_out:
        raise GrammarError("no physical blocks (check the group, J values and spaces)")
    # linear independence of the terms over all blocks
    if terms:
        cols = []
        for k in range(len(terms)):
            v = np.concatenate([_block_dense(b["ops"][k]).ravel() for b in blocks_out])
            cols.append(v)
        A = np.array(cols).T
        sv = np.linalg.svd(A, compute_uv=False) if A.size else np.array([])
        rank = int(np.sum(sv > 1e-10 * (sv[0] if len(sv) else 1)))
        detail = "term matrices are linearly independent over the compiled blocks (no duplicate operators)"
        if rank < len(terms):
            # name the dependency: null vectors of the (column-normalized) term matrix
            nrm = np.linalg.norm(A, axis=0)
            _, s2, Vt = np.linalg.svd(A / np.where(nrm > 0, nrm, 1), full_matrices=True)
            rels = []
            for v in Vt[rank:]:
                c = v / np.where(nrm > 0, nrm, 1)
                c = c / np.max(np.abs(c))
                parts = [f"{c[k]:+.3g}*{terms[k].param}" for k in range(len(terms)) if abs(c[k]) > 1e-6]
                rels.append(" ".join(parts) + " = 0")
            zero = [terms[k].param for k in range(len(terms)) if nrm[k] < 1e-12]
            detail += ("; dependent: " + "; ".join(rels)
                       + (f" (terms {zero} vanish at the compiled J)" if zero else "")
                       + ". Remove one term of each relation.")
        require("independent_terms", rank == len(terms), rank, len(terms), detail)
    model = _core.CompiledModel.from_parts("built", spec.to_json(), names, blocks_out, certs)
    return BuiltModel(spec, group, ctx, terms, model, built_blocks, certs, counts)


def _block_dense(op):
    ip, ix, dt, shape = op
    return sp.csr_matrix((dt, ix, ip), shape=shape).toarray()


def _check_representation(ctx, ops, Us, gen_idx, tj, require, rng):
    """U(g) A U(g)^dagger = g.A for every generator and basic atom; homomorphism on generators."""
    worst = 0.0
    for si, s in enumerate(ctx.spaces):
        atoms = _basic_atoms(ctx, s)
        for gi in gen_idx:
            Ug = s.U(gi, tj)
            for a in atoms:
                A = s.atom(a, tj)
                lhs = Ug @ A @ Ug.conj().T
                rhs = sp.csr_matrix(A.shape, dtype=complex)
                for b, c in ctx.act(gi, a):
                    rhs = rhs + c * s.atom(b, tj)
                worst = max(worst, _fro(lhs - rhs) / max(_fro(A), 1e-300))
    tworst = 0.0
    for si, s in enumerate(ctx.spaces):
        T = s.theta(tj)
        for a in _basic_atoms(ctx, s):
            A = s.atom(a, tj)
            sgn = -1 if a[0] in ("J", "S", "p") else 1
            tworst = max(tworst, _fro(T @ A.conj() @ T.conj().T - sgn * A) / max(_fro(A), 1e-300))
    require(f"time_reversal_atoms_J2={tj}", tworst < 1e-8, tworst, 1e-8,
            "Theta J Theta^-1 = -J, Theta S Theta^-1 = -S, Theta p Theta^-1 = -p, other atoms even")
    require(f"representation_consistency_J2={tj}", worst < 1e-8, worst, 1e-8,
            "every space's U(g) implements the grammar's group action on its atoms")
    v = rng.standard_normal(ops.N) + 1j * rng.standard_normal(ops.N)
    g = ctx.g
    hom = 0.0
    for a in gen_idx:
        for b in range(g.order):
            hom = max(hom, float(np.linalg.norm(Us[a] @ (Us[b] @ v) - Us[int(g.table[a, b])] @ v)))
    require(f"representation_homomorphism_J2={tj}", hom < 1e-8 * np.linalg.norm(v), hom, 1e-8,
            "U(g) U(h) = U(gh) on the product space")


def _basic_atoms(ctx, s):
    if s.kind == "rot":
        return [("J", a) for a in "xyz"]
    if s.kind == "spin":
        return [("S", a) for a in "xyz"]
    if s.kind == "elec":
        return [("e", a, b) for a in s.comps for b in s.comps]
    if s.kind == "vib":
        return [("b", "a", m, c) for m, c in s.osc]
    if s.kind == "torsion":
        ks = [k / s.kappa for k in range(1, 3 * s.kappa + 1)]
        return [("p", s.name)] + [("f", fn, k, s.name) for k in ks for fn in ("cos", "sin")]
    if s.kind == "osc":
        return [("q", s.name), ("p", s.name)]
    if s.kind == "well":
        return [("w", i + 1, j + 1) for i in range(s.n) for j in range(s.n)]
    return []


# ---- spectra -------------------------------------------------------------------------------------

def line_strengths(bm: BuiltModel, theta, dipole: dict, *, min_strength: float = 1e-10,
                   y_sign: float = 1.0) -> list[dict]:
    """Line strengths S(f<-i) = (2J+1)(2J'+1) |sum_{k k' q} (-1)^k' (J' 1 J; -k' q k) <f,k'|mu_q|i,k>|^2
    (summed over M, M' and lab components). dipole = {"x": expr, "y": expr, "z": expr} with
    molecule-fixed components as grammar expressions on the non-rotational spaces (or numbers)."""
    if not bm.ctx.rot:
        raise GrammarError("line strengths need a rotation space")
    rest = bm.ctx.spaces[1:]
    rest_ctx = Context(bm.group, rest) if rest else None
    dims_rest = [s.dim(0) for s in rest]
    Nrest = int(np.prod(dims_rest)) if rest else 1

    def comp(expr):
        if isinstance(expr, (int, float)):
            return float(expr) * sp.identity(Nrest, dtype=complex, format="csr")
        poly = Parser(rest_ctx).parse_expr(expr)
        o = Operators(rest_ctx, 0)
        return o.matrix(poly)
    mx, my, mz = (comp(dipole.get(a, 0.0)) for a in "xyz")
    r2 = 1 / math.sqrt(2)
    mu = {-1: r2 * (mx - 1j * y_sign * my), 0: mz, 1: -r2 * (mx + 1j * y_sign * my)}
    vecs = bm.full_vectors(theta)
    by_j = {}
    for b, (w, V) in enumerate(vecs):
        by_j.setdefault(bm.blocks[b].two_j, []).append((b, w, V))
    from .angular import wigner3j
    lines = []
    tables = {}
    for tj, lowers in by_j.items():
        for tjp in (tj - 2, tj, tj + 2):
            if tjp not in by_j or tjp < 0:
                continue
            J, Jp = tj / 2, tjp / 2
            if (tjp, tj) not in tables:
                C = np.zeros((3, tjp + 1, tj + 1))
                for qi, q in enumerate((-1, 0, 1)):
                    for a in range(tjp + 1):
                        kp = (-tjp + 2 * a) / 2
                        for c in range(tj + 1):
                            k = (-tj + 2 * c) / 2
                            if abs(-kp + q + k) < 1e-9:
                                C[qi, a, c] = (-1) ** round(kp - (tjp % 2) / 2) * wigner3j(Jp, 1, J, -kp, q, k)
                tables[(tjp, tj)] = C
            C = tables[(tjp, tj)]
            for (bi, wi, Vi) in lowers:
                Xi = {q: None for q in (-1, 0, 1)}
                Ri = Vi.reshape(tj + 1, Nrest, -1)                      # (k, rest, level)
                for q in (-1, 0, 1):
                    Xi[q] = np.einsum("rs,ksl->krl", mu[q].toarray() if sp.issparse(mu[q]) else mu[q], Ri)
                for (bf, wf, Vf) in by_j[tjp]:
                    Rf = Vf.reshape(tjp + 1, Nrest, -1)
                    T = np.zeros((Rf.shape[2], Ri.shape[2]), complex)
                    for qi, q in enumerate((-1, 0, 1)):
                        M = np.einsum("arf,krl->akfl", Rf.conj(), Xi[q])      # <f,k'|mu_q|i,k>
                        T += np.einsum("ak,akfl->fl", C[qi], M)
                    S = (tj + 1) * (tjp + 1) * np.abs(T) ** 2
                    for f in range(S.shape[0]):
                        for l in range(S.shape[1]):
                            if S[f, l] > min_strength and (wf[f] > wi[l] + 1e-9):
                                lines.append({"upper_block": bf, "upper_index": f, "lower_block": bi, "lower_index": l,
                                              "wavenumber": float(wf[f] - wi[l]), "strength": float(S[f, l]),
                                              "upper_energy": float(wf[f]), "lower_energy": float(wi[l]),
                                              "upper_J": Jp, "lower_J": J,
                                              "upper_irrep": bm.blocks[bf].irrep, "lower_irrep": bm.blocks[bi].irrep})
    lines.sort(key=lambda d: d["wavenumber"])
    return lines


def intensities(bm: BuiltModel, theta, lines: list[dict], T: float, spin_weights: dict | None = None) -> float:
    """Add 'intensity' (relative absorption, cm^-1 units dropped) to each line; returns Q(T)."""
    c2 = 1.4387769
    Q = 0.0
    for b, (w, _) in enumerate(bm.eig(theta)):
        gns = (spin_weights or {}).get(bm.blocks[b].irrep, 1)
        Q += gns * (bm.blocks[b].two_j + 1) * np.sum(np.exp(-c2 * (w - 0.0) / T))
    for ln in lines:
        gns = (spin_weights or {}).get(ln["lower_irrep"], 1)
        nu = ln["wavenumber"]
        ln["intensity"] = float(gns * nu * np.exp(-c2 * ln["lower_energy"] / T) * (1 - np.exp(-c2 * nu / T))
                                * ln["strength"] / Q)
    return float(Q)
