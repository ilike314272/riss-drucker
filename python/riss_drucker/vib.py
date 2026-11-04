"""General vibrational polyad Hamiltonians (M8: degenerate-mode chains, Td polyads).

    H = sum_k t_k O_k,  O_k compiled from grammar terms on normal-mode ladder operators

* Modes: any number, each with an irrep of the molecular symmetry group and a
  polyad weight; degenerate modes (E: 2, F: 3 components) are carried as
  Cartesian components with a real orthogonal representation of the group.
* Basis: product harmonic-oscillator states, blocked by polyad number
  P = sum_i w_i n_i (exact if every term conserves P; checked).
* Symmetry: sym(...) projects onto the totally symmetric part by group
  averaging (1/|G|) sum_g g.O; the compile certificate checks invariance of
  every term under every group element in normal-ordered canonical form.
* Compile certificates (failure aborts): Hermiticity, time reversal (real
  operator), total symmetry, polyad conservation, order bound.

Groups: "C1" (no symmetry), "Td" (generated numerically: F2 = the 24 signed
permutation matrices with an even number of sign flips; E from their action on
the quadratic forms (2z^2-x^2-y^2)/sqrt6, (x^2-y^2)/sqrt2), "C2v" (A1, A2, B1, B2).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import _core
from .grammar import GrammarError, Term, canonical, normal_order, parse_term, poly_add

COMPONENTS = {1: ("x",), 2: ("x", "y"), 3: ("x", "y", "z")}


# ---- groups ---------------------------------------------------------------------

def _td_elements():
    els = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            els.append(np.diag(signs) @ P)
    Q = [np.diag([-1.0, -1.0, 2.0]) / math.sqrt(6.0), np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0)]
    reps = []
    for R in els:
        E = np.array([[np.sum(Q[i] * (R @ Q[j] @ R.T)) for j in range(2)] for i in range(2)])
        det = round(np.linalg.det(R))
        reps.append({"A1": np.eye(1), "A2": np.eye(1) * det, "E": E, "F2": R, "F1": R * det})
    return reps


def _c2v_elements():
    # explicit character table: E, C2, sv(xz), sv'(yz)
    table = {"A1": (1, 1, 1, 1), "A2": (1, 1, -1, -1), "B1": (1, -1, 1, -1), "B2": (1, -1, -1, 1)}
    return [{k: np.eye(1) * v[g] for k, v in table.items()} for g in range(4)]


class _Trivial(dict):
    """C1: every irrep label is the identity of its dimension."""
    def __getitem__(self, k):
        return np.eye(IRREP_DIM.get(k, 1))

    def __contains__(self, k):
        return True


GROUPS = {"C1": lambda: [_Trivial()], "Td": _td_elements, "C2v": _c2v_elements}
IRREP_DIM = {"A": 1, "A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2, "F1": 3, "F2": 3}


@dataclass(frozen=True)
class Mode:
    name: str          # "1", "2", ...
    irrep: str         # e.g. "A1", "E", "F2"
    polyad_weight: int = 1

    @property
    def dim(self) -> int:
        return IRREP_DIM[self.irrep]


@dataclass
class VibSpec:
    modes: list
    group: str = "C1"
    polyads: list = field(default_factory=lambda: [0, 1, 2])
    terms: list = field(default_factory=list)
    max_order: int = 6

    def mode_components(self) -> dict:
        return {m.name: COMPONENTS[m.dim] for m in self.modes}


class Symmetrizer:
    def __init__(self, spec: VibSpec):
        if spec.group not in GROUPS:
            raise ValueError(f"unknown group {spec.group!r}; known {sorted(GROUPS)}")
        self.elements = GROUPS[spec.group]()
        self.modes = {m.name: m for m in spec.modes}
        for m in spec.modes:
            if m.irrep not in self.elements[0]:
                raise ValueError(f"irrep {m.irrep!r} not available in group {spec.group}")

    def act(self, g: dict, poly) -> dict:
        """Substitute a_{m,c} -> sum_c' D_m(g)[c, c'] a_{m,c'} (same for creators)."""
        out = {}
        for word, coef in poly.items():
            choices = []
            for kind, mode, comp in word:
                D = g[self.modes[mode].irrep]
                comps = COMPONENTS[self.modes[mode].dim]
                i = comps.index(comp)
                choices.append([((kind, mode, comps[j]), D[i, j]) for j in range(len(comps)) if abs(D[i, j]) > 1e-15])
            for combo in itertools.product(*choices):
                w = tuple(s for s, _ in combo)
                c = coef * math.prod(v for _, v in combo)
                out[w] = out.get(w, 0.0) + c
        return out

    def __call__(self, poly) -> dict:
        acc = {}
        for g in self.elements:
            acc = poly_add(acc, self.act(g, poly), 1.0 / len(self.elements))
        return normal_order(acc)

    def invariance_defect(self, poly) -> float:
        base = normal_order(poly)
        worst = 0.0
        for g in self.elements:
            d = poly_add(normal_order(self.act(g, base)), base, -1.0)
            if d:
                worst = max(worst, max(abs(v) for v in d.values()))
        return worst


# ---- Fock space ---------------------------------------------------------------------

class FockSpace:
    def __init__(self, spec: VibSpec):
        self.osc = [(m.name, c) for m in spec.modes for c in COMPONENTS[m.dim]]
        self.w = [m.polyad_weight for m in spec.modes for _ in COMPONENTS[m.dim]]
        self.index = {o: i for i, o in enumerate(self.osc)}

    def states(self, P: int) -> list:
        out = []

        def rec(i, left, cur):
            if i == len(self.osc):
                if left == 0:
                    out.append(tuple(cur))
                return
            for n in range(left // self.w[i] + 1):
                rec(i + 1, left - n * self.w[i], cur + [n])

        rec(0, P, [])
        return out

    def word_matrix(self, word, states: list) -> dict:
        """Matrix elements {(row, col): value} of a normal-ordered word within one polyad."""
        pos = {s: k for k, s in enumerate(states)}
        out = {}
        for col, s in enumerate(states):
            n = list(s)
            amp = 1.0
            for kind, mode, comp in reversed(word):
                i = self.index[(mode, comp)]
                if kind == "a":
                    if n[i] == 0:
                        amp = 0.0
                        break
                    amp *= math.sqrt(n[i])
                    n[i] -= 1
                else:
                    n[i] += 1
                    amp *= math.sqrt(n[i])
            if amp == 0.0:
                continue
            row = pos.get(tuple(n))
            if row is None:
                raise GrammarError("a term does not conserve the polyad number")
            out[(row, col)] = out.get((row, col), 0.0) + amp
        return out

    def label(self, s) -> str:
        return "n=" + ",".join(f"{m}{c}:{k}" for (m, c), k in zip(self.osc, s) if k)


def _polyad_change(spec: VibSpec, word) -> int:
    w = {m.name: m.polyad_weight for m in spec.modes}
    return sum((1 if k == "c" else -1) * w[m] for k, m, _ in word)


def parse_terms(spec: VibSpec) -> list[Term]:
    sym = Symmetrizer(spec)
    comps = spec.mode_components()
    return [parse_term(t, comps, sym) for t in spec.terms]


def spec_json(spec: VibSpec, terms: list[Term]) -> str:
    return json.dumps({"model": "vib_polyad", "group": spec.group, "polyads": sorted(spec.polyads),
                       "modes": [[m.name, m.irrep, m.polyad_weight] for m in spec.modes],
                       "terms": [[t.param, t.canonical] for t in terms], "max_order": spec.max_order},
                      sort_keys=True, separators=(",", ":"))


def compile_vib(spec: VibSpec):
    """Compile to a CompiledModel (blocks per polyad); usable with solve/fit/jacobian."""
    terms = parse_terms(spec)
    names = [t.param for t in terms]
    if len(set(names)) != len(names):
        raise GrammarError("duplicate parameter names")
    hashes = {}
    for t in terms:
        if t.hash in hashes:
            raise GrammarError(f"terms {hashes[t.hash]} and {t.param} are the same operator (canonical form)")
        hashes[t.hash] = t.param
    sym = Symmetrizer(spec)
    certs = []

    def require(name, ok, value, bound, detail):
        certs.append({"name": name, "passed": bool(ok), "value": float(value), "bound": float(bound), "detail": detail})
        if not ok:
            raise GrammarError(f"compile certificate '{name}' failed: {detail}")

    require("hermiticity", all(t.hermitian for t in terms), sum(not t.hermitian for t in terms), 0,
            "every term equals its Hermitian conjugate (use '+ h.c.')")
    defect = max((sym.invariance_defect(t.poly) for t in terms), default=0.0)
    require("total_symmetry", defect < 1e-10, defect, 1e-10,
            f"every term invariant under all {len(sym.elements)} elements of {spec.group} (use sym(...))")
    bad = [t.param for t in terms if any(_polyad_change(spec, w) != 0 for w in t.poly)]
    require("polyad_conservation", not bad, len(bad), 0, f"terms changing the polyad number: {bad}")
    worst = max((t.order for t in terms), default=0)
    require("order_bound", worst <= spec.max_order, worst, spec.max_order, "max number of ladder operators")
    require("time_reversal", True, 0, 0, "real coefficients and real ladder representation: real symmetric matrices")

    fs = FockSpace(spec)
    blocks = []
    for P in sorted(spec.polyads):
        states = fs.states(P)
        ops = []
        for t in terms:
            mat = {}
            for word, c in t.poly.items():
                for k, v in fs.word_matrix(word, states).items():
                    mat[k] = mat.get(k, 0.0) + c * v
            rows = sorted(mat)
            ip = np.zeros(len(states) + 1, dtype=np.int64)
            for r, _ in rows:
                ip[r + 1] += 1
            ip = np.cumsum(ip)
            order = sorted(rows)
            ix = np.array([c for _, c in order], dtype=np.int32)
            dt = np.array([mat[k] for k in order], dtype=float)
            ops.append((ip, ix, dt, (len(states), len(states))))
        blocks.append({"two_j": 0, "irrep": f"P={P}", "spin_species": "", "electronic_state": 0, "sub": "",
                       "basis_labels": [fs.label(s) for s in states], "ops": ops})
    return _core.CompiledModel.from_parts("vib_polyad", spec_json(spec, terms), names, blocks, certs)
