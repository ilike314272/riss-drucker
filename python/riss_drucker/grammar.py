"""Operator grammar for effective vibrational Hamiltonians (README 7).

A structure proposer (enumerative, sparse-Bayesian, or a language model) emits
*terms* as strings in a small typed language; it never writes matrix elements,
angular algebra or numeric coefficients. The compiler turns terms into exact
operators and type-checks them.

Term syntax (one parameter per term):
    "<param>: <expr>"
    expr    := product ( ("+" | "-") product )* [ "+ h.c." ]
    product := [number "*"] factor ( "*" | " " factor )*
    factor  := "n<mode>"                 number operator of a mode (summed over components)
             | "a<mode><comp>+" | "a<mode><comp>"   creation / annihilation of one component
             | "sym(" expr ")"           projection onto the totally symmetric part (group average)
             | "(" expr ")"
             | "I"                       identity
Modes are named by integers ("1", "2", ...) and components by letters x, y, z
(for one-dimensional modes the component may be omitted). Examples:
    "omega3: n3"
    "x34: n3*n4"
    "k144: sym(a1+ a4x a4x) + h.c."          Fermi resonance nu1 / 2 nu4 (methane)
    "g33: sym(a3x+ a3x+ a3x a3x)"            anharmonic coupling within an F2 mode

Polynomials are kept in normal-ordered canonical form (creators left of
annihilators, sorted), which gives a canonical string and hash per term, so
equivalent proposals are deduplicated before any numerics.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass, field

import numpy as np

# A symbol is (kind, mode, comp) with kind "c" (creation) or "a" (annihilation).
Symbol = tuple
Poly = dict  # tuple[Symbol, ...] -> float


def _add(p: Poly, word: tuple, c: float) -> None:
    if c == 0.0:
        return
    v = p.get(word, 0.0) + c
    if abs(v) < 1e-14:
        p.pop(word, None)
    else:
        p[word] = v


def poly_add(p: Poly, q: Poly, s: float = 1.0) -> Poly:
    r = dict(p)
    for w, c in q.items():
        _add(r, w, s * c)
    return r


def poly_mul(p: Poly, q: Poly) -> Poly:
    r: Poly = {}
    for w1, c1 in p.items():
        for w2, c2 in q.items():
            _add(r, w1 + w2, c1 * c2)
    return r


def normal_order(p: Poly) -> Poly:
    """Rewrite with all creators left of annihilators using [a_i, a_j^+] = delta_ij,
    then sort creators and annihilators (they commute among themselves)."""
    out: Poly = {}
    stack = list(p.items())
    while stack:
        word, c = stack.pop()
        k = next((i for i in range(len(word) - 1) if word[i][0] == "a" and word[i + 1][0] == "c"), None)
        if k is None:
            cr = sorted(s for s in word if s[0] == "c")
            an = sorted(s for s in word if s[0] == "a")
            _add(out, tuple(cr + an), c)
            continue
        x, y = word[k], word[k + 1]
        swapped = word[:k] + (y, x) + word[k + 2:]
        stack.append((swapped, c))
        if x[1:] == y[1:]:  # same mode and component: a a^+ = a^+ a + 1
            stack.append((word[:k] + word[k + 2:], c))
    return out


def dagger(p: Poly) -> Poly:
    flip = {"c": "a", "a": "c"}
    return {tuple((flip[s[0]], s[1], s[2]) for s in reversed(w)): c for w, c in p.items()}


def canonical(p: Poly, tol: float = 1e-12) -> str:
    """Canonical string of a normal-ordered polynomial, scaled so the largest
    coefficient is +1 (an overall factor is absorbed into the parameter)."""
    q = normal_order(p)
    if not q:
        return "0"
    big = max(q.items(), key=lambda kv: (abs(kv[1]), str(kv[0])))[1]
    parts = []
    for w in sorted(q):
        c = q[w] / big
        if abs(c) < tol:
            continue
        sym = " ".join(f"a{s[1]}{s[2]}{'+' if s[0] == 'c' else ''}" for s in w) or "I"
        parts.append(f"{c:+.10g} {sym}")
    return " ".join(parts)


def term_hash(p: Poly) -> str:
    return hashlib.sha256(canonical(p).encode()).hexdigest()[:16]


# ---- parser ---------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(sym\(|h\.c\.|\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|n\w+|a\w+\+?|I|[()*+\-])")


class GrammarError(ValueError):
    pass


@dataclass
class Term:
    param: str
    source: str
    poly: Poly
    canonical: str
    hash: str
    order: int
    hermitian: bool
    meta: dict = field(default_factory=dict)


def _number_op(mode: str, comps) -> Poly:
    return {((("c", mode, c), ("a", mode, c))): 1.0 for c in comps}


class Parser:
    def __init__(self, text: str, modes: dict, symmetrizer):
        self.toks = [t for t in _TOKEN.findall(text)]
        if "".join(self.toks).replace(" ", "") != re.sub(r"\s+", "", text):
            raise GrammarError(f"cannot tokenize {text!r}")
        self.i = 0
        self.modes = modes
        self.sym = symmetrizer

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, t=None):
        tok = self.peek()
        if tok is None or (t is not None and tok != t):
            raise GrammarError(f"expected {t!r}, got {tok!r}")
        self.i += 1
        return tok

    def expr(self) -> Poly:
        p = self.product()
        hc = False
        while self.peek() in ("+", "-"):
            op = self.take()
            if self.peek() == "h.c.":
                if op != "+":
                    raise GrammarError("use '+ h.c.'")
                self.take()
                hc = True
                continue
            p = poly_add(p, self.product(), 1.0 if op == "+" else -1.0)
        return poly_add(p, dagger(p)) if hc else p

    def product(self) -> Poly:
        p: Poly = {(): 1.0}
        first = True
        if self.peek() in (None, ")", "+", "-"):
            raise GrammarError(f"expected a factor, got {self.peek()!r}")
        while True:
            tok = self.peek()
            if tok is None or tok in (")", "+", "-"):
                break
            if tok == "*":
                self.take()
                continue
            p = poly_mul(p, self.factor(first))
            first = False
        return p

    def factor(self, first: bool) -> Poly:
        tok = self.take()
        if re.fullmatch(r"\d+\.?\d*(?:[eE][-+]?\d+)?", tok):
            return {(): float(tok)}
        if tok == "I":
            return {(): 1.0}
        if tok == "(":
            p = self.expr()
            self.take(")")
            return p
        if tok == "sym(":
            p = self.expr()
            self.take(")")
            return self.sym(p)
        if tok.startswith("n"):
            mode = tok[1:]
            if mode not in self.modes:
                raise GrammarError(f"unknown mode {mode!r}")
            return _number_op(mode, self.modes[mode])
        if tok.startswith("a"):
            create = tok.endswith("+")
            body = tok[1:-1] if create else tok[1:]
            m = re.fullmatch(r"(\d+)([xyz]?)", body)
            if not m or m.group(1) not in self.modes:
                raise GrammarError(f"bad ladder operator {tok!r}")
            mode, comp = m.group(1), m.group(2) or self.modes[m.group(1)][0]
            if comp not in self.modes[mode]:
                raise GrammarError(f"mode {mode} has no component {comp!r}")
            return {(("c" if create else "a", mode, comp),): 1.0}
        raise GrammarError(f"unexpected token {tok!r}")


def parse_term(text: str, modes: dict, symmetrizer) -> Term:
    if ":" not in text:
        raise GrammarError("a term is '<param>: <expr>'")
    param, expr = (s.strip() for s in text.split(":", 1))
    if not re.fullmatch(r"[A-Za-z_]\w*", param):
        raise GrammarError(f"bad parameter name {param!r}")
    ps = Parser(expr, modes, symmetrizer)
    poly = ps.expr()
    if ps.peek() is not None:
        raise GrammarError(f"trailing input at {ps.peek()!r}")
    poly = normal_order(poly)
    if not poly:
        raise GrammarError(f"term {param} is identically zero (e.g. a non-symmetric seed under sym())")
    order = max(len(w) for w in poly)
    herm = canonical(poly_add(poly, dagger(poly), -1.0)) == "0"
    return Term(param, text, poly, canonical(poly), term_hash(poly), order, herm)
