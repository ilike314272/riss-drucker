"""Symbolic language model over effective-Hamiltonian terms (README 7).

A *symbolic* (not neural) generative model of operator "sentences": the vocabulary is
the grammar, candidates come from the certified operator dictionary (rd.algebra), and a
log-linear model scores each candidate from symbolic features of the operator and the
molecule:

    p(term useful | features) = sigmoid(w . f),

with f = per-space orders, number of modes involved, resonance character and harmonic
detuning (|sum omega created - sum omega annihilated| / omega_min), rotation-vibration
mixing, K mixing, spin and electronic off-diagonal character, imaginary-unit terms.
Weights are learned from what `autobuild` accepted or rejected in the case studies
(`harvest` + `fit`); `default()` loads the trained weights shipped with the package
(falling back to a physical prior: low order first, small detuning first).

Residual diagnostics re-weight generation: a residual that grows with J(J+1) favours
higher rotational orders; a residual concentrated in one polyad favours resonances.
As a proposer it only orders the search; autobuild compiles, certifies and fits.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

FEATURES = ["bias", "order", "vib", "rot", "spin", "elec_offdiag", "n_modes", "resonance", "detuning",
            "rot_vib", "k_mixing", "imag", "rot_diag", "vib_diag"]

PRIOR = {"bias": 0.0, "order": -0.6, "vib": 0.0, "rot": 0.0, "spin": 0.0, "elec_offdiag": 0.3, "n_modes": -0.3,
         "resonance": 0.5, "detuning": -2.0, "rot_vib": -0.2, "k_mixing": -0.5, "imag": -0.2, "rot_diag": 0.5,
         "vib_diag": 0.5}


def term_features(text: str, spec=None, frequencies: dict | None = None) -> dict:
    expr = text.split(":", 1)[1] if ":" in text else text
    toks = re.findall(r"a\d+[a-z]?\+?|J[xyz]|S[xyz]|e\[[^\]]*\]|n\d+|i\*|p\[\w+\]|(?:cos|sin)[\d.]+\[\w+\]", expr)
    vib = [t for t in toks if t.startswith("a")]
    rot = [t for t in toks if t.startswith("J")]
    spin = [t for t in toks if t.startswith("S")]
    el = [t for t in toks if t.startswith("e[")]
    modes = {re.match(r"a(\d+)", t).group(1) for t in vib}
    cre = [re.match(r"a(\d+)", t).group(1) for t in vib if t.endswith("+")]
    ann = [re.match(r"a(\d+)", t).group(1) for t in vib if not t.endswith("+")]
    resonance = float(len(vib) > 0 and sorted(cre) != sorted(ann))
    det = 0.0
    if resonance and frequencies:
        w = [frequencies.get(m) for m in modes]
        if all(x is not None for x in w):
            wmin = min(frequencies.values())
            det = abs(sum(frequencies[m] for m in cre) - sum(frequencies[m] for m in ann)) / wmin
    offd = float(any(e.split(",")[0][2:].rstrip("xyz") != e.split(",")[1].rstrip("]xyz") or
                     e.split(",")[0][2:] != e.split(",")[1][:-1] for e in el))
    nx = sum(1 for t in rot if t in ("Jx", "Jy"))
    return {"bias": 1.0, "order": float(len(vib) + len(rot) + len(spin)), "vib": float(len(vib)),
            "rot": float(len(rot)), "spin": float(len(spin)), "elec_offdiag": offd, "n_modes": float(len(modes)),
            "resonance": resonance, "detuning": min(det, 3.0), "rot_vib": float(bool(vib) and bool(rot)),
            "k_mixing": float(nx % 2 == 1 or (len(rot) and "Jz" in rot and nx % 2 == 1)),
            "imag": float("i*" in expr), "rot_diag": float(bool(rot) and nx % 2 == 0 and not vib),
            "vib_diag": float(bool(vib) and not resonance)}


@dataclass
class SymbolicLM:
    weights: dict = field(default_factory=lambda: dict(PRIOR))
    trained_on: list = field(default_factory=list)

    # ---- scoring ---------------------------------------------------------------------------
    def logit(self, f: dict) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in f.items())

    def prob(self, text: str, spec=None, frequencies=None) -> float:
        return 1 / (1 + math.exp(-self.logit(term_features(text, spec, frequencies))))

    # ---- learning ----------------------------------------------------------------------------
    def fit(self, records: list, l2: float = 3.0, iters: int = 100) -> "SymbolicLM":
        """records: [(features dict, label 0/1)]; L2-regularized logistic regression (Newton),
        regularized towards the physical prior."""
        X = np.array([[r[0].get(k, 0.0) for k in FEATURES] for r in records])
        y = np.array([float(r[1]) for r in records])
        w0 = np.array([PRIOR[k] for k in FEATURES])
        w = w0.copy()
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (p - y) + l2 * (w - w0)
            H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            step *= min(1.0, 1.0 / max(np.max(np.abs(step)), 1e-300))     # damped Newton
            w -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.weights = dict(zip(FEATURES, w.tolist()))
        return self

    @staticmethod
    def harvest(result, frequencies=None, source: str = "") -> list:
        """(features, label) from an autobuild result: accepted -> 1; rejected on evidence or as
        contact-transformation removable -> 0. Compiler rejections are not learning signal."""
        out = []
        for s in result.steps:
            if not s.term or "rejected by compiler" in s.reason:
                continue
            out.append((term_features(s.term, None, frequencies), 1 if s.accepted else 0, source))
        return out

    @staticmethod
    def records_from_model(spec, final_terms, degrees, frequencies=None, source: str = "") -> list:
        """Positives: dictionary terms of `degrees` kept in the final model (matched by operator);
        negatives: the other dictionary terms of those degrees."""
        from . import algebra
        from .build import Context, Parser, make_group, runtime_spaces
        g = make_group(spec)
        ctx = Context(g, runtime_spaces(spec, g))
        P = Parser(ctx)
        cands = algebra.enumerate_terms(spec.with_terms([]), degrees, prefix="q").terms

        def key(t):
            p = ctx.canon(P.parse_expr(t.split(":", 1)[1]))
            m = max(abs(v) for v in p.values())
            ref = max(p, key=lambda w: (abs(p[w]), str(w)))
            return tuple(sorted((str(w), round((v / p[ref]).real, 6)) for w, v in p.items()))
        final = {key(t) for t in final_terms}
        return [(term_features(t, spec, frequencies), 1 if key(t) in final else 0, source) for t in cands]

    # ---- generation ----------------------------------------------------------------------------
    def generate(self, spec, degrees, *, top: int = 10, frequencies=None, boosts: dict | None = None) -> list:
        from . import algebra
        cands = algebra.enumerate_terms(spec, degrees, exclude=list(spec.terms), prefix="s").terms
        w = dict(self.weights)
        for k, v in (boosts or {}).items():
            w[k] = w.get(k, 0.0) + v
        scored = []
        for t in cands:
            f = term_features(t, spec, frequencies)
            scored.append((sum(w.get(k, 0.0) * v for k, v in f.items()), t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top]]

    def as_proposer(self, *, top: int = 8, degrees_fn=None, frequencies=None):
        return _SLMProposer(self, top, degrees_fn, frequencies)

    # ---- persistence -----------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "trained_on": self.trained_on}, indent=1)

    @classmethod
    def default(cls) -> "SymbolicLM":
        try:
            from ._slm_weights import TRAINED_ON, WEIGHTS
            return cls(dict(WEIGHTS), list(TRAINED_ON))
        except ImportError:
            return cls()


def default_degrees(spec):
    """Next orders to explore for the spaces present (one order beyond what the model has)."""
    from . import algebra
    terms = " ".join(spec.terms)
    kinds = [s.kind for s in spec.spaces]
    rot_now = max([len(re.findall(r"J[xyz]", t)) for t in spec.terms] + [0])
    vib_now = max([len(re.findall(r"a\d", t)) + 2 * len(re.findall(r"n\d", t)) for t in spec.terms] + [0])
    kw = {}
    if "rot" in kinds:
        kw["rot"] = (2, min(rot_now + 2, 6))
    if "vib" in kinds:
        kw["vib"] = (1 if "elec" in kinds else 2, min(max(vib_now + 1, 3), 4))
    degs = algebra.multidegrees(spec, total=max(kw.get("rot", (0, 0))[1], kw.get("vib", (0, 0))[1]), **kw)
    return [d for d in degrees_filter(degs)]


def degrees_filter(degs):
    out = []
    for d in degs:
        tot = sum(x[1] for x in d if x[0] in ("vib", "rot"))
        if tot >= 1:
            out.append(d)
    return out


class _SLMProposer:
    name = "symbolic_lm"

    def __init__(self, lm, top, degrees_fn, frequencies):
        self.lm, self.top, self.degrees_fn, self.frequencies = lm, top, degrees_fn, frequencies

    def propose(self, context: dict) -> list:
        spec = context["spec"]
        degs = (self.degrees_fn or default_degrees)(spec)
        boosts = {}
        diag = context.get("diagnostics") or {}
        blocks = diag.get("by_block", [])
        if blocks:
            slopes = [abs(b.get("slope_vs_J(J+1)", 0.0)) for b in blocks]
            rms = [b.get("rms", 0.0) for b in blocks]
            if max(slopes + [0]) * 30 > max(rms + [1e-300]):
                boosts["rot"] = 0.5
            if len(rms) > 1 and max(rms) > 5 * np.median(rms):
                boosts["resonance"] = 1.0
        return self.lm.generate(spec, degs, top=self.top, frequencies=self.frequencies, boosts=boosts)
