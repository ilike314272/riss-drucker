"""Amortized structure inference: which operators does a spectrum need?

Simulate many models from one compiled *super-model* (base terms + candidate
terms) with random subsets of the candidates switched on, featurize the levels,
and train a multi-label classifier p(candidate k present | levels). The
`AmortizedProposer` then ranks dictionary candidates for observed data in one
forward pass; autobuild still compiles, fits and scores every proposal, so the
network only orders the search.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..build import BuildSpec, compile_build
from .neural import MLPClassifier


def level_features(bm, theta, n_levels: int, keys: list) -> np.ndarray:
    """Lowest n_levels of each block in `keys` order ((two_j, irrep, polyad)); NaN if absent."""
    eig = {(b.two_j, b.irrep, b.polyad): w for b, (w, _) in zip(bm.blocks, bm.eig(theta))}
    out = []
    for k in keys:
        w = eig.get(k, np.array([]))
        v = np.full(n_levels, np.nan)
        v[:min(n_levels, len(w))] = w[:n_levels]
        out.append(v)
    return np.concatenate(out)


def data_features(data, n_levels: int, keys: list) -> np.ndarray:
    """Same features from observed LevelDatum lists (missing -> NaN)."""
    d = {}
    for x in data:
        d[(round(2 * x.J), x.irrep, x.polyad, x.index)] = x.energy
    out = []
    for k in keys:
        out.append([d.get((k[0], k[1], k[2], i), np.nan) for i in range(n_levels)])
    return np.array(out, float).ravel()


@dataclass
class StructureDataset:
    X: np.ndarray
    Z: np.ndarray         # inclusion indicators (n, n_candidates)
    theta: np.ndarray
    candidates: list
    keys: list
    n_levels: int


def simulate(spec: BuildSpec, candidates: list, base_sampler, cand_scale: dict | float, n: int, *,
             p_include: float = 0.4, n_levels: int = 4, noise: float = 0.0, seed: int = 0) -> StructureDataset:
    """base_sampler(rng) -> dict of base-parameter values; candidate values ~ N(0, scale^2)
    when included, 0 otherwise."""
    bm = compile_build(spec.with_terms(list(spec.terms) + list(candidates)))
    names = bm.param_names
    cnames = [c.split(":")[0].strip() for c in candidates]
    keys = sorted({(b.two_j, b.irrep, b.polyad) for b in bm.blocks}, key=str)
    rng = np.random.default_rng(seed)
    X, Zs, TH = [], [], []
    for _ in range(n):
        th = dict(base_sampler(rng))
        z = rng.random(len(cnames)) < p_include
        for c, on in zip(cnames, z):
            s = cand_scale.get(c, 1.0) if isinstance(cand_scale, dict) else cand_scale
            th[c] = float(rng.normal(0, s)) if on else 0.0
        f = level_features(bm, th, n_levels, keys)
        if noise:
            f = f + noise * rng.standard_normal(f.shape)
        X.append(f)
        Zs.append(z.astype(float))
        TH.append([th[k] for k in names])
    return StructureDataset(np.array(X), np.array(Zs), np.array(TH), list(candidates), keys, n_levels)


class AmortizedProposer:
    """Rank candidates by the classifier's inclusion probability for the observed data."""

    name = "amortized"

    def __init__(self, ds: StructureDataset, classifier: MLPClassifier | None = None, *, threshold: float = 0.2,
                 top: int = 6, featurize=None):
        self.ds = ds
        self.clf = classifier or MLPClassifier().fit(_relative(ds.X), ds.Z)
        self.threshold, self.top = threshold, top
        self.featurize = featurize

    def probabilities(self, data) -> dict:
        x = data_features(data, self.ds.n_levels, self.ds.keys)
        p = self.clf.predict_proba(_relative(x[None, :]))[0]
        return dict(zip(self.ds.candidates, p.tolist()))

    def propose(self, context: dict) -> list[str]:
        data = context.get("data")
        if data is None:
            return []
        present = {t.split(":", 1)[1].strip() for t in context["spec"].terms}
        ranked = sorted(self.probabilities(data).items(), key=lambda kv: -kv[1])
        return [c for c, p in ranked if p >= self.threshold and c.split(":", 1)[1].strip() not in present][:self.top]


def _relative(X):
    """Features relative to the lowest level of each sample (offset-invariant)."""
    X = np.asarray(X, float)
    base = np.nanmin(X, axis=1, keepdims=True)
    return X - base
