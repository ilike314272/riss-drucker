"""Synthetic training data for amortized inference (README 5.7, 7).

sample structure -> sample theta from a prior -> compile/solve (certified) ->
features (levels, Chebyshev moments) + targets (theta, structure id) -> shards.

Shards are .npz files with a JSON manifest (spec canonical JSON + hash, prior,
seeds, library version, feature layout), so a Kaggle training notebook can
load them without the library and every sample is reproducible. Splits are by
structure or by parameter region, never by random lines (README 6.3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .. import _core
from .._version import __version__
from ..io_utils import atomic_write_bytes, atomic_write_json
from ..solve import solve
from .moments import model_moments


@dataclass
class Prior:
    """Independent priors per parameter: ("normal", mean, sd) | ("uniform", lo, hi) | ("loguniform", lo, hi)."""
    dists: dict

    def sample(self, rng, names) -> dict:
        out = {}
        for n in names:
            kind, a, b = self.dists.get(n, ("normal", 0.0, 0.0))
            if kind == "normal":
                out[n] = float(rng.normal(a, b))
            elif kind == "uniform":
                out[n] = float(rng.uniform(a, b))
            elif kind == "loguniform":
                out[n] = float(np.exp(rng.uniform(np.log(a), np.log(b))))
            else:
                raise ValueError(f"unknown prior kind {kind!r}")
        return out


def level_features(model, theta, n_levels: int = 8) -> np.ndarray:
    """Lowest n_levels per block (padded with NaN), certified solve."""
    sol = solve(model, theta, nlowest=n_levels, method="dense")
    if not sol.certified:
        raise RuntimeError("uncertified solve while generating training data")
    feats = []
    for r in sol.results:
        e = np.full(n_levels, np.nan)
        v = np.asarray(r["eigenvalues"])[:n_levels]
        e[: len(v)] = v
        feats.append(e)
    return np.concatenate(feats)


def generate(model, prior: Prior, n: int, out_dir, *, seed: int = 0, shard_size: int = 256, n_levels: int = 8,
             n_moments: int = 16, noise: float = 0.0, structure_id: int = 0,
             spec_canonical: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(model.param_names)
    X, M, Y, shards = [], [], [], []
    for i in range(n):
        th = prior.sample(rng, names)
        lv = level_features(model, th, n_levels)
        if noise > 0:
            lv = lv + rng.normal(0, noise, lv.shape)
        X.append(lv)
        M.append(model_moments(model, th, n_moments=n_moments, seed=int(rng.integers(1 << 31))))
        Y.append([th[k] for k in names])
        if len(X) == shard_size or i == n - 1:
            name = f"shard_{len(shards):05d}.npz"
            import io
            buf = io.BytesIO()
            np.savez(buf, levels=np.array(X), moments=np.array(M), theta=np.array(Y),
                     structure=np.full(len(X), structure_id))
            atomic_write_bytes(out / name, buf.getvalue())
            shards.append(name)
            X, M, Y = [], [], []
    manifest = {"library_version": __version__, "git_commit": _core.build_info()["git_commit"],
                "model": model.name, "spec_json": model.spec_json, "spec": spec_canonical,
                "param_names": names, "prior": prior.dists, "seed": seed, "n": n, "noise": noise,
                "features": {"levels": f"{n_levels} lowest per block x {model.num_blocks} blocks (NaN-padded)",
                             "moments": f"[center, halfwidth, mu_0..mu_{n_moments - 1}] per block"},
                "blocks": [b["key"] for b in model.blocks], "shards": shards}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def load(out_dir) -> dict:
    out = Path(out_dir)
    man = json.loads((out / "manifest.json").read_text())
    parts = [np.load(out / s) for s in man["shards"]]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("levels", "moments", "theta", "structure")}
    data["manifest"] = man
    return data
