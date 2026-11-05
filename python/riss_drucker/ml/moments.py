"""Chebyshev (kernel polynomial) moments of H(theta) per block by stochastic
trace estimation (README 5.6): no diagonalization, cost O(n_moments * nnz *
n_vectors), using the SIMD-dispatched CSR matvec. Moments are basis- and
gauge-invariant fingerprints (they do NOT identify Hund's cases) and are the
default spectral features for amortized inference.

  mu_k = (1/D) Tr T_k(H~),  H~ = (H - c) / h maps the spectrum into [-1, 1]
"""
from __future__ import annotations

import numpy as np

from .. import _core


def _bounds(indptr, indices, data, n):
    lo, hi = np.inf, -np.inf
    for r in range(n):
        s = slice(indptr[r], indptr[r + 1])
        diag = data[s][indices[s] == r].sum()
        off = np.abs(data[s][indices[s] != r]).sum()
        lo, hi = min(lo, diag - off), max(hi, diag + off)
    return lo, hi


def chebyshev_moments(indptr, indices, data, n_moments: int = 64, n_vectors: int = 16, seed: int = 0,
                      bounds: tuple | None = None, exact: bool = False):
    """Return (mu, center, halfwidth). exact=True uses unit vectors (Tr exactly)."""
    n = len(indptr) - 1
    lo, hi = bounds if bounds else _bounds(indptr, indices, data, n)
    c, h = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.01 + 1e-300
    mv = lambda v: (_core.csr_matvec(indptr, indices, data, v) - c * v) / h
    rng = np.random.default_rng(seed)
    vecs = np.eye(n) if exact else rng.choice([-1.0, 1.0], size=(n_vectors, n))
    mu = np.zeros(n_moments)
    for v in (vecs if exact else vecs):
        t0, t1 = v, mv(v)
        mu[0] += v @ t0
        if n_moments > 1:
            mu[1] += v @ t1
        for k in range(2, n_moments):
            t0, t1 = t1, 2 * mv(t1) - t0
            mu[k] += v @ t1
    # exact: sum_i e_i^T T e_i = Tr T; stochastic: E[v^T T v] = Tr T. Either way / (n_vectors_used) / D
    return (mu / n if exact else mu / (len(vecs) * n)), c, h


def jackson_dos(mu, center, halfwidth, energies):
    """Density of states from moments with the Jackson kernel."""
    N = len(mu)
    k = np.arange(N)
    g = ((N - k + 1) * np.cos(np.pi * k / (N + 1)) + np.sin(np.pi * k / (N + 1)) / np.tan(np.pi / (N + 1))) / (N + 1)
    x = np.clip((np.asarray(energies) - center) / halfwidth, -0.999999, 0.999999)
    T = np.cos(np.outer(np.arccos(x), k))
    w = np.where(k == 0, 1.0, 2.0)
    return (T @ (g * w * mu)) / (np.pi * np.sqrt(1 - x * x)) / halfwidth


def model_moments(model, theta, n_moments: int = 32, n_vectors: int = 16, seed: int = 0) -> np.ndarray:
    """Concatenated moments of every block (gauge-invariant fingerprint)."""
    vec = [float(theta[n]) for n in model.param_names] if isinstance(theta, dict) else list(theta)
    out = []
    for b in range(model.num_blocks):
        ip, ix, dt, shape = model.assemble(b, vec)
        mu, c, h = chebyshev_moments(ip, ix, dt, n_moments, n_vectors, seed + b, exact=shape[0] <= n_vectors)
        out.append(np.concatenate([[c, h], mu]))
    return np.concatenate(out)
