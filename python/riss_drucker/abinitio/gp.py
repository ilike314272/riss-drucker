"""Gaussian-process regression with predictive variance (numpy/scipy only).

Kernels: "rbf" exp(-(x-x')^2 / (2 l^2)) and "periodic" exp(-2 sin^2((x-x')/2) / l^2)
(period 2 pi, for torsion angles). Hyperparameters (signal sd, length, noise sd)
by maximizing the log marginal likelihood (multiple restarts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kernel(x1, x2, kind: str, length: float):
    d = np.subtract.outer(np.asarray(x1, float), np.asarray(x2, float))
    if kind == "rbf":
        return np.exp(-0.5 * (d / length) ** 2)
    if kind == "periodic":
        return np.exp(-2.0 * np.sin(d / 2.0) ** 2 / length**2)
    raise ValueError(kind)


@dataclass
class GP:
    kind: str = "rbf"
    signal: float = 1.0
    length: float = 1.0
    noise: float = 1e-6
    mean: float = 0.0

    def fit(self, x, y, *, optimize: bool = True, restarts: int = 6, seed: int = 0, noise_floor: float = 1e-9):
        from scipy.optimize import minimize

        self.x = np.asarray(x, float)
        y = np.asarray(y, float)
        self.mean = float(y.mean())
        self.y = y - self.mean
        if optimize:
            rng = np.random.default_rng(seed)
            scale = float(np.std(self.y)) or 1.0
            best = None
            for r in range(restarts):
                p0 = np.log([scale * rng.uniform(0.5, 2), rng.uniform(0.3, 2.0), scale * 1e-3 * rng.uniform(0.1, 10)])
                res = minimize(lambda p: -self._lml(np.exp(p), noise_floor), p0, method="L-BFGS-B",
                               bounds=[(np.log(scale * 1e-3), np.log(scale * 1e3)), (np.log(0.05), np.log(20.0)),
                                       (np.log(noise_floor), np.log(scale))])
                if best is None or res.fun < best.fun:
                    best = res
            self.signal, self.length, self.noise = np.exp(best.x)
            self.noise = max(self.noise, noise_floor)
        K = self.signal**2 * kernel(self.x, self.x, self.kind, self.length) + self.noise**2 * np.eye(len(self.x))
        self.L = np.linalg.cholesky(K + 1e-12 * np.trace(K) / len(K) * np.eye(len(K)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def _lml(self, p, floor):
        s, l, n = p
        K = s**2 * kernel(self.x, self.x, self.kind, l) + max(n, floor) ** 2 * np.eye(len(self.x))
        try:
            L = np.linalg.cholesky(K + 1e-12 * np.trace(K) / len(K) * np.eye(len(K)))
        except np.linalg.LinAlgError:
            return -1e300
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return float(-0.5 * self.y @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(self.y) * np.log(2 * np.pi))

    def predict(self, xs, full_cov: bool = False):
        Ks = self.signal**2 * kernel(xs, self.x, self.kind, self.length)
        mu = Ks @ self.alpha + self.mean
        v = np.linalg.solve(self.L, Ks.T)
        if full_cov:
            K2 = self.signal**2 * kernel(xs, xs, self.kind, self.length)
            return mu, K2 - v.T @ v
        return mu, np.maximum(self.signal**2 - np.sum(v * v, axis=0), 0.0)

    def sample(self, xs, n: int, seed: int = 0):
        mu, C = self.predict(xs, full_cov=True)
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        rng = np.random.default_rng(seed)
        return mu + (rng.standard_normal((n, len(w))) * np.sqrt(np.clip(w, 0, None))) @ V.T
