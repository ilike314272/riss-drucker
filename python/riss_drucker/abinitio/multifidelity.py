"""Delta-learning: E_high(x) ~ E_low(x) + delta(x), delta ~ GP on few high points.

The low-fidelity surface is evaluated where needed (cheap); the GP carries the
correction and its predictive variance. low_at must be a callable or a dense
grid that is interpolated (periodic or not)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gp import GP


@dataclass
class DeltaModel:
    """E_high(x) = E_low(x) + delta(x). Both parts are GPs: the low-fidelity
    surface is interpolated by its own GP (dense, nearly exact data), the
    correction delta by a GP on the few high points (at the low GP's mean).
    The predictive variance is the sum of both, so interpolation error of the
    low-level grid is not hidden."""
    low_gp: GP
    delta_gp: GP
    periodic: bool

    @classmethod
    def fit(cls, low_x, low_y, high_x, high_y, *, kind: str = "rbf", seed: int = 0):
        low_x, low_y = np.asarray(low_x, float), np.asarray(low_y, float)
        lo = GP(kind=kind).fit(low_x, low_y, seed=seed)
        delta = np.asarray(high_y, float) - lo.predict(np.asarray(high_x, float))[0]
        dg = GP(kind=kind).fit(high_x, delta, seed=seed + 1)
        return cls(lo, dg, kind == "periodic")

    def low(self, x):
        return self.low_gp.predict(np.asarray(x, float))[0]

    def predict(self, x):
        x = np.asarray(x, float)
        ml, vl = self.low_gp.predict(x)
        md, vd = self.delta_gp.predict(x)
        return ml + md, vl + vd

    def sample(self, x, n: int, seed: int = 0):
        x = np.asarray(x, float)
        return self.low_gp.sample(x, n, seed) + self.delta_gp.sample(x, n, seed + 7919)
