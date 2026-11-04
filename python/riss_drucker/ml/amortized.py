"""Amortized inference baseline: features -> posterior over theta (README 2).

A Bayesian linear model on standardized (optionally expanded) features gives
a Gaussian predictive mean and covariance for theta in one matrix product.
It is a baseline and a contract, not the final network: any learned model
with the same predict() signature plugs in. Its output is never the source of
record: verify() starts the certified physics fit from the prediction and
reports both, and calibration() measures empirical coverage of the predicted
intervals on held-out data (README 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clean(F):
    F = np.asarray(F, dtype=float)
    return np.where(np.isfinite(F), F, 0.0)


@dataclass
class BayesLinearPosterior:
    alpha: float = 1e-3        # weight prior precision (standardized units)
    quadratic: bool = True     # add squared features

    def _expand(self, F):
        F = _clean(F)
        return np.hstack([F, F**2]) if self.quadratic else F

    def fit(self, features, theta):
        X = self._expand(features)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12
        Y = np.asarray(theta, dtype=float)
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-300
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mx) / self.sx])
        Yz = (Y - self.my) / self.sy
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.Ainv = np.linalg.inv(A)
        self.W = self.Ainv @ Z.T @ Yz
        resid = Yz - Z @ self.W
        dof = max(len(X) - Z.shape[1], 1)
        self.noise = (resid.T @ resid) / dof   # residual covariance (standardized)
        return self

    def predict(self, features):
        X = self._expand(np.atleast_2d(features))
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mx) / self.sx])
        mean = (Z @ self.W) * self.sy + self.my
        lev = np.einsum("ij,jk,ik->i", Z, self.Ainv, Z)
        cov = np.array([(self.noise * (1 + l)) * np.outer(self.sy, self.sy) for l in lev])
        return mean, cov


def calibration(model, features, theta, levels=(0.6827, 0.9545)) -> dict:
    from scipy.stats import norm

    mean, cov = model.predict(features)
    sd = np.sqrt(np.einsum("kii->ki", cov))
    z = np.abs((np.asarray(theta) - mean) / sd)
    return {"nominal": list(levels),
            "empirical": [float(np.mean(z <= norm.ppf(0.5 + l / 2))) for l in levels],
            "per_parameter_rmse": np.sqrt(np.mean((np.asarray(theta) - mean) ** 2, axis=0)).tolist()}


def verify(compiled_model, observations, prediction: dict, **fit_kw):
    """ML proposes, physics verifies: certified fit started from the prediction."""
    from ..fit import fit

    res = fit(compiled_model, observations, prediction, **fit_kw)
    return {"fit": res, "certified": res.certified, "converged": res.converged,
            "shift_sigma": {n: (res.theta[n] - prediction[n]) / res.sigma[n] if res.sigma[n] == res.sigma[n] and res.sigma[n] > 0
                            else float("nan") for n in res.param_names}}
