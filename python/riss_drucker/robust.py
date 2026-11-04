"""Outlier mixture for unmodelled perturbers (README 5.5).

Residuals r_i = E_obs - E_calc are modelled as
    (1 - w) N(0, sigma_i^2) + w N(0, sigma_b^2)
EM alternates a weighted fit (effective sigma_i / sqrt(responsibility)) with
updates of the per-observation outlier probabilities and of w. Outliers are
down-weighted, never silently dropped: every probability is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .fit import FitResult, Observation, fit


@dataclass
class RobustFitResult:
    fit: FitResult
    outlier_probability: list[float]
    outlier_fraction: float
    broad_sigma: float
    em_iterations: int
    converged: bool


def _normal_logpdf(r, s):
    return -0.5 * (r / s) ** 2 - np.log(s) - 0.5 * math.log(2 * math.pi)


def fit_robust(model, observations: Sequence[Observation], theta0, *, outlier_fraction: float = 0.05,
               broad_sigma: float | None = None, max_em: int = 25, tol: float = 1e-4,
               resp_floor: float = 1e-6, **fit_kw) -> RobustFitResult:
    obs = list(observations)
    sig = np.array([o.sigma for o in obs])
    sb = broad_sigma if broad_sigma is not None else 100.0 * float(np.median(sig))
    w = float(outlier_fraction)
    resp = np.ones(len(obs))  # probability of the narrow component
    theta = theta0
    res = None
    converged = False
    it = 0
    for it in range(1, max_em + 1):
        eff = [replace(o, sigma=float(o.sigma / math.sqrt(max(p, resp_floor)))) for o, p in zip(obs, resp)]
        res = fit(model, eff, theta, **fit_kw)
        theta = res.theta
        r = np.array([lv["residual"] for lv in res.levels])
        ln_n = math.log(max(1 - w, 1e-300)) + _normal_logpdf(r, sig)
        ln_b = math.log(max(w, 1e-300)) + _normal_logpdf(r, sb)
        new = 1.0 / (1.0 + np.exp(np.clip(ln_b - ln_n, -700, 700)))
        w_new = float(np.clip(1.0 - new.mean(), 1e-6, 0.5))
        delta = float(np.max(np.abs(new - resp)))
        resp, w = new, w_new
        if delta < tol and it > 1:
            converged = True
            break
    return RobustFitResult(fit=res, outlier_probability=(1.0 - resp).tolist(), outlier_fraction=w,
                           broad_sigma=sb, em_iterations=it, converged=converged)
