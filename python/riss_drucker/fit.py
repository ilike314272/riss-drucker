"""Fit H(theta) to assigned levels (README 5.4).

Levenberg-Marquardt on whitened residuals r = (E_obs - E_calc)/sigma with the
Hellmann-Feynman Jacobian (never autodiff through the eigensolver), an optional
Gaussian prior on theta, column scaling, and an SVD that separates identifiable
directions from gauge (null) directions:

  * steps and the covariance live in the identifiable subspace only. Column
    scaling D is frozen at the starting point, so each gauge is fixed exactly:
    w . theta keeps its starting value, with w = D v_null reported in
    ``FitResult.gauge_constraints``;
  * Sigma_theta is the pseudo-inverse of J^T J along identifiable directions;
  * null directions are reported in parameter space (README 13: report
    invariants first, parameters second).

Certificates of the final solve at theta_hat are part of the result; a fit
whose final solve is not certified is reported as such (README 10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .solve import solve


@dataclass
class Observation:
    """An observed level, or a transition when lower_block/lower_index are set.

    For a transition, `block`/`index` name the upper level and `energy` is the
    observed wavenumber E_upper - E_lower.
    """
    block: str            # BlockId key
    index: int            # level index within the block (ascending energy)
    energy: float         # observed level or wavenumber (cm^-1)
    sigma: float          # 1-sigma uncertainty (cm^-1)
    label: str | None = None
    true_energy: float | None = None  # synthetic data only
    lower_block: str | None = None
    lower_index: int | None = None
    # Blend: observed position = sum_k w_k (E_u,k - E_l,k), weights summing to 1
    # (intensity-weighted); entries (upper_block, upper_index, lower_block, lower_index, w).
    blend: list | None = None

    @property
    def is_line(self) -> bool:
        return self.lower_block is not None

    def components(self):
        if self.blend:
            return [tuple(c) for c in self.blend]
        if self.is_line:
            return [(self.block, self.index, self.lower_block, self.lower_index, 1.0)]
        return [(self.block, self.index, None, None, 1.0)]


@dataclass
class Prior:
    """Independent Gaussian prior on a subset of parameters (others flat)."""
    mean: Mapping[str, float]
    sigma: Mapping[str, float]


@dataclass
class FitResult:
    param_names: list[str]
    theta: dict[str, float]
    sigma: dict[str, float]           # nan for fixed or non-identifiable parameters
    covariance: np.ndarray            # full parameter order; zero rows/cols for fixed
    free: list[str]
    fixed: list[str]
    null_directions: list[dict[str, float]]
    gauge_constraints: list[dict[str, float]]  # w with w . theta held at its starting value
    singular_values: list[float]      # of the whitened, column-scaled Jacobian
    rank: int
    chi2: float
    dof: int
    n_obs: int
    rms: float                        # unweighted rms of E_obs - E_calc (cm^-1)
    iterations: int
    converged: bool
    certified: bool
    failed_certificates: list[dict]
    levels: list[dict]                # per observation: observed, calculated, residual, sigma_pred
    history: list[float] = field(default_factory=list)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            c = self.covariance / np.outer(d, d)
        return np.nan_to_num(c)


def _vector(names, theta) -> np.ndarray:
    if isinstance(theta, Mapping):
        missing = [n for n in names if n not in theta]
        extra = [k for k in theta if k not in names]
        if missing or extra:
            raise ValueError(f"theta must give exactly {names}; missing {missing}, unknown {extra}")
        return np.array([float(theta[n]) for n in names])
    v = np.asarray(theta, dtype=float)
    if v.shape != (len(names),):
        raise ValueError(f"theta must have {len(names)} entries")
    return v


class _Evaluator:
    def __init__(self, model, observations, solve_kw):
        self.obs = list(observations)
        if not self.obs:
            raise ValueError("no observations")
        keys = {b["key"]: i for i, b in enumerate(model.blocks)}
        used = []
        for o in self.obs:
            for ub, _, lb, _, _ in o.components():
                for blk in (ub, lb):
                    if blk is None:
                        continue
                    if blk not in keys:
                        raise ValueError(f"observation refers to unknown block {blk!r}")
                    if keys[blk] not in used:
                        used.append(keys[blk])
            if not (o.sigma > 0):
                raise ValueError(f"observation {o.label or (o.block, o.index)} needs sigma > 0")
            if o.blend and not math.isclose(sum(c[4] for c in o.blend), 1.0, rel_tol=1e-9):
                raise ValueError(f"blend weights of {o.label} must sum to 1")
        used.sort()
        # Solve only the blocks that carry observations.
        self.model = model.subset(used)
        local = {model.blocks[g]["key"]: i for i, g in enumerate(used)}
        self.comps = []
        need = {}
        for o in self.obs:
            cs = []
            for ub, ui, lb, li, w in o.components():
                u = local[ub]
                need[u] = max(need.get(u, 0), ui + 1)
                low = None
                if lb is not None:
                    low = (local[lb], li)
                    need[low[0]] = max(need.get(low[0], 0), li + 1)
                cs.append(((u, ui), low, float(w)))
            self.comps.append(cs)
        self.nlowest = max(need.values())
        self.solve_kw = solve_kw
        self.e_obs = np.array([o.energy for o in self.obs])
        self.sig = np.array([o.sigma for o in self.obs])

    def __call__(self, theta):
        sol = solve(self.model, list(theta), nlowest=self.nlowest, keep_vectors=True, **self.solve_kw)
        jac_cache = {}
        e_calc = np.empty(len(self.obs))
        jac = np.empty((len(self.obs), len(theta)))
        def level(b, idx, key):
            res = sol.results[b]
            if idx >= len(res["eigenvalues"]):
                raise RuntimeError(f"block {key} returned {len(res['eigenvalues'])} levels; "
                                   f"observation needs index {idx}")
            if b not in jac_cache:
                jac_cache[b] = self.model.jacobian(b, np.asarray(res["eigenvectors"]))
            return res["eigenvalues"][idx], jac_cache[b][idx]

        for row, (o, cs) in enumerate(zip(self.obs, self.comps)):
            e_row, g_row = 0.0, 0.0
            for (u, ui), low, w in cs:
                e, g = level(u, ui, o.block)
                if low is not None:
                    el, gl = level(low[0], low[1], o.lower_block)
                    e, g = e - el, g - gl
                e_row += w * e
                g_row = g_row + w * g
            e_calc[row] = e_row
            jac[row] = g_row
        return e_calc, jac, sol


def fit_by_order(models, observations: Sequence[Observation], theta0, **fit_kw) -> list[FitResult]:
    """Order continuation: fit a sequence of nested models (e.g. Watson order
    4 -> 6 -> 8), starting each from the previous solution with the new
    parameters at 0. Parameters spanning many decades (A ~ 10, L ~ 1e-13 cm^-1)
    make a cold start of a high-order fit unreliable; this is the robust path."""
    results = []
    theta = dict(theta0)
    for m in models:
        start = {n: float(theta.get(n, 0.0)) for n in m.param_names}
        r = fit(m, observations, start, **fit_kw)
        results.append(r)
        theta = r.theta
    return results


def fit(model, observations: Sequence[Observation], theta0, *, prior: Prior | None = None,
        fixed: Sequence[str] = (), max_iter: int = 50, rtol: float = 1e-12, rank_tol: float = 1e-9,
        lam0: float = 1e-3, scale_cov_by_chi2: bool = False, model_error: float = 0.0,
        **solve_kw) -> FitResult:
    """model_error: additive model-error sigma tau (cm^-1); every observation is
    weighted with sqrt(sigma_i^2 + tau^2). Estimate it with
    riss_drucker.posterior.calibrate_model_error (README 5.4 hyperparameter)."""
    names = list(model.param_names)
    theta = _vector(names, theta0)
    fixed = list(fixed)
    for f in fixed:
        if f not in names:
            raise ValueError(f"unknown fixed parameter {f!r}")
    free_idx = [i for i, n in enumerate(names) if n not in fixed]
    if not free_idx:
        raise ValueError("no free parameters")
    if model_error < 0:
        raise ValueError("model_error must be >= 0")
    ev = _Evaluator(model, observations, solve_kw)
    if model_error > 0:
        ev.sig = np.sqrt(ev.sig**2 + model_error**2)

    p_idx, p_mu, p_s = [], [], []
    if prior is not None:
        for n, s in prior.sigma.items():
            if n not in names:
                raise ValueError(f"prior on unknown parameter {n!r}")
            if not (s > 0):
                raise ValueError(f"prior sigma for {n} must be > 0")
            if names.index(n) in free_idx:
                p_idx.append(free_idx.index(names.index(n)))
                p_mu.append(float(prior.mean[n]))
                p_s.append(float(s))
    p_idx, p_mu, p_s = np.array(p_idx, int), np.array(p_mu), np.array(p_s)

    def system(th):
        e_calc, jac, sol = ev(th)
        r = (ev.e_obs - e_calc) / ev.sig
        J = jac[:, free_idx] / ev.sig[:, None]
        if len(p_idx):
            rp = (p_mu - th[np.array(free_idx)][p_idx]) / p_s
            Jp = np.zeros((len(p_idx), len(free_idx)))
            Jp[np.arange(len(p_idx)), p_idx] = 1.0 / p_s
            r, J = np.concatenate([r, rp]), np.vstack([J, Jp])
        return r, J, e_calc, jac, sol

    d0 = None  # column scaling, frozen at the starting point

    def svd(J):
        nonlocal d0
        if d0 is None:
            d0 = np.linalg.norm(J, axis=0)
            d0[d0 == 0] = 1.0
        d = d0
        U, S, Vt = np.linalg.svd(J / d, full_matrices=False)
        rank = int(np.sum(S > rank_tol * (S[0] if len(S) and S[0] > 0 else 1.0)))
        return d, U, S, Vt, rank

    r, J, e_calc, jac, sol = system(theta)
    chi2 = float(r @ r)
    history = [chi2]
    lam = lam0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        d, U, S, Vt, rank = svd(J)
        Ur, Sr, Vr = U[:, :rank], S[:rank], Vt[:rank].T
        g = Ur.T @ r
        accepted = False
        for _ in range(12):
            step_s = Vr @ (Sr / (Sr**2 + lam * Sr[0] ** 2) * g)
            step = step_s / d
            trial = theta.copy()
            trial[free_idx] += step
            r_t, J_t, e_t, jac_t, sol_t = system(trial)
            chi2_t = float(r_t @ r_t)
            if chi2_t <= chi2:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            converged = True  # no downhill step at any damping: at a minimum (to rounding)
            break
        dchi = chi2 - chi2_t
        theta, r, J, e_calc, jac, sol, chi2 = trial, r_t, J_t, e_t, jac_t, sol_t, chi2_t
        history.append(chi2)
        lam = max(lam / 10.0, 1e-12)
        if dchi <= rtol * max(chi2, 1.0) and np.max(np.abs(step_s)) < 1e-8:
            converged = True
            break

    d, U, S, Vt, rank = svd(J)
    Vr, Sr = Vt[:rank].T, S[:rank]
    cov_s = (Vr / Sr**2) @ Vr.T
    cov_free = cov_s / np.outer(d, d)
    n_obs = len(ev.obs)
    dof = n_obs + len(p_idx) - rank
    if scale_cov_by_chi2 and dof > 0:
        cov_free = cov_free * (chi2 / dof)
    cov = np.zeros((len(names), len(names)))
    cov[np.ix_(free_idx, free_idx)] = cov_free

    nulls, constraints = [], []
    for v in Vt[rank:]:
        w = v * d
        constraints.append({names[free_idx[i]]: float(c) for i, c in enumerate(w) if abs(c) > 1e-300})
        vt = v / d
        vt = vt / np.linalg.norm(vt)
        k = int(np.argmax(np.abs(vt)))
        vt = vt * np.sign(vt[k])
        nulls.append({names[free_idx[i]]: float(c) for i, c in enumerate(vt) if abs(c) > 1e-12})

    null_params = set()
    for nd in nulls:
        null_params.update(k for k, v in nd.items() if abs(v) > 1e-6)
    sigma = {}
    for i, n in enumerate(names):
        sigma[n] = float("nan") if (n in fixed or n in null_params) else float(np.sqrt(max(cov[i, i], 0.0)))

    levels = []
    for row, o in enumerate(ev.obs):
        g = jac[row]
        levels.append({
            "block": o.block, "index": o.index, "label": o.label, "observed": o.energy,
            **({"lower_block": o.lower_block, "lower_index": o.lower_index} if o.is_line else {}),
            **({"blend": [list(c) for c in o.blend]} if o.blend else {}),
            "calculated": float(e_calc[row]), "residual": float(o.energy - e_calc[row]), "sigma_obs": o.sigma,
            "sigma_pred": float(np.sqrt(max(g @ cov @ g, 0.0))),
            **({"true_energy": o.true_energy} if o.true_energy is not None else {}),
        })
    resid = ev.e_obs - e_calc
    return FitResult(
        param_names=names, theta={n: float(v) for n, v in zip(names, theta)}, sigma=sigma, covariance=cov,
        free=[names[i] for i in free_idx], fixed=fixed, null_directions=nulls, gauge_constraints=constraints,
        singular_values=[float(s) for s in S], rank=rank, chi2=chi2, dof=dof, n_obs=n_obs,
        rms=float(np.sqrt(np.mean(resid**2))), iterations=it, converged=converged,
        certified=sol.certified, failed_certificates=sol.failed_certificates(), levels=levels, history=history)
