"""Posterior pushforward to levels (README 5.5).

level_windows       sigma_i^2 = g_i^T Sigma_theta g_i (Hellmann-Feynman g), windows
                    E_i +- k sigma_i, near-crossing flags (sigma_i / gap > 0.3), a
                    support flag (outside the J/energy range constrained by data)
                    and a Weyl-fence flag (window wider than a fraction of the
                    local mean level spacing from a smoothed counting function).
resample_near_crossings
                    Linearized windows fail where levels nearly cross, so only
                    the blocks holding flagged levels are re-solved at theta
                    samples drawn from N(theta_hat, Sigma_theta).
torsion_truncation  Second-order estimate of the shift from basis states
                    outside |m| <= m_max (README 5.5, truncation check).
coverage_by_band    Fit on all bands but one, predict the held-out band,
                    compare empirical and nominal coverage (README 12).
"""
from __future__ import annotations

import json
import math
from typing import Callable, Sequence

import numpy as np

from .fit import FitResult, Observation, fit
from .solve import solve


def _theta_vec(model, theta):
    return np.array([float(theta[n]) for n in model.param_names])


def _mean_spacing(levels: np.ndarray, i: int) -> float:
    """Local mean spacing at level i from a smoothed counting function N(E)."""
    n = len(levels)
    if n < 2:
        return float("inf")
    if n < 5:
        return float((levels[-1] - levels[0]) / (n - 1))
    deg = 3 if n >= 8 else 1
    x = (levels - levels[0]) / max(levels[-1] - levels[0], 1e-300)
    coef = np.polyfit(x, np.arange(n), deg)
    dn = np.polyval(np.polyder(coef), x[i]) / max(levels[-1] - levels[0], 1e-300)
    return float(1.0 / dn) if dn > 0 else float((levels[-1] - levels[0]) / (n - 1))


def level_windows(model, fitres: FitResult, *, k: float = 2.0, nlowest: int | None = None,
                  near_crossing_ratio: float = 0.3, weyl_fraction: float = 0.5,
                  observations: Sequence[Observation] | None = None, labels: dict | None = None) -> list[dict]:
    """Windows for every computed level at theta_hat (lowest `nlowest` per block, default all)."""
    n = nlowest or max(b["dim"] for b in model.blocks)
    sol = solve(model, fitres.theta, nlowest=n, keep_vectors=True, method="dense")
    cov = np.asarray(fitres.covariance)
    j_obs = e_obs = None
    if observations:
        js = [int(o.block.split("|")[0][3:]) / 2 for o in observations]
        j_obs = (min(js), max(js))
        # Energy support: highest observed level plus a noise-sized margin.
        e_obs = max(o.energy for o in observations) + 5.0 * max(o.sigma for o in observations)
    out = []
    for bi, (b, r) in enumerate(zip(sol.blocks, sol.results)):
        E = np.asarray(r["eigenvalues"])
        jac = model.jacobian(bi, np.asarray(r["eigenvectors"]))
        sig = np.sqrt(np.clip(np.einsum("ik,kl,il->i", jac, cov, jac), 0, None))
        for i, e in enumerate(E):
            gaps = [abs(e - E[j]) for j in (i - 1, i + 1) if 0 <= j < len(E)]
            gap = min(gaps) if gaps else float("inf")
            spacing = _mean_spacing(E, i)
            J = b["two_j"] / 2
            row = {
                "block": b["key"], "index": i, "label": labels[b["key"]][i] if labels else None, "J": J,
                "energy": float(e), "sigma": float(sig[i]), "window": (float(e - k * sig[i]), float(e + k * sig[i])),
                "gap": float(gap), "sigma_over_gap": float(sig[i] / gap) if gap > 0 else float("inf"),
                "near_crossing": bool(gap == 0 or sig[i] / gap > near_crossing_ratio),
                "mean_spacing": spacing,
                "beyond_weyl_fence": bool(k * sig[i] > weyl_fraction * spacing),
                "extrapolated": bool(j_obs is not None and (J < j_obs[0] or J > j_obs[1] or e > e_obs)),
                "certified": r["certified"],
            }
            out.append(row)
    return out


def sample_theta(fitres: FitResult, n: int, seed: int = 0) -> np.ndarray:
    """Samples from N(theta_hat, Sigma_theta); null directions are not moved."""
    rng = np.random.default_rng(seed)
    mean = np.array([fitres.theta[p] for p in fitres.param_names])
    w, V = np.linalg.eigh(np.asarray(fitres.covariance))
    w = np.clip(w, 0, None)
    return mean + (rng.standard_normal((n, len(w))) * np.sqrt(w)) @ V.T


def resample_near_crossings(model, fitres: FitResult, windows: list[dict], *, n_samples: int = 200,
                            seed: int = 0, quantiles=(0.025, 0.16, 0.5, 0.84, 0.975)) -> list[dict]:
    """Empirical level distributions for near-crossing levels only."""
    flagged = [w for w in windows if w["near_crossing"]]
    if not flagged:
        return []
    keys = [b["key"] for b in model.blocks]
    blocks = sorted({keys.index(w["block"]) for w in flagged})
    sub = model.subset(blocks)
    need = max(w["index"] for w in flagged) + 1
    samples = sample_theta(fitres, n_samples, seed)
    vals = {(w["block"], w["index"]): [] for w in flagged}
    for th in samples:
        sol = solve(sub, list(th), nlowest=need, method="dense")
        for b, r in zip(sol.blocks, sol.results):
            for (key, idx), store in vals.items():
                if key == b["key"]:
                    store.append(r["eigenvalues"][idx])
    out = []
    for w in flagged:
        v = np.asarray(vals[(w["block"], w["index"])])
        out.append({"block": w["block"], "index": w["index"], "label": w["label"], "linear_sigma": w["sigma"],
                    "sample_std": float(v.std(ddof=1)), "sample_mean": float(v.mean()),
                    "quantiles": dict(zip(quantiles, np.quantile(v, quantiles).tolist())), "n_samples": len(v)})
    return out


def torsion_truncation(model, theta, *, nlowest: int = 5) -> list[dict]:
    """Second-order shift estimate from free-rotor states with |m| > m_max.

    For level psi_i: delta_i ~ sum_n |<n|V|psi_i>|^2 / (E_i - E_n^0) over neglected n,
    with E_n^0 the diagonal energy of |m = n>. Reports |delta_i| per level.
    """
    if model.name != "torsion_rotor":
        raise ValueError("torsion_truncation needs a torsion_rotor model")
    spec = json.loads(model.spec_json)
    n_fold, m_max = spec["n_fold"], spec["m_max"]
    th = dict(zip(model.param_names, theta)) if not isinstance(theta, dict) else dict(theta)
    vterms = {int(p[1:]): th[p] for p in model.param_names if p.startswith("V")}
    sol = solve(model, th, nlowest=nlowest, keep_vectors=True, method="dense")
    out = []
    for b, r in zip(sol.blocks, sol.results):
        J = b["two_j"] // 2
        K = int(b["sub"].split("=")[1])
        comps = []  # per basis function: list of (m, coefficient)
        for lab in b["basis_labels"]:
            body = lab.split("=")[1]
            if body.startswith("+-"):
                m = int(body[2:].split(",")[0])
                s = 1 / math.sqrt(2)
                comps.append([(m, s), (-m, s if body.endswith("cos") else -s)])
            else:
                comps.append([(int(body), 1.0)])

        def e0(nm):
            e = th["F"] * nm * nm + sum(v / 2 for v in vterms.values())
            if "A" in th:
                e += th["A"] * K * K + th["B"] * (J * (J + 1) - K * K)
            if "D_Kp" in th:
                e += th["D_Kp"] * K * nm
            return e

        vecs = np.asarray(r["eigenvectors"])
        for i, E in enumerate(r["eigenvalues"]):
            coef: dict[int, float] = {}
            for bf, v in zip(comps, vecs[:, i]):
                for m, c in bf:
                    coef[m] = coef.get(m, 0.0) + c * v
            amp: dict[int, float] = {}
            for m, c in coef.items():
                for L, VL in vterms.items():
                    for nm in (m + L, m - L):
                        if abs(nm) > m_max:
                            amp[nm] = amp.get(nm, 0.0) - VL / 4 * c
            shift = sum(a * a / (E - e0(nm)) for nm, a in amp.items() if E != e0(nm))
            out.append({"block": b["key"], "index": i, "energy": float(E), "shift_estimate": float(abs(shift)),
                        "n_neglected_coupled": len(amp)})
    return out


def coverage_by_band(model, observations: Sequence[Observation], theta0, band_of: Callable[[Observation], str], *,
                     levels: Sequence[float] = (0.6827, 0.9545), **fit_kw) -> dict:
    """Leave-one-band-out coverage of predictive intervals (README 12, calibration).

    For each band: fit on the others, z = (E_obs - E_pred) / sqrt(sigma_pred^2 + sigma_obs^2)
    for the held-out levels; report the fraction inside each nominal level.
    """
    from scipy.stats import norm

    obs = list(observations)
    bands = sorted({band_of(o) for o in obs})
    if len(bands) < 2:
        raise ValueError("coverage_by_band needs at least two bands")
    report = {"nominal": list(levels), "bands": {}}
    all_z = []
    keys = [b["key"] for b in model.blocks]
    for band in bands:
        train = [o for o in obs if band_of(o) != band]
        test = [o for o in obs if band_of(o) == band]
        res = fit(model, train, theta0, **fit_kw)
        cov = np.asarray(res.covariance)
        need = max(o.index for o in test) + 1
        idx = sorted({keys.index(o.block) for o in test})
        sub = model.subset(idx)
        sol = solve(sub, res.theta, nlowest=need, keep_vectors=True, method="dense")
        z = []
        for o in test:
            bi = idx.index(keys.index(o.block))
            r = sol.results[bi]
            g = sub.jacobian(bi, np.asarray(r["eigenvectors"]))[o.index]
            sp = math.sqrt(max(g @ cov @ g, 0.0))
            z.append((o.energy - r["eigenvalues"][o.index]) / math.hypot(sp, o.sigma))
        z = np.abs(np.asarray(z))
        all_z.append(z)
        report["bands"][band] = {
            "n": int(len(z)), "train_n": len(train), "fit_reduced_chi2": res.reduced_chi2,
            "empirical": [float(np.mean(z <= norm.ppf(0.5 + lv / 2))) for lv in levels],
        }
    z = np.concatenate(all_z)
    report["overall"] = {
        "n": int(len(z)),
        "empirical": [float(np.mean(z <= norm.ppf(0.5 + lv / 2))) for lv in levels],
        "binomial_se": [float(math.sqrt(lv * (1 - lv) / len(z))) for lv in levels],
    }
    return report
