"""Synthetic data (README 5.7): assigned levels and line spectra from a known
theta, with noise, and ground truth kept with every observation or peak.
Level-sequence generators (beta-Hermite, Brody) live in riss_drucker.stats."""
from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np

from .fit import Observation
from .solve import solve


def synthesize_levels(model, theta: Mapping[str, float], *, noise: float = 0.0, seed: int = 0,
                      nlowest: int | None = None, block_filter: Callable[[dict], bool] | None = None,
                      labels: Mapping[str, list[str]] | None = None, threads: int | None = None) -> list[Observation]:
    """All (or the lowest `nlowest`) levels of the selected blocks as observations.

    `energy` = true level + N(0, noise^2); `sigma` = noise (or 1e-12 if noise == 0).
    """
    rng = np.random.default_rng(seed)
    n = nlowest if nlowest is not None else max(b["dim"] for b in model.blocks)
    sol = solve(model, theta, nlowest=n, method="dense", threads=threads)
    if not sol.certified:
        raise RuntimeError(f"synthetic solve not certified: {sol.failed_certificates()[:3]}")
    obs = []
    for b, r in zip(sol.blocks, sol.results):
        if block_filter is not None and not block_filter(b):
            continue
        labs = labels.get(b["key"]) if labels else None
        for i, e in enumerate(r["eigenvalues"]):
            obs.append(Observation(block=b["key"], index=i, energy=float(e + noise * rng.standard_normal()),
                                   sigma=noise if noise > 0 else 1e-12,
                                   label=labs[i] if labs else None, true_energy=float(e)))
    return obs


def perturb_theta(theta: Mapping[str, float], rel: float, seed: int = 0) -> dict[str, float]:
    """theta_k * (1 + rel * N(0,1)), a starting point for recovery tests."""
    rng = np.random.default_rng(seed)
    return {k: float(v * (1 + rel * rng.standard_normal())) for k, v in theta.items()}


def synthesize_spectrum(model, theta: Mapping[str, float], mu, T: float, *, fwhm: float = 1e-3,
                        snr: float = 2000.0, detect_snr: float = 3.0, sigma_log_intensity: float = 0.15,
                        spurious_density: float = 0.0, seed: int = 0, spin_weight=None):
    """Peaks from the true spectrum (README 5.7): selection rules, Boltzmann and
    spin weights, Gaussian position noise sigma = FWHM / SNR_line, log-normal
    intensity noise, a detection threshold, and optional spurious peaks.

    Lines closer than fwhm/2 are one blended peak (assign.merge_blends).
    Returns (peaks, true_lines). peak.truth is the true (blend) line id or None.
    """
    from .assign import Peak, merge_blends, predict_lines

    rng = np.random.default_rng(seed)
    pred, lines = predict_lines(model, theta, mu, T, spin_weight=spin_weight)
    pred = merge_blends(pred, fwhm / 2)
    imax = max(p.intensity for p in pred)
    peaks = []
    for p in pred:
        s = snr * p.intensity / imax
        if s < detect_snr:
            continue
        sig = fwhm / s
        peaks.append(Peak(position=float(p.position + sig * rng.standard_normal()), sigma=sig,
                          intensity=float(p.intensity * math.exp(sigma_log_intensity * rng.standard_normal())),
                          sigma_log_intensity=sigma_log_intensity, truth=p.id))
    if spurious_density > 0 and peaks:
        lo, hi = min(p.position for p in peaks), max(p.position for p in peaks)
        for _ in range(rng.poisson(spurious_density * (hi - lo))):
            s = detect_snr * math.exp(abs(rng.normal(0, 1)))
            peaks.append(Peak(position=float(rng.uniform(lo, hi)), sigma=fwhm / s,
                              intensity=float(imax * s / snr), sigma_log_intensity=sigma_log_intensity, truth=None))
    peaks.sort(key=lambda p: p.position)
    return peaks, pred
