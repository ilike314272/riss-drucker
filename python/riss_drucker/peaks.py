"""From a measured spectrum to a peak list (M3 gap: peak finding + profile fit).

simulate_spectrum  Voigt-convolved stick spectrum with Gaussian noise (also the
                   Part 4 "simulated vs measured" overlay).
find_peaks         Candidates from a matched filter (the line profile) above an SNR
                   threshold, grouped into clusters whose profiles overlap, then a
                   joint least-squares fit of a sum of Voigt profiles plus a
                   local linear baseline (absorbs wings of lines outside the
                   window). Fitted peaks below the area SNR threshold are dropped.
                   Position and area uncertainties come from the fit covariance
                   scaled by the residual noise, so they are calibrated (tests
                   check unit pull widths).

Profiles: Voigt with Gaussian sigma_g and Lorentzian HWHM gamma; widths shared
within a cluster (instrument + Doppler + pressure are common to neighbours).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .assign import Peak


def voigt(x, center, area, sigma_g, gamma):
    from scipy.special import voigt_profile

    return area * voigt_profile(np.asarray(x) - center, sigma_g, gamma)


def voigt_fwhm(sigma_g: float, gamma: float) -> float:
    """Olivero-Longbothum approximation (0.02% accurate)."""
    fg, fl = 2 * math.sqrt(2 * math.log(2)) * sigma_g, 2 * gamma
    return 0.5346 * fl + math.sqrt(0.2166 * fl * fl + fg * fg)


def simulate_spectrum(positions, areas, x, *, sigma_g: float, gamma: float, noise: float = 0.0,
                      baseline: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.full(len(x), float(baseline))
    for p, a in zip(positions, areas):
        w = 30 * voigt_fwhm(sigma_g, gamma)
        lo, hi = np.searchsorted(x, [p - w, p + w])
        y[lo:hi] += voigt(x[lo:hi], p, a, sigma_g, gamma)
    if noise > 0:
        y = y + rng.normal(0.0, noise, len(x))
    return y


@dataclass
class FittedPeak:
    position: float
    position_sigma: float
    area: float
    area_sigma: float
    sigma_g: float
    gamma: float
    cluster: int
    chi2_reduced: float

    def to_peak(self) -> Peak:
        slog = self.area_sigma / self.area if self.area > 0 else 1.0
        return Peak(position=self.position, sigma=self.position_sigma, intensity=self.area,
                    sigma_log_intensity=float(slog))


def estimate_noise(y) -> float:
    """Robust noise sigma from first differences (MAD), insensitive to sparse peaks."""
    d = np.diff(np.asarray(y))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))


def find_peaks(x, y, *, sigma_g: float, gamma: float, noise: float | None = None, min_snr: float = 5.0,
               fit_widths: bool = False, window_fwhm: float = 3.0, baseline: float = 0.0) -> list[FittedPeak]:
    """Peak list with calibrated uncertainties from a uniformly sampled spectrum."""
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks as sp_find_peaks
    from scipy.special import voigt_profile

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - baseline
    dx = float(np.median(np.diff(x)))
    noise = estimate_noise(y) if noise is None else float(noise)
    fw = voigt_fwhm(sigma_g, gamma)
    height0 = voigt_profile(0.0, sigma_g, gamma)  # peak height per unit area
    # Matched filter: correlate with the unit-area profile; its noise is
    # noise * ||k||, and a line of area a gives a peak ~ a * ||k||^2 / dx.
    half = int(math.ceil(3 * fw / dx))
    k = voigt_profile(np.arange(-half, half + 1) * dx, sigma_g, gamma)
    k = k / np.linalg.norm(k)
    yf = np.convolve(y, k[::-1], mode="same")
    idx, _ = sp_find_peaks(yf, height=min_snr * noise, distance=max(1, int(0.5 * fw / dx)))
    if len(idx) == 0:
        return []
    # Clusters: candidates closer than window_fwhm * FWHM are fitted jointly.
    clusters, cur = [], [idx[0]]
    for i in idx[1:]:
        if x[i] - x[cur[-1]] < window_fwhm * fw:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)

    out: list[FittedPeak] = []
    for ci, cl in enumerate(clusters):
        lo, hi = np.searchsorted(x, [x[cl[0]] - window_fwhm * fw, x[cl[-1]] + window_fwhm * fw])
        xs, ys = x[lo:hi], y[lo:hi]
        n = len(cl)
        xc = 0.5 * (xs[0] + xs[-1])
        p0 = []
        for i in cl:
            p0 += [x[i], max(y[i], noise) / height0]
        p0 += [0.0, 0.0]  # local baseline: offset, slope
        if fit_widths:
            p0 += [sigma_g, gamma]
        nb = 2 * n  # index of the baseline parameters

        def model(p):
            sg, gm = (p[-2], p[-1]) if fit_widths else (sigma_g, gamma)
            m = p[nb] + p[nb + 1] * (xs - xc)
            for k in range(n):
                m = m + voigt(xs, p[2 * k], p[2 * k + 1], abs(sg), abs(gm))
            return m

        res = least_squares(lambda p: (model(p) - ys) / noise, p0, x_scale="jac")
        dof = max(len(xs) - len(res.x), 1)
        chi2r = float(np.sum(res.fun**2) / dof)
        J = res.jac
        try:
            cov = np.linalg.pinv(J.T @ J) * max(chi2r, 1.0)
        except np.linalg.LinAlgError:
            cov = np.full((len(res.x), len(res.x)), np.nan)
        sg, gm = (abs(res.x[-2]), abs(res.x[-1])) if fit_widths else (sigma_g, gamma)
        for k in range(n):
            pos, area = res.x[2 * k], res.x[2 * k + 1]
            a_sig = math.sqrt(max(cov[2 * k + 1, 2 * k + 1], 0))
            if area <= 0 or not (xs[0] <= pos <= xs[-1]) or area < min_snr * a_sig:
                continue
            out.append(FittedPeak(float(pos), float(math.sqrt(max(cov[2 * k, 2 * k], 0))), float(area),
                                  float(math.sqrt(max(cov[2 * k + 1, 2 * k + 1], 0))), float(sg), float(gm),
                                  ci, chi2r))
    out.sort(key=lambda p: p.position)
    return out
