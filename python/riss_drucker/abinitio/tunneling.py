"""One-dimensional reaction-path tunneling between equivalent minima.

    H = -(hbar^2/2) d^2/ds^2 + V(s) + ZPE_perp(s)

s is the mass-weighted arc length along the minimum-energy path (amu^1/2 Angstrom), V the
electronic energy and ZPE_perp the harmonic zero-point energy of the modes orthogonal to the
path (projected Hessians). For a symmetric double well (minimum - saddle - equivalent minimum)
only the half path from the saddle to one minimum is needed: the profile is mirrored about
the saddle, and beyond the minimum it is mirrored about the minimum (a harmonic-like outer
wall). The splitting of the lowest pair is 2|sigma|, sigma the tunneling matrix element.

Approximations (stated in results): adiabatic separation of the orthogonal modes, no
corner cutting (small-curvature effects are neglected, so splittings are lower bounds in
that respect), harmonic ZPE.
"""
from __future__ import annotations

import numpy as np

HBAR2_HALF = 16.8576292   # hbar^2 / (2 amu Angstrom^2) in cm^-1


def kabsch(ref: np.ndarray, X: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """X rotated/translated onto ref (mass-weighted least squares, proper rotation)."""
    w = masses / masses.sum()
    cr, cx = (w[:, None] * ref).sum(0), (w[:, None] * X).sum(0)
    A, B = ref - cr, X - cx
    H = (B * w[:, None]).T @ A
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return B @ R.T + cr


def arc_length(path: list, masses: np.ndarray) -> np.ndarray:
    """Mass-weighted arc length along consecutive (aligned) geometries."""
    s = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        b = kabsch(a, b, masses)
        s.append(s[-1] + float(np.sqrt(np.sum(masses[:, None] * (b - a) ** 2))))
    return np.array(s)


def double_well_levels(s_half: np.ndarray, U_half: np.ndarray, *, n_grid: int = 600, nlev: int = 6):
    """Levels of the symmetric double well built from the half profile U(s) given from the
    saddle (s = 0) to the minimum (s = s_min). Returns (levels relative to the profile minimum,
    grid, potential)."""
    from scipy.interpolate import CubicSpline
    s_half = np.asarray(s_half, float)
    U_half = np.asarray(U_half, float)
    smin = s_half[-1]
    f = CubicSpline(s_half, U_half, bc_type=((1, 0.0), (1, 0.0)))   # flat at saddle and minimum

    def U(x):
        x = np.abs(x)
        return np.where(x <= smin, f(np.minimum(x, smin)), f(np.clip(2 * smin - x, 0, smin)))
    L = 2 * smin
    x = np.linspace(-L, L, n_grid)
    dx = x[1] - x[0]
    # sinc-DVR kinetic energy (Colbert-Miller)
    i = np.arange(n_grid)
    D = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(D == 0, np.pi**2 / 3, 2.0 / np.where(D == 0, 1, D) ** 2 * (-1.0) ** D)
    T = HBAR2_HALF * T / dx**2
    V = U(x)
    w = np.linalg.eigvalsh(T + np.diag(V))
    return w[:nlev] - V.min(), x, V


def splitting(s_half, U_half, **kw) -> dict:
    lev, x, V = double_well_levels(s_half, U_half, **kw)
    return {"levels": lev.tolist(), "pair_splitting": float(lev[1] - lev[0]), "sigma_abs": float((lev[1] - lev[0]) / 2),
            "barrier_above_ground": float(np.max(V) - V.min() - lev[0])}
