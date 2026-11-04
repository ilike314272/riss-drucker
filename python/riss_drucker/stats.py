"""Level-sequence statistics and generators (README 5.7, 12).

* beta_hermite_levels: Dumitriu-Edelman tridiagonal beta-Hermite ensemble
  (beta = 1 GOE, 2 GUE, 4 GSE); beta = 0 gives Poisson (uncorrelated) levels.
* circular_phases: COE/CUE eigenphases (uniform density, no unfolding needed).
* Brody spacing sampler and maximum-likelihood fit of q.
* gap ratio <r> (needs no unfolding); reference values below.
* superposition of independent sequences: P(0) = 1 - sum f_i^2.

Caution (README 13): a low fitted Brody q is ambiguous (weak chaos versus
missing or mislabelled levels). Do not read it physically without an
independent level count (inertia counts, density of states).
"""
from __future__ import annotations

import math

import numpy as np

# Large-N <r> (Atas et al. 2013); Poisson exactly 2 ln 2 - 1.
GAP_RATIO_REFERENCE = {"poisson": 2 * math.log(2) - 1, "goe": 0.5307, "gue": 0.5996, "gse": 0.6744}


def beta_hermite_levels(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Sorted eigenvalues of an n x n beta-Hermite matrix (beta = 0: Poisson)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if beta == 0:
        return np.sort(rng.uniform(0.0, n, n))
    if beta < 0:
        raise ValueError("beta must be >= 0")
    diag = rng.normal(0.0, math.sqrt(2.0), n)
    off = np.sqrt(rng.chisquare(beta * np.arange(n - 1, 0, -1)))
    from scipy.linalg import eigvalsh_tridiagonal  # scipy is in the Kaggle image

    return eigvalsh_tridiagonal(diag / math.sqrt(2.0), off / math.sqrt(2.0))


def circular_phases(n: int, beta: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted eigenphases in [0, 2 pi) of a COE (beta=1) or CUE (beta=2) matrix."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))  # Haar measure
    if beta == 2:
        u = q
    elif beta == 1:
        u = q.T @ q
    else:
        raise ValueError("circular_phases supports beta = 1 or 2")
    return np.sort(np.mod(np.angle(np.linalg.eigvals(u)), 2 * math.pi))


def gap_ratios(levels: np.ndarray) -> np.ndarray:
    """r_n = min(s_n, s_n+1) / max(s_n, s_n+1) of sorted levels (no unfolding)."""
    s = np.diff(np.sort(np.asarray(levels, dtype=float)))
    a, b = s[:-1], s[1:]
    mx = np.maximum(a, b)
    ok = mx > 0
    return np.minimum(a, b)[ok] / mx[ok]


def mean_gap_ratio(levels) -> float:
    r = gap_ratios(levels)
    return float(r.mean()) if r.size else float("nan")


def central(levels: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Middle `fraction` of a sorted spectrum (avoids ensemble edges)."""
    lv = np.sort(levels)
    k = int(len(lv) * (1 - fraction) / 2)
    return lv[k: len(lv) - k]


def brody_b(q: float) -> float:
    return math.gamma((q + 2) / (q + 1)) ** (q + 1)


def brody_pdf(s, q: float):
    b = brody_b(q)
    s = np.asarray(s, dtype=float)
    return (q + 1) * b * s**q * np.exp(-b * s ** (q + 1))


def brody_spacings(n: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean spacings from the Brody distribution (q = 0 Poisson, q = 1 Wigner-like)."""
    if not 0 <= q <= 1:
        raise ValueError("Brody q must be in [0, 1]")
    return (rng.exponential(1.0, n) / brody_b(q)) ** (1.0 / (q + 1))


def fit_brody(spacings) -> float:
    """Maximum-likelihood Brody q in [0, 1] for unit-mean (unfolded) spacings."""
    s = np.asarray(spacings, dtype=float)
    s = s[s > 0] / np.mean(s)
    from scipy.optimize import minimize_scalar

    nll = lambda q: -np.sum(np.log(brody_pdf(s, q)))
    return float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)


def superposition_p0(fractions) -> float:
    """Spacing density at s = 0 for independent sequences with level repulsion."""
    f = np.asarray(fractions, dtype=float)
    if not np.isclose(f.sum(), 1.0):
        raise ValueError("fractions must sum to 1")
    return float(1.0 - np.sum(f**2))


def spacing_density_at_zero(unfolded_spacings, width: float = 0.05, nbins: int = 4) -> float:
    """P(0) estimated by a straight-line fit to the first histogram bins."""
    s = np.asarray(unfolded_spacings)
    edges = np.arange(nbins + 1) * width
    counts, _ = np.histogram(s, bins=edges)
    dens = counts / (len(s) * width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    slope, intercept = np.polyfit(centers, dens, 1)
    return float(intercept)
