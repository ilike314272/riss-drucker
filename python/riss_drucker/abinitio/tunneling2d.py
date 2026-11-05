"""Two-dimensional reaction-surface tunneling (path s + effective transverse mode q).

Miller-Handy-Adams reaction-path Hamiltonian restricted to the curvature direction u(s):

    metric g = diag(eta^2, 1),  eta(s, q) = 1 - kappa(s) q,
    T = -hbar^2/2 [ eta^-1 d/ds eta^-1 d/ds + eta^-1 d/dq eta d/dq ],
    U(s, q) = V(s) + ZPE_rest(s) + V_q(s, q).

Variational solution of the quadratic form
    E[psi] = int [ HBAR2_HALF ( eta^-1 |d_s psi|^2 + eta |d_q psi|^2 ) + U |psi|^2 eta ] ds dq
on a uniform grid (second-order finite elements of the derivatives at half points), i.e. the
generalized symmetric problem K psi = E M psi with M = diag(eta). The double well is built from
the half path (saddle at s = 0, minimum at s = smin) mirrored about the saddle and, beyond the
minimum, about the minimum (as in the 1D solver). kappa = 0 recovers the 1D path model plus an
uncoupled transverse oscillator, which the tests check.

Corner cutting enters through eta: the wave function shortcuts the curved path on its inner side.
Neglected: twisting of u(s) (coupling to the other transverse modes), anharmonic coupling of the
remaining modes (they enter as harmonic adiabatic ZPE).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

HBAR2_HALF = 16.8576292


def mirrored(f_half, smin):
    """Even extension of a function given on [0, smin] to [-2 smin, 2 smin] (mirror at 0 and smin)."""
    def F(s, *a):
        x = np.abs(s)
        x = np.where(x <= smin, x, 2 * smin - x)
        return f_half(np.clip(x, 0, smin), *a)
    return F


def solve_2d(smin, U_path, kappa, Vq, qmax, *, ns=241, nq=41, nlev=6, eta_min=0.08, bc="dirichlet"):
    """Levels (relative to the minimum of U on the grid) and wave functions of the 2D double well.

    U_path(s): V + ZPE_rest on [0, smin]; kappa(s): curvature; Vq(s, q): transverse potential.
    The q range must stay inside the caustic q < 1/kappa (curvilinear coordinates fold there); bc is
    'dirichlet' (psi = 0 outside the grid, an upper bound) or 'neumann'.
    Returns dict with levels, pair splitting, grid and eigenvectors (psi normalized with eta)."""
    from scipy.sparse.linalg import eigsh
    Uf, Kf = mirrored(U_path, smin), mirrored(kappa, smin)
    Vqf = mirrored(lambda x, q: Vq(x, q), smin)
    s = np.linspace(-2 * smin, 2 * smin, ns)
    q = np.linspace(-qmax, qmax, nq)
    ds, dq = s[1] - s[0], q[1] - q[0]
    S, Qg = np.meshgrid(s, q, indexing="ij")
    kap = Kf(s)
    eta = 1.0 - kap[:, None] * q[None, :]
    if eta.min() < eta_min:
        raise ValueError(f"eta = 1 - kappa q reaches {eta.min():.3f} on the grid: reduce qmax "
                         f"(kappa max {kap.max():.2f}, qmax {qmax:.3f})")
    U = Uf(s)[:, None] + Vqf(S, Qg)
    n = ns * nq
    idx = np.arange(n).reshape(ns, nq)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        vals.append(v)
    # s-derivative term: coefficient eta^-1 at (i+1/2, j)
    eh = 0.5 * (eta[1:, :] + eta[:-1, :])
    cs = HBAR2_HALF / eh * dq / ds
    for a, b, c in ((idx[:-1, :], idx[1:, :], cs),):
        a, b, c = a.ravel(), b.ravel(), c.ravel()
        rows += list(a) + list(b) + list(a) + list(b)
        cols += list(a) + list(b) + list(b) + list(a)
        vals += list(c) + list(c) + list(-c) + list(-c)
    # q-derivative term: coefficient eta at (i, j+1/2)
    eq = 0.5 * (eta[:, 1:] + eta[:, :-1])
    cq = HBAR2_HALF * eq * ds / dq
    a, b, c = idx[:, :-1].ravel(), idx[:, 1:].ravel(), cq.ravel()
    rows += list(a) + list(b) + list(a) + list(b)
    cols += list(a) + list(b) + list(b) + list(a)
    vals += list(c) + list(c) + list(-c) + list(-c)
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if bc == "dirichlet":                            # psi = 0 one step outside the grid
        d = np.zeros((ns, nq))
        d[:, 0] += HBAR2_HALF * eta[:, 0] * ds / dq
        d[:, -1] += HBAR2_HALF * eta[:, -1] * ds / dq
        d[0, :] += HBAR2_HALF / eta[0, :] * dq / ds
        d[-1, :] += HBAR2_HALF / eta[-1, :] * dq / ds
        K = K + sp.diags(d.ravel())
    w = (eta * ds * dq).ravel()
    K = K + sp.diags((U * eta * ds * dq).ravel())
    M = sp.diags(w)
    # symmetric standard form: M^-1/2 K M^-1/2
    mi = sp.diags(1 / np.sqrt(w))
    A = (mi @ K @ mi).tocsc()
    shift = float(U.min()) - 10.0
    ev, vec = eigsh(A, k=nlev, sigma=shift, which="LM")
    o = np.argsort(ev)
    ev, vec = ev[o], vec[:, o]
    psi = (mi @ vec)                                     # normalized: sum psi^2 w = 1
    lev = ev - float(U.min())
    return {"levels": lev.tolist(), "pair_splitting": float(lev[1] - lev[0]), "sigma_abs": float((lev[1] - lev[0]) / 2),
            "s": s, "q": q, "eta": eta, "U": U, "psi": psi.T.reshape(nlev, ns, nq), "weight": w.reshape(ns, nq),
            "grid": {"ns": ns, "nq": nq, "qmax": qmax}}


def converged_splitting(smin, U_path, kappa, Vq, qmax, *, grids=((161, 31), (241, 41), (321, 51)), **kw):
    """Splitting on successively finer grids (a convergence certificate)."""
    out = []
    for ns, nq in grids:
        r = solve_2d(smin, U_path, kappa, Vq, qmax, ns=ns, nq=nq, **kw)
        out.append({"ns": ns, "nq": nq, "pair_splitting": r["pair_splitting"], "ground": r["levels"][0]})
    return out


def sct_effective_mass(s, kappa, omega_u):
    """Small-curvature tunneling effective mass ratio mu_eff/mu (Liu, Lu, Truhlar et al.,
    J. Am. Chem. Soc. 115, 2408 (1993)) for one effective transverse mode:
        a = kappa t,  t = sqrt(hbar / (mu omega)) (zero-point turning point, mass-weighted),
        mu_eff/mu = min(1, exp(-2a - a^2 + (dt/ds)^2)).
    omega_u in cm^-1; t in amu^1/2 Angstrom (t^2 = 2 HBAR2_HALF / omega)."""
    t = np.sqrt(2 * HBAR2_HALF / np.abs(omega_u))
    a = np.abs(kappa) * t
    dt = np.gradient(t, s)
    return np.minimum(1.0, np.exp(-2 * a - a ** 2 + dt ** 2))


def splitting_variable_mass(s_half, U_half, mu_ratio_half, *, n_grid=801):
    """1D double well with position-dependent mass, T = -HBAR2_HALF d/ds (1/mu(s)) d/ds (finite
    differences, Dirichlet ends); same mirroring as the 1D solver. Returns levels and splitting."""
    from scipy.interpolate import CubicSpline
    from scipy.linalg import eigh_tridiagonal
    smin = float(s_half[-1])
    fU = CubicSpline(s_half, U_half, bc_type=((1, 0.0), (1, 0.0)))
    fm = lambda x: np.interp(x, s_half, mu_ratio_half)
    F = mirrored(lambda x: fU(x), smin)
    Fm = mirrored(lambda x: fm(x), smin)
    x = np.linspace(-2 * smin, 2 * smin, n_grid)
    h = x[1] - x[0]
    xm = 0.5 * (x[1:] + x[:-1])
    c = HBAR2_HALF / Fm(xm) / h ** 2
    diag = np.r_[c, 0] + np.r_[0, c] + F(x)
    diag[0] += c[0]
    diag[-1] += c[-1]
    w = eigh_tridiagonal(diag, -c, select="i", select_range=(0, 5))[0]
    lev = w - F(x).min()
    return {"levels": lev.tolist(), "pair_splitting": float(lev[1] - lev[0]), "sigma_abs": float((lev[1] - lev[0]) / 2)}


def solve_plane(potential, lo, hi, h, *, nlev=4):
    """Cartesian 2D Schroedinger equation -HBAR2_HALF (d_x^2 + d_y^2) + V(x, y) on a uniform grid
    (spacing h, second-order finite differences, psi = 0 outside). potential(P) takes an (n, 2)
    array of points. Returns levels (relative to min V), eigenvectors on the grid and the grid."""
    from scipy.sparse.linalg import eigsh
    gx = np.arange(lo[0], hi[0] + h / 2, h)
    gy = np.arange(lo[1], hi[1] + h / 2, h)
    nx, ny = len(gx), len(gy)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    V = np.asarray(potential(G), float)
    D = lambda n: sp.diags([np.full(n, 2.0), np.full(n - 1, -1.0), np.full(n - 1, -1.0)], [0, -1, 1]) / h ** 2
    T = HBAR2_HALF * (sp.kron(D(nx), sp.identity(ny)) + sp.kron(sp.identity(nx), D(ny)))
    H = (T + sp.diags(V)).tocsc()
    ev, vec = eigsh(H, k=nlev, sigma=float(V.min()) - 10.0, which="LM")
    o = np.argsort(ev)
    return {"levels": (ev[o] - V.min()).tolist(), "pair_splitting": float(ev[o][1] - ev[o][0]),
            "psi": vec[:, o].T.reshape(nlev, nx, ny) / h, "gx": gx, "gy": gy, "V": V.reshape(nx, ny),
            "points": G, "h": h}


def plane_splitting_converged(potential, lo, hi, hs=(0.016, 0.012, 0.009)):
    """Pair splitting on a sequence of grids, plus the h^2 Richardson extrapolation from the last two."""
    rows = []
    for h in hs:
        r = solve_plane(potential, lo, hi, h)
        rows.append({"h": h, "pair_splitting": r["pair_splitting"], "ground": r["levels"][0]})
    h1, h2 = hs[-2], hs[-1]
    p1, p2 = rows[-2]["pair_splitting"], rows[-1]["pair_splitting"]
    extrap = p2 + (p2 - p1) * h2 ** 2 / (h1 ** 2 - h2 ** 2)
    return {"grids": rows, "pair_splitting_extrapolated": float(extrap),
            "relative_change_last_grid": float(abs(p2 - p1) / abs(p2)), "last": r}
