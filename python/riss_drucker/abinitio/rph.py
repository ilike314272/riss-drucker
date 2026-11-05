"""Reaction-path (internal-axis) Hamiltonian for one large-amplitude coordinate.

From relaxed ab initio geometries along a torsion path this builds, with no
fitted constants:
* the generalized inertia tensor I(gamma) (3 rotations + the torsion) of the
  body frame, its inverse G(gamma), and the Podolsky Hamiltonian
      H = 1/2 sum_ij pi_i G_ij pi_j + V'(gamma) + V(gamma) + dZPE(gamma)
  with V' = hbar^2 [-(G_tt L)'/8 + G_tt L^2/32], L = d ln det I / d gamma
  (the J-p cross terms reduce exactly to J_a {G_at, p}/2; rotational terms commute),
* the adiabatic zero-point energy of the other 3N-7 modes from Hessians projected
  on translations, rotations and the path tangent (Miller-Handy-Adams),
* each coefficient function expanded in the Fourier functions the symmetry group
  allows for that operator (chosen by the compiler's own invariance test, so half-
  integer orders appear exactly where a gauge element requires them),
* a rd.build.BuildSpec with all terms and their values.

Units: amu, Angstrom, radians, cm^-1. hbar^2/(amu A^2) = 33.715258 cm^-1.
"""
from __future__ import annotations

import math

import numpy as np

HBAR2 = 33.7152583                    # hbar^2 / (amu Angstrom^2) in cm^-1 (2 x 16.8576292)
HARTREE_TO_CM = 219474.6313705
FREQ_AU = 5140.48714                  # sqrt(Eh / (bohr^2 amu)) / (2 pi c) in cm^-1
BOHR = 0.529177210903
MASS = {"H": 1.00782503207, "O": 15.99491461956, "C": 12.0, "N": 14.0030740048, "D": 2.01410177812}


# ---- frames ------------------------------------------------------------------------------------

def h2o2_body_frame(geom):
    """geom: [[sym, x, y, z], ...] in the order O1 O2 H1 H2 (Angstrom).
    Body frame: origin COM, z along O1->O2, H1 projection at azimuth +gamma/2, H2 at -gamma/2.
    Returns (body coordinates (4,3), gamma in [0, 2pi), lab->body rotation rows, masses)."""
    sym = [g[0] for g in geom]
    X = np.array([g[1:] for g in geom], float)
    m = np.array([MASS[s] for s in sym])
    X = X - (m[:, None] * X).sum(0) / m.sum()
    z = X[1] - X[0]
    z /= np.linalg.norm(z)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - (ref @ z) * z
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    ang = [math.atan2((X[i] - X[0]) @ e2, (X[i] - X[0]) @ e1) for i in (2, 3)]   # azimuth about the O-O axis
    gamma = (ang[0] - ang[1]) % (2 * math.pi)
    a = ang[0] - gamma / 2
    x = math.cos(a) * e1 + math.sin(a) * e2
    y = np.cross(z, x)
    Rb = np.array([x, y, z])
    return X @ Rb.T, gamma, Rb, m


def extend_h2o2_path(gammas, coords):
    """Body coordinates on [0, pi] -> samples on [0, 4pi) from the G8 operations:
    r(-g) = (x, -y, z) r(g)  [E*],  r(g + 2pi) = R_z(pi) r(g)  [gauge]."""
    out_g, out_r = [], []
    mir = np.diag([1.0, -1, 1])
    rz = np.diag([-1.0, -1, 1])
    for g, r in zip(gammas, coords):
        for gg, rr in ((g, r), (-g, r @ mir), (g + 2 * math.pi, r @ rz), (-g + 2 * math.pi, r @ mir @ rz)):
            gg = gg % (4 * math.pi)
            if not any(abs(gg - x) < 1e-9 or abs(abs(gg - x) - 4 * math.pi) < 1e-9 for x in out_g):
                out_g.append(gg)
                out_r.append(rr)
    order = np.argsort(out_g)
    return np.array(out_g)[order], np.array(out_r)[order]


# ---- Fourier series on a 4pi period ------------------------------------------------------------

class HalfFourier:
    """f(g) = sum_k a_k cos(k g/2) + b_k sin(k g/2), k = 0..K (least squares)."""

    def __init__(self, g, f, K):
        self.K = K
        A = self._basis(np.asarray(g))
        self.coef, *_ = np.linalg.lstsq(A, np.asarray(f), rcond=None)
        self.resid = float(np.max(np.abs(A @ self.coef - f))) if len(g) else 0.0

    def _basis(self, g):
        cols = [np.ones_like(g)]
        for k in range(1, self.K + 1):
            cols += [np.cos(k * g / 2), np.sin(k * g / 2)]
        return np.stack(cols, axis=-1)

    def _dbasis(self, g):
        cols = [np.zeros_like(g)]
        for k in range(1, self.K + 1):
            cols += [-k / 2 * np.sin(k * g / 2), k / 2 * np.cos(k * g / 2)]
        return np.stack(cols, axis=-1)

    def __call__(self, g):
        return self._basis(np.asarray(g, float)) @ self.coef

    def deriv(self, g):
        return self._dbasis(np.asarray(g, float)) @ self.coef


def kinetic_tensor(gammas4, coords4, masses, K: int = 24):
    """Generalized inertia tensor and G = I^-1 on a fine grid; returns callables."""
    natm = coords4.shape[1]
    fits = [[HalfFourier(gammas4, coords4[:, i, c], K) for c in range(3)] for i in range(natm)]

    def r_and_dr(g):
        g = np.atleast_1d(g)
        r = np.stack([np.stack([fits[i][c](g) for c in range(3)], -1) for i in range(natm)], 1)
        dr = np.stack([np.stack([fits[i][c].deriv(g) for c in range(3)], -1) for i in range(natm)], 1)
        return r, dr

    def inertia(g):
        r, dr = r_and_dr(g)
        out = np.zeros((len(r), 4, 4))
        for n in range(len(r)):
            Irot = sum(masses[i] * ((r[n, i] @ r[n, i]) * np.eye(3) - np.outer(r[n, i], r[n, i]))
                       for i in range(natm))
            X = sum(masses[i] * np.cross(r[n, i], dr[n, i]) for i in range(natm))
            Itt = sum(masses[i] * dr[n, i] @ dr[n, i] for i in range(natm))
            out[n, :3, :3], out[n, :3, 3], out[n, 3, :3], out[n, 3, 3] = Irot, X, X, Itt
        return out
    fit_err = max(f.resid for row in fits for f in row)
    com_defect = 0.0
    return inertia, fit_err, com_defect


# ---- projected frequencies ---------------------------------------------------------------------

def projected_frequencies(hess_cart, coords_A, masses, tangent=None):
    """Harmonic frequencies (cm^-1) after projecting translations, rotations and (optionally)
    the path tangent out of the mass-weighted Cartesian Hessian (Eh/bohr^2)."""
    n = len(masses)
    sm = np.repeat(np.sqrt(masses), 3)
    Hm = hess_cart / np.outer(sm, sm)
    X = np.asarray(coords_A, float) / BOHR
    X = X - (masses[:, None] * X).sum(0) / masses.sum()
    vecs = []
    for c in range(3):
        v = np.zeros((n, 3))
        v[:, c] = 1
        vecs.append((v * np.sqrt(masses)[:, None]).ravel())
    for c in range(3):
        e = np.zeros(3)
        e[c] = 1
        v = np.cross(e, X) * np.sqrt(masses)[:, None]
        vecs.append(v.ravel())
    if tangent is not None:
        vecs.append((np.asarray(tangent) * np.sqrt(masses)[:, None]).ravel())
    Q, _ = np.linalg.qr(np.array(vecs).T)
    s = np.linalg.svd(np.array(vecs).T, compute_uv=False)
    Q = Q[:, :int(np.sum(s > 1e-8 * s[0]))]
    P = np.eye(3 * n) - Q @ Q.T
    w = np.linalg.eigvalsh(P @ Hm @ P)
    w = np.sort(w)[Q.shape[1]:]
    return np.sign(w) * np.sqrt(np.abs(w)) * FREQ_AU


# ---- symmetry-adapted expansion and BuildSpec ------------------------------------------------

def _allowed_functions(ctx, op_text, kmax2, coord="t"):
    """Which cos/sin(k/2 g) (k = 0..kmax2) times op_text are invariant (compiler's own test)."""
    from ..build import Parser
    parser = Parser(ctx)
    out = []
    for k in range(0, kmax2 + 1):
        for fn in ("cos", "sin"):
            if k == 0 and fn == "sin":
                continue
            ftxt = "" if k == 0 else f"{fn}{k / 2:g}[{coord}]"
            if "{f}" in op_text:
                txt = op_text.replace("{f}", ftxt if ftxt else "I")
            else:
                txt = f"({op_text}) {ftxt}" if ftxt else op_text
            try:
                p = ctx.canon(parser.parse_expr(txt))
            except Exception:
                continue
            if p and ctx.invariance_defect(p) < 1e-9:
                out.append((fn, k, ftxt))
    return out


def actual_dihedrals(geoms) -> np.ndarray:
    """HOOH dihedral (radians, folded into [0, pi]) of each geometry."""
    out = []
    for geo in geoms:
        _, gg, _, _ = h2o2_body_frame(geo)
        out.append(2 * math.pi - gg if gg > math.pi else gg)
    return np.array(out)


def build_h2o2_spec(gammas_deg, geoms, V_cm, zpe_cm=None, *, m_max: int = 40, J=(0, 1, 2), K: int = 24,
                    kmax2: int = 16, tol: float = 1e-4, include_rotation: bool = True,
                    include_pseudo: bool = True):
    """Assemble a BuildSpec (+ parameter values) for H2O2 torsion-rotation from relaxed
    geometries (scan labels gammas_deg are not used as angles: the actual dihedral of each
    geometry is), energies V_cm (cm^-1, any offset) and optional adiabatic ZPE (cm^-1)."""
    from ..build import BuildSpec, Context, RotSpace, TorsionSpace, make_group, runtime_spaces
    body, gam = [], []
    for geo in geoms:
        r, gg, _, masses = h2o2_body_frame(geo)
        if gg > math.pi:                     # (x, -y, z) then R_z(pi): the same molecule on the [0, pi] branch
            r = r @ np.diag([-1.0, 1, 1])
            gg = 2 * math.pi - gg
        body.append(r)
        gam.append(gg)
    gam = np.array(gam)
    g4, r4 = extend_h2o2_path(np.array(gam), np.array(body))
    inertia, fit_err, _ = kinetic_tensor(g4, r4, masses, K=K)
    grid = np.linspace(0, 4 * np.pi, 720, endpoint=False)
    I = inertia(grid)
    G = np.linalg.inv(I)
    detI = np.linalg.det(I)
    lnd = HalfFourier(grid, np.log(detI), 2 * K)
    L = lnd.deriv(grid)
    Gtt_f = HalfFourier(grid, G[:, 3, 3], 2 * K)
    GL = HalfFourier(grid, G[:, 3, 3] * L, 2 * K)
    vpseudo = HBAR2 * (-GL.deriv(grid) / 8 + G[:, 3, 3] * L**2 / 32) if include_pseudo else np.zeros_like(grid)
    # potential (2pi periodic, even) on the 4pi grid
    gd = gam                                     # actual dihedral of each relaxed geometry
    Vs = np.asarray(V_cm, float) + (np.asarray(zpe_cm, float) if zpe_cm is not None else 0.0)
    ph = np.concatenate([gd, 2 * np.pi - gd[(gd > 0) & (gd < np.pi)]])
    pv = np.concatenate([Vs, Vs[(gd > 0) & (gd < np.pi)]])
    Vfit = HalfFourier(np.concatenate([ph, ph + 2 * np.pi]), np.concatenate([pv, pv]), K)
    Vtot = Vfit(grid) + vpseudo

    decl = BuildSpec([RotSpace(), TorsionSpace("t", m_max, period_pi=4)], "G8(H2O2)", [], J=list(J))
    group = make_group(decl)
    ctx = Context(group, runtime_spaces(decl, group))
    terms, theta = [], {}

    def expand(name, op_text, values, scale):
        allowed = _allowed_functions(ctx, op_text, kmax2)
        if not allowed:
            return 0.0
        cols = []
        for fn, k, _ in allowed:
            cols.append(np.ones_like(grid) if k == 0 else (np.cos if fn == "cos" else np.sin)(k * grid / 2))
        A = np.stack(cols, -1)
        c, *_ = np.linalg.lstsq(A, values, rcond=None)
        resid = float(np.max(np.abs(A @ c - values))) / max(np.max(np.abs(values)), 1e-300)
        for (fn, k, ftxt), ck in zip(allowed, c):
            if abs(ck) * scale < tol:
                continue
            pname = f"{name}_{fn[0]}{k}"
            txt = op_text.replace("{f}", ftxt if ftxt else "I") if "{f}" in op_text else \
                (f"({op_text}) {ftxt}" if ftxt else op_text)
            terms.append(f"{pname}: {txt}")
            theta[pname] = float(ck * scale)
        return resid

    resid = {}
    half = HBAR2 / 2
    resid["V"] = expand("V", "{f}", Vtot, 1.0)
    resid["Gtt"] = expand("Gtt", "p[t] {f} p[t]", G[:, 3, 3], half)
    if include_rotation:
        ax = "xyz"
        for a in range(3):
            resid[f"G{ax[a]}{ax[a]}"] = expand(f"G{ax[a]}{ax[a]}", f"J{ax[a]} J{ax[a]}", G[:, a, a], half)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            resid[f"G{ax[a]}{ax[b]}"] = expand(f"G{ax[a]}{ax[b]}", f"J{ax[a]} J{ax[b]} + J{ax[b]} J{ax[a]}",
                                               G[:, a, b], half)
        for a in range(3):
            resid[f"G{ax[a]}t"] = expand(f"G{ax[a]}t", f"J{ax[a]} (p[t] {{f}} + {{f}} p[t])", G[:, a, 3], half)
    spec = BuildSpec(decl.spaces, decl.group, terms, list(J), max_order=6, name="h2o2_rph")
    info = {"geometry_fit_max_error_A": fit_err, "expansion_rel_residual": resid,
            "G_tt_cm": {"cis": float(half * Gtt_f(0.0)), "trans": float(half * Gtt_f(np.pi))},
            "pseudopotential_cm": {"min": float(vpseudo.min()), "max": float(vpseudo.max())},
            "symmetry_violation_G": _symmetry_violation(grid, G),
            "actual_dihedral_deg": np.degrees(gam).tolist(),
            "inertia_min_eigenvalue": float(np.min(np.linalg.eigvalsh(I)))}
    return spec, theta, info


def _symmetry_violation(grid, G):
    """G-matrix elements that symmetry forbids at gamma = 0, pi (cis/trans C2v points): |G_xy|, |G_xt|..."""
    i0, ip = 0, len(grid) // 4
    return {"G_xy_cis": float(abs(G[i0, 0, 1])), "G_yt_cis": float(abs(G[i0, 1, 3])),
            "G_xy_trans": float(abs(G[ip, 0, 1])), "G_zt_trans": float(abs(G[ip, 2, 3]))}
