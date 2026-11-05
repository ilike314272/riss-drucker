"""Reaction-surface data for tunneling between equivalent minima (pure NumPy/SciPy).

From the cached ab initio stages (minimum, saddle, half reaction path saddle -> minimum, Hessians
along it, coupled-cluster single points) this builds, for a chosen isotopologue:

* s: mass-weighted arc length (amu^1/2 Angstrom) on the chain aligned to zero angular momentum
  (consecutive Kabsch alignment; for infinitesimal steps this is the Eckart/Sayvetz condition);
* the path curvature vector kappa(s) (orthogonal to the tangent) and its unit vector u(s): the
  effective transverse mode of the Miller-Handy-Adams reaction-path Hamiltonian, along which
  corner cutting happens;
* V(s): MP2 profile corrected by [CCSD(T) - MP2] interpolated along s;
* lambda(s): harmonic force constant along u (projected Hessians), and ZPE_rest(s): harmonic
  zero-point energy of the remaining modes (tangent, u, rotations, translations projected out);
* optionally an ab initio grid E(s, q) along u (stage 'surf2d') replacing the harmonic lambda;
* geometries X(s, q) = X(s) + q u(s)/sqrt(m) and inverse inertia tensors (vibrational averaging);
* the frame rotation accumulated along the full path L -> saddle -> R in the zero-angular-
  momentum frame, relative to the frames of a common tetrahedral reference (the frame in which
  the effective Hamiltonian's Td operations act): the first-principles source of
  rotation-tunneling coupling.

Units: energies cm^-1, s and q in amu^1/2 Angstrom, lambda in cm^-1 / (amu Angstrom^2) with
V = lambda q^2 / 2 and hbar^2/(2 amu Angstrom^2) = 16.8576292 cm^-1.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

H2CM = 219474.6313705
BOHR = 0.529177210903
FREQ_AU = 5140.48714            # sqrt(Eh / (bohr^2 amu)) / (2 pi c) in cm^-1
HBAR2_HALF = 16.8576292         # hbar^2 / (2 amu Angstrom^2) in cm^-1
MASS = {"H": 1.00782503207, "D": 2.01410177812, "C": 12.0, "C13": 13.00335483507}


def load_stages(d) -> dict:
    d = Path(d)
    out = {}
    for name in ("min", "ts", "irc", "hess", "ccsdt", "neutral", "surf2d", "plane2d"):
        f = d / f"ch4plus_{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def kabsch(ref, X, masses, allow_reflection=False):
    w = masses / masses.sum()
    cr, cx = (w[:, None] * ref).sum(0), (w[:, None] * X).sum(0)
    A, B = ref - cr, X - cx
    H = (B * w[:, None]).T @ A
    U, _, Vt = np.linalg.svd(H)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return B @ R.T + cr, R


def inverse_inertia(X, masses):
    """I^-1 (amu^-1 Angstrom^-2) about the centre of mass."""
    c = X - (masses[:, None] * X).sum(0) / masses.sum()
    I = np.zeros((3, 3))
    for m, r in zip(masses, c):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return np.linalg.inv(I)


def lam_cm(lam_au):
    """Mass-weighted Hessian eigenvalue Eh/(bohr^2 amu) -> cm^-1/(amu Angstrom^2) for V = lam q^2/2."""
    return lam_au * H2CM / BOHR ** 2


def omega_cm(lam):
    """Harmonic wave number (cm^-1) of V = lam q^2/2 with T = HBAR2_HALF p^2 (mass-weighted)."""
    return np.sign(lam) * np.sqrt(2 * HBAR2_HALF * np.abs(lam))


def _projector(vectors, n):
    V = np.array(vectors).T
    sv = np.linalg.svd(V, compute_uv=False)
    Q, _ = np.linalg.qr(V)
    Q = Q[:, :int(np.sum(sv > 1e-8 * sv[0]))]
    return np.eye(n) - Q @ Q.T, Q


def _rt_vectors(X, masses):
    """Mass-weighted translation and rotation vectors (bohr units irrelevant: directions only)."""
    n = len(masses)
    Xc = X - (masses[:, None] * X).sum(0) / masses.sum()
    out = []
    for c in range(3):
        v = np.zeros((n, 3))
        v[:, c] = 1
        out.append((v * np.sqrt(masses)[:, None]).ravel())
    for c in range(3):
        e = np.zeros(3)
        e[c] = 1
        out.append((np.cross(e, Xc) * np.sqrt(masses)[:, None]).ravel())
    return out


class ReactionSurface:
    def __init__(self, stages: dict, iso: str = "H", use_grid: bool = True):
        from scipy.interpolate import CubicSpline
        self.stages, self.iso = stages, iso
        self.m = np.array([MASS["C"]] + [MASS[iso]] * 4)
        m = self.m
        P = [np.array(p, float) for p in stages["irc"]["path"]]
        for i in range(1, len(P)):
            P[i], _ = kabsch(P[i - 1], P[i], m)
        # common origin: centre of mass at 0
        P = [p - (m[:, None] * p).sum(0) / m.sum() for p in P]
        self.P = P
        s = [0.0]
        for a, b in zip(P[:-1], P[1:]):
            s.append(s[-1] + float(np.sqrt(np.sum(m[:, None] * (b - a) ** 2))))
        self.s = np.array(s)
        self.smin = self.s[-1]
        Y = np.array([(np.sqrt(m)[:, None] * p).ravel() for p in P])
        self._Y = CubicSpline(self.s, Y, axis=0)
        # energies
        E = np.array(stages["irc"]["e_ump2"]) * H2CM
        cc = stages.get("ccsdt")
        if cc is not None:
            ci = cc["indices"]
            corr = np.array(cc["e_ccsdt"]) * H2CM - E[ci]
            E = E + np.interp(self.s, self.s[ci], corr)
        self.V = E - E[-1]
        Em = np.array(stages["irc"]["e_ump2"]) * H2CM
        self.e_mp2_min = float(Em[-1])
        self._vmp2 = Em - Em[-1]
        # curvature, effective mode, force constants and ZPE at the Hessian points
        hs = stages["hess"]
        self.h_idx = list(hs["indices"])
        lam, zpe = [], []
        for k, H in zip(self.h_idx, hs["hessians"]):
            t, u, kap = self.frame(self.s[k])
            sm = np.repeat(np.sqrt(m), 3)
            Hm = np.array(H) / np.outer(sm, sm)                      # Eh / (bohr^2 amu)
            rt = _rt_vectors(P[k] / BOHR, m)
            Pr, _ = _projector(rt + [t], len(sm))
            Hp = Pr @ Hm @ Pr
            lam.append(lam_cm(float(u @ Hp @ u)))
            Pr2, Q2 = _projector(rt + [t, u], len(sm))
            w, V = np.linalg.eigh(Pr2 @ Hm @ Pr2)
            inside = np.linalg.norm(Q2.T @ V, axis=0)
            keep = np.argsort(inside)[:len(sm) - Q2.shape[1]]
            ww = w[keep]
            zpe.append(0.5 * float(np.sum(np.sqrt(np.clip(ww, 0, None)) * FREQ_AU)))
        hsv = self.s[self.h_idx]
        self._lam = (hsv, np.array(lam))
        self._zpe = (hsv, np.array(zpe) - zpe[-1])
        self.zpe_rest_min = zpe[-1]
        self.grid = stages.get("surf2d") if use_grid else None
        if self.grid is not None and self.grid.get("iso", "H") != "H" and iso == "H":
            self.grid = None

    # ---------------------------------------------------------------- path frame
    def frame(self, s):
        """(tangent, curvature unit vector u, curvature kappa) in mass-weighted Cartesians."""
        s = float(np.clip(s, 0, self.smin))
        t = self._Y(s, 1)
        t = t / np.linalg.norm(t)
        c = self._Y(s, 2)
        c = c - (c @ t) * t
        k = float(np.linalg.norm(c))
        u = c / k if k > 1e-12 else np.zeros_like(c)
        return t, u, k

    def kappa(self, s):
        return np.array([self.frame(x)[2] for x in np.atleast_1d(s)])

    def lam(self, s):
        return np.interp(np.clip(s, 0, self.smin), *self._lam)

    def zpe_rest(self, s):
        return np.interp(np.clip(s, 0, self.smin), *self._zpe)

    def _v_mp2(self, s):
        return np.interp(np.clip(s, 0, self.smin), self.s, self._vmp2)

    def V_path(self, s):
        return np.interp(np.clip(s, 0, self.smin), self.s, self.V)

    def Vq(self, s, q):
        """Transverse potential along u at path point s (cm^-1): ab initio grid if present, else harmonic."""
        s = np.clip(s, 0, self.smin)
        if self.grid is None:
            return 0.5 * self.lam(s) * q ** 2
        from scipy.interpolate import RectBivariateSpline
        if not hasattr(self, "_gspl"):
            g = self.grid
            gs = self.s[g["indices"]]
            qq = np.array(g["q"])
            E = np.array(g["e_ump2"]) * H2CM
            E = E - E[:, [int(np.argmin(np.abs(qq)))]]
            self._gspl = RectBivariateSpline(gs, qq, E, kx=min(3, len(gs) - 1), ky=min(3, len(qq) - 1))
            self._gq = (qq.min(), qq.max())
        qc = np.clip(q, *self._gq)
        harm = 0.5 * self.lam(s) * (q ** 2 - qc ** 2)          # harmonic continuation outside the grid
        return self._gspl(s, qc, grid=False) + harm

    def geometry(self, s, q=0.0):
        s_ = float(np.clip(s, 0, self.smin))
        y = self._Y(s_) + q * self.frame(s_)[1]
        return (y.reshape(-1, 3) / np.sqrt(self.m)[:, None])

    def grid_geometries(self, s_idx, qs, iso_masses=None):
        """Cartesian geometries for the ab initio grid stage (H masses define the mode)."""
        return [[self.geometry(self.s[k], q).tolist() for q in qs] for k in s_idx]

    # ---------------------------------------------------------------- frames and wells
    def td_alignment(self, X, perm=None):
        """Proper rotation aligning X to the tetrahedral reference frame. The H labelling is searched
        unless `perm` is given (labels must be carried along a path, not re-chosen)."""
        r = 1.09 / np.sqrt(3)
        Td = np.array([[0, 0, 0], [r, r, r], [r, -r, -r], [-r, r, -r], [-r, -r, r]])
        best = None
        for perm in ([tuple(perm)] if perm is not None else itertools.permutations(range(4))):
            Xp = X[[0] + [1 + p for p in perm]]
            A, R = kabsch(Td, Xp, self.m)
            rms = float(np.sqrt(np.mean(np.sum((A - Td) ** 2, 1))))
            if best is None or rms < best[0]:
                best = (rms, perm, A, R)
        return {"rms": best[0], "perm": best[1], "aligned": best[2], "R": best[3]}

    @staticmethod
    def well_vector(Xtd):
        """Unit vector along the H-H direction of the long-bond pair (the well's T2 hole function)."""
        H = Xtd[1:] - Xtd[0]
        d = np.linalg.norm(H, axis=1)
        a, b = np.argsort(d)[-2:]
        v = H[a] - H[b]
        return v / np.linalg.norm(v)

    def frame_rotation(self):
        """Rotation accumulated along L-min -> saddle -> R-min in the zero-angular-momentum frame,
        relative to the tetrahedral reference frames of both minima. Returns the rotation R_theta
        (3x3, in the reference frame), its angle and axis, and the well vectors of L and R."""
        nb = self.stages["ts"]["neighbour"]
        perm = [p - 1 for p in nb["permutation"]]
        refl = nb.get("reflection", 1)

        def P_op(X):
            Y = X.copy()
            Y[1:] = X[1:][perm]
            if refl == -1:
                Y = Y * np.array([1, 1, -1])
            return Y
        half = [p.copy() for p in self.P]                       # saddle -> L minimum
        # the neighbour operation must map the saddle onto itself (labels kept), else the halves don't join
        Sa = half[0]
        Sb, _ = kabsch(Sa, P_op(Sa), self.m)
        saddle_rms = float(np.sqrt(np.mean(np.sum((Sb - Sa) ** 2, 1))))
        if saddle_rms > 0.05:
            raise ValueError(f"neighbour operation does not map the saddle onto itself (rms {saddle_rms:.3f} A)")
        chain = half[::-1] + [P_op(p) for p in half[1:]]         # L min -> saddle -> R min, labels kept
        for i in range(1, len(chain)):
            chain[i], _ = kabsch(chain[i - 1], chain[i], self.m)
        L, R_end = chain[0], chain[-1]
        aL = self.td_alignment(L)
        # put the whole chain in L's reference frame
        cm = (self.m[:, None] * L).sum(0) / self.m.sum()
        chain_td = [(x - (self.m[:, None] * x).sum(0) / self.m.sum()) @ aL["R"].T for x in chain]
        aR = self.td_alignment(chain_td[-1], perm=aL["perm"])
        Rth = aR["R"]                         # maps the transported R frame onto the reference frame
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rth) - 1) / 2, -1, 1))))
        w, V = np.linalg.eig(Rth)
        axis = np.real(V[:, int(np.argmin(np.abs(w - 1)))])
        vL = self.well_vector(aL["aligned"])
        vR = self.well_vector(aR["aligned"])
        return {"R_theta": Rth.tolist(), "angle_deg": ang, "axis": (axis / np.linalg.norm(axis)).tolist(),
                "well_vector_L": vL.tolist(), "well_vector_R": vR.tolist(),
                "td_rms_L": aL["rms"], "td_rms_R": aR["rms"], "R_L": aL["R"].tolist(), "perm_L": list(aL["perm"]),
                "saddle_symmetry_rms_A": saddle_rms}


class ReactionPlane:
    """Cartesian reaction plane for the double well L-min -> saddle -> R-min.

    The full path (half path plus its image under the neighbour operation, labels kept) is put in
    one frame (aligned to the saddle) in mass-weighted Cartesians. Its two principal components
    span the plane (x, y): there the kinetic energy is exactly -HBAR2_HALF (d_x^2 + d_y^2), with no
    curvilinear fold. The remaining coordinates follow the path: a plane point carries the path's
    out-of-plane relaxation, kernel-averaged over nearby path points (smooth across the inner side
    of the bend). Path-attached quantities (ZPE of the other modes, the CCSD(T) - MP2 correction)
    are kernel-averaged the same way.

    Potential on the plane: E_MP2(x, y) from the 'plane2d' stage where computed (spline), else the
    harmonic valley V(s*) + lambda(s*) d^2 / 2 around the path (d: distance to the path in the plane).
    """

    def __init__(self, rs: "ReactionSurface", n_path: int = 161, kernel: float = 0.04):
        self.rs, self.kernel = rs, kernel
        st, m = rs.stages, rs.m
        nb = st["ts"]["neighbour"]
        perm = [p - 1 for p in nb["permutation"]]
        refl = nb.get("reflection", 1)

        def P_op(X):
            Y = X.copy()
            Y[1:] = X[1:][perm]
            return Y * np.array([1, 1, -1]) if refl == -1 else Y
        ss = np.linspace(0, rs.smin, n_path)
        half = [rs.geometry(x) for x in ss]
        chain = half[::-1] + [P_op(p) for p in half[1:]]          # L ... saddle ... R
        self.s_signed = np.r_[-ss[::-1], ss[1:]]
        sad = half[0]
        al = [kabsch(sad, x, m)[0] for x in chain]
        al = [x - (m[:, None] * x).sum(0) / m.sum() for x in al]
        self.chain = al
        Y = np.array([(np.sqrt(m)[:, None] * x).ravel() for x in al])
        self.c0 = Y.mean(0)
        _, sv, Vt = np.linalg.svd(Y - self.c0, full_matrices=False)
        self.E = Vt[:2]                                              # plane basis (2 x 15)
        self.variance_in_plane = float(np.sum(sv[:2] ** 2) / np.sum(sv ** 2))
        self.curve = (Y - self.c0) @ self.E.T                        # path in the plane
        self.resid = (Y - self.c0) - self.curve @ self.E              # out-of-plane relaxation along the path
        # extension beyond the minima: point reflection about each minimum (as the 1D mirror)
        k = n_path - 1
        c, sg = self.curve, self.s_signed
        left = (2 * c[0] - c[1:k])[::-1]                 # point reflection about the L minimum
        left_s = (2 * sg[0] - sg[1:k])[::-1]
        right = (2 * c[-1] - c[-k:-1])[::-1]             # ... and about the R minimum
        right_s = (2 * sg[-1] - sg[-k:-1])[::-1]
        self.n_ext = len(left)                       # points on each mirrored extension
        self.curve_ext = np.r_[left, c, right]
        self.s_ext = np.r_[left_s, sg, right_s]
        assert np.all(np.diff(self.s_ext) > 0)

    def fold(self, s):
        x = np.abs(s)
        return np.where(x <= self.rs.smin, x, 2 * self.rs.smin - x)

    def _nearest(self, P):
        d2 = ((P[:, None, :] - self.curve_ext[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, 1)
        return j, np.sqrt(d2[np.arange(len(P)), j]), d2

    def path_average(self, P, values_of_s):
        """Kernel average over the (extended) path of a function of |s| (folded)."""
        _, _, d2 = self._nearest(P)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * self.kernel ** 2))
        f = values_of_s(self.fold(self.s_ext))
        return (w * f[None, :]).sum(1) / w.sum(1)

    def harmonic_potential(self, P):
        rs = self.rs
        j, d, _ = self._nearest(P)
        sf = self.fold(self.s_ext[j])
        return rs.V_path(sf) + 0.5 * rs.lam(sf) * d ** 2

    def zpe_rest(self, P):
        return self.path_average(P, self.rs.zpe_rest)

    def geometry(self, p):
        """Cartesian geometry (Angstrom) of plane point p = (x, y), with the path's out-of-plane
        relaxation (kernel-averaged over the unextended path)."""
        P = np.atleast_2d(p)
        d2 = ((P[:, None, :] - self.curve[None, :, :]) ** 2).sum(-1)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * self.kernel ** 2))
        R = (w @ self.resid) / w.sum(1, keepdims=True)
        Y = self.c0 + P @ self.E + R
        out = [(y.reshape(-1, 3) / np.sqrt(self.rs.m)[:, None]) for y in Y]
        return out[0] if np.ndim(p) == 1 else out

    def bounds(self, pad=0.25):
        lo = self.curve_ext.min(0) - pad
        hi = self.curve_ext.max(0) + pad
        return lo, hi

    def transverse_width(self):
        """Largest zero-point rms width (amu^1/2 Angstrom) of the transverse mode along the path."""
        ss = np.linspace(0, self.rs.smin, 200)
        w = np.abs(omega_cm(self.rs.lam(ss)))
        return float(np.sqrt(HBAR2_HALF / w.min()))

    def abinitio_points(self, h=0.03, nsig=3.5, ecut=8000.0):
        """Plane points for the ab initio grid (spacing h): within nsig zero-point widths of the actual
        path (the mirrored extensions beyond the minima are reflected into this region when the
        potential is evaluated) and below ecut (cm^-1) in the harmonic estimate. Returns (points, axes)."""
        dmax = nsig * self.transverse_width()
        lo = self.curve.min(0) - dmax
        hi = self.curve.max(0) + dmax
        gx = np.arange(lo[0], hi[0] + h / 2, h)
        gy = np.arange(lo[1], hi[1] + h / 2, h)
        G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        d = np.sqrt(((G[:, None, :] - self.curve[None, :, :]) ** 2).sum(-1)).min(1)
        keep = (d < dmax) & (self.harmonic_potential(G) < ecut)
        return G[keep], (gx, gy)

    def reflect_into_path(self, P):
        """Points whose nearest path point lies on a mirrored extension are reflected about that
        minimum (the same point reflection that defines the extension), so every point is evaluated
        in the region spanned by the actual path."""
        P = np.array(P, float)
        k = self.n_ext
        j, _, _ = self._nearest(P)
        left, right = j < k, j >= k + len(self.curve)
        P[left] = 2 * self.curve[0] - P[left]
        P[right] = 2 * self.curve[-1] - P[right]
        return P

    def potential(self, P, grid=None):
        """V(x, y) (cm^-1, relative to the path minimum) + ZPE of the remaining modes.

        Without a grid: the harmonic valley V(s*) + lambda(s*) d^2/2 around the path. With the ab initio
        plane (stage 'plane2d'): E_MP2 - E_MP2(min) + [CCSD(T) - MP2](s*), interpolated exactly by a
        thin-plate spline inside the computed region; beyond it the surface continues from the nearest
        computed value with a harmonic wall lambda d_out^2/2 (continuous at the boundary, independent of
        the harmonic valley model). Points on the mirrored extensions are first reflected into the
        region spanned by the actual path."""
        P = np.atleast_2d(P)
        Pr = self.reflect_into_path(P)
        if grid is None:
            V = self.harmonic_potential(Pr)
        else:
            V = self._abinitio(grid)(Pr)
        return V + self.zpe_rest(Pr)

    def _abinitio(self, grid):
        key = id(grid)
        if getattr(self, "_ai_key", None) == key:
            return self._ai
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial import cKDTree
        pts = np.array(grid["points"])
        h = float(grid.get("h", 0.03))
        E = np.array(grid["e_ump2"]) * H2CM - self.rs.e_mp2_min
        E = E + self.path_average(pts, lambda s: self.rs.V_path(s) - self.rs._v_mp2(s))
        rbf = RBFInterpolator(pts, E, kernel="thin_plate_spline", neighbors=60, smoothing=0.0)
        tree = cKDTree(pts)
        lam_wall = float(np.max(self.rs.lam(np.linspace(0, self.rs.smin, 50))))

        def V(P):
            d, j = tree.query(P)
            inside = d <= 0.75 * h
            out = E[j] + 0.5 * lam_wall * np.clip(d - 0.75 * h, 0, None) ** 2
            if np.any(inside):
                out[inside] = rbf(P[inside])
            return out
        self._ai_key, self._ai = key, V
        return V

    def coverage(self, P, grid):
        """Fraction weights: 1 where a point (after reflection) is within one spacing of computed points."""
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.array(grid["points"])).query(self.reflect_into_path(P))
        return d < float(grid.get("h", 0.04))
