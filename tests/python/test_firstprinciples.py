"""First-principles tunneling and rovibronic levels (riss_drucker.abinitio: rpsurface, tunneling2d, firstprinciples)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DZ = ROOT / "data" / "abinitio" / "ch4plus_ccpvdz"


def test_plane_solver_separable_and_rph_limits():
    from riss_drucker.abinitio.tunneling import splitting
    from riss_drucker.abinitio.tunneling2d import (mirrored, solve_2d, plane_splitting_converged,
                                                   splitting_variable_mass)
    smin = 0.8
    U = lambda s: 1000 * (1 - (s / smin) ** 2) ** 2
    Um = mirrored(U, smin)
    lam = 5000.0
    x = np.linspace(0, smin, 80)
    ref = splitting(x, U(x), n_grid=800)["pair_splitting"]
    # Cartesian plane: separable double well x harmonic oscillator -> the 1D splitting
    r = plane_splitting_converged(lambda P: Um(np.clip(P[:, 0], -2 * smin, 2 * smin)) + 0.5 * lam * P[:, 1] ** 2,
                                  (-1.6, -0.6), (1.6, 0.6), hs=(0.016, 0.011))
    assert abs(r["pair_splitting_extrapolated"] - ref) < 2e-3 * ref
    # curvilinear 2D with kappa = 0 (Dirichlet) -> the 1D splitting and the transverse zero point
    r2 = solve_2d(smin, U, lambda s: 0 * s, lambda s, q: 0.5 * lam * q ** 2, 1.0, ns=241, nq=81)
    assert abs(r2["pair_splitting"] - ref) < 2e-3 * ref
    # variable-mass solver with unit mass -> the 1D splitting
    assert abs(splitting_variable_mass(x, U(x), np.ones_like(x))["pair_splitting"] - ref) < 2e-3 * ref


def test_rotation_operator_convention():
    """D(R) J_mf D(R)^+ = sum_b R_ba J_mf,b: the frame rotation uses the builder's own convention."""
    from scipy.spatial.transform import Rotation
    from riss_drucker.build import _C2Y
    from riss_drucker.symmetry import angular_momentum_matrices, su2_lift, wigner_D
    R = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    for tj in (2, 5, 8):
        jx, jy, jz = angular_momentum_matrices(tj)
        J = [jx, -jy, jz]
        D = wigner_D(tj, _C2Y @ su2_lift(R) @ _C2Y.conj().T)
        err = max(np.abs(D @ J[a] @ D.conj().T - sum(R[b, a] * J[b] for b in range(3))).max() for a in range(3))
        assert err < 1e-12


def test_reaction_plane_geometry_and_frame_rotation():
    from riss_drucker.abinitio.rpsurface import ReactionPlane, ReactionSurface, load_stages
    st = load_stages(DZ)
    for iso in ("H", "D"):
        rs = ReactionSurface(st, iso, use_grid=False)
        pl = ReactionPlane(rs)
        assert pl.variance_in_plane > 0.98
        # the plane potential reproduces the folded 1D profile along the (extended) path
        assert np.abs(pl.harmonic_potential(pl.curve_ext) - rs.V_path(pl.fold(pl.s_ext))).max() < 1e-6
        # reflection: path points stay, mirrored points land on the path
        Q = pl.reflect_into_path(pl.curve_ext)
        n = np.arange(len(Q))
        on = (n >= pl.n_ext) & (n < pl.n_ext + len(pl.curve))
        assert np.abs(Q[on] - pl.curve_ext[on]).max() == 0
        assert np.sqrt(((Q[~on][:, None] - pl.curve[None]) ** 2).sum(-1)).min(1).max() < 1e-9
        fr = rs.frame_rotation()
        assert fr["saddle_symmetry_rms_A"] < 1e-4                          # geometry precision
        assert abs(abs(np.dot(fr["well_vector_L"], fr["well_vector_R"])) - 0.5) < 1e-3   # neighbouring wells
        assert 9.5 < fr["angle_deg"] < 11.5


def test_firstprinciples_levels_dz():
    """cc-pVDZ ab initio plane: delta and the rovibronic levels (symmetry blocks exact, Ham's theorem)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import case_methane_cation_rovibronic as R
    from riss_drucker.abinitio import firstprinciples as FP
    from riss_drucker.abinitio.rpsurface import load_stages
    st = load_stages(DZ)
    if "plane2d" not in st:
        pytest.skip("no reaction-plane stage")
    ta = FP.tunneling_analysis(st, "H", hs=(0.014, 0.010))
    assert abs(ta["delta"] - 13.38) < 0.25                     # converged value 13.375 (h -> 0)
    spec = R.Model(5).spec.to_dict()
    lv = FP.rovibronic_levels({k: spec[k] for k in ("group", "spaces", "terms")}, range(3), ta)
    assert lv["symmetry_leakage"] < 1e-10 and lv["ground_irrep_N0"] == "F2"
    n0 = sorted(v for (N, _, _), v in lv["levels"].items() if N == 0)
    assert abs(n0[-1] - n0[0] - ta["delta"]) < 1e-6           # N+ = 0: F2 and F1 split by delta = 4 |sigma|
