"""First-principles rovibronic levels of a multi-well tunneling system (CH4+ ground vibronic state).

Everything here comes from the ab initio stages; no measured line enters.

1. Tunneling: the pair splitting 2|sigma| of neighbouring wells from the Cartesian reaction plane
   (variational, converged in the grid, h^2-extrapolated), with the ab initio plane grid when it
   exists and the harmonic valley otherwise. For context the 1D path, the curvilinear 2D reaction
   path Hamiltonian (lower bound: its coordinates fold at 1/kappa) and the small-curvature
   effective-mass estimate are reported as well.
2. Vibrationally averaged rotor: <I^-1> over the localized tunneling ground state (L = (psi0 +
   psi1)/sqrt2) on the plane, giving the full rotational tensor of one well in the tetrahedral
   reference frame (Q_ab J_a J_b with Q = HBAR2_HALF <I^-1>).
3. Frame rotation: the rotation R_theta accumulated along L -> saddle -> R in the zero-angular-
   momentum frame relative to the reference frames of the two minima.
4. Rovibronic Hamiltonian on rotation x wells (any well-vector set from a build spec):
       H = n_w/|G| sum_g U_g (|L><L| x Q.JJ) U_g^+  +  s sigma / c0 * 1/|G| sum_g U_g (|R><L| x D(R_theta) + h.c.) U_g^+
   with D(R) = wigner_D(C2y u(R) C2y^+), the builder's own rotation operator (so the group action
   and the frame rotation share one convention), c0 the N = 0 normalization (nearest-well element
   = sigma), and s = +-1 chosen so the N = 0 ground level is F2 (Ham's theorem: the vibronic
   ground state of a strongly Jahn-Teller-distorted T2 state has the electronic symmetry). The
   matrix is projected on the compiled symmetry blocks; the off-block remainder is a certificate.
"""
from __future__ import annotations

import numpy as np

from .rpsurface import HBAR2_HALF, ReactionPlane, ReactionSurface, inverse_inertia, load_stages, omega_cm


def tunneling_analysis(stages: dict, iso: str, *, hs=(0.014, 0.010, 0.007), pad=0.25, use_grid=True,
                       modify=None) -> dict:
    """modify(rs): optional change of the ReactionSurface before anything is computed (uncertainty
    budgets: scale the CCSD(T) correction or the ZPE of the remaining modes, ...)."""
    from .tunneling import splitting
    from .tunneling2d import (plane_splitting_converged, sct_effective_mass, solve_2d, solve_plane,
                              splitting_variable_mass)
    rs = ReactionSurface(stages, iso, use_grid=False)
    if modify is not None:
        modify(rs)
    pl = ReactionPlane(rs)
    grid = None
    if use_grid and "plane2d" in stages and iso in stages["plane2d"]:
        grid = stages["plane2d"][iso]
    pot = lambda P: pl.potential(P, grid)
    lo, hi = pl.bounds(pad)
    conv = plane_splitting_converged(pot, lo, hi, hs)
    last = conv["last"]
    ss = np.linspace(0, rs.smin, 240)
    w_u = omega_cm(rs.lam(ss))
    U1 = rs.V_path(ss) + rs.zpe_rest(ss) + 0.5 * (w_u - w_u[-1])
    one_d = splitting(ss, U1)["pair_splitting"]
    kmax = rs.kappa(ss).max()
    rph = solve_2d(rs.smin, lambda s: rs.V_path(s) + rs.zpe_rest(s), lambda s: rs.kappa(s), rs.Vq,
                   0.9 / kmax)["pair_splitting"]
    sct = splitting_variable_mass(ss, U1, sct_effective_mass(ss, rs.kappa(ss), w_u))["pair_splitting"]
    # localized ground state and vibrational averaging of I^-1 (in the saddle-aligned plane frame)
    psi0, psi1 = last["psi"][0].ravel(), last["psi"][1].ravel()
    P = last["points"]
    dL = np.sum((P - pl.curve[0]) ** 2, 1)
    near_L = dL < np.sum((P - pl.curve[-1]) ** 2, 1)
    psiL = psi0 + psi1 if np.sum((psi0 + psi1)[near_L] ** 2) > np.sum((psi0 - psi1)[near_L] ** 2) else psi0 - psi1
    rho = psiL ** 2
    rho = rho / rho.sum()
    sel = rho > 1e-7 * rho.max()
    Iinv = np.zeros((3, 3))
    for p, r in zip(P[sel], rho[sel] / rho[sel].sum()):
        Iinv += r * inverse_inertia(pl.geometry(p), rs.m)
    fr = rs.frame_rotation()
    # one alignment for everything: the tetrahedral fit has six equally good solutions (one per well),
    # so the rotor tensor must use the labelling the frame rotation used
    a = rs.td_alignment(pl.chain[0], perm=fr["perm_L"])
    R = a["R"]
    vQ = rs.well_vector(a["aligned"])
    if abs(abs(vQ @ np.array(fr["well_vector_L"])) - 1) > 1e-3:
        raise ValueError("rotor tensor and frame rotation refer to different wells")
    Q_td = HBAR2_HALF * (R @ Iinv @ R.T)
    Q_eq = HBAR2_HALF * (R @ inverse_inertia(pl.chain[0], rs.m) @ R.T)
    sigma = conv["pair_splitting_extrapolated"] / 2
    return {"iso": iso, "surface": "ab initio plane grid (MP2 + path CCSD(T) correction)" if grid else
            "harmonic valley around the path", "sigma_abs": sigma, "delta": 4 * sigma,
            "plane_convergence": conv["grids"], "plane_relative_change_last_grid": conv["relative_change_last_grid"],
            "plane_variance_fraction": pl.variance_in_plane,
            "context_pair_splittings": {"1D path": one_d, "2D reaction-path Hamiltonian (lower bound)": rph,
                                        "small-curvature effective mass": sct,
                                        "reaction plane (this result)": 2 * sigma},
            "rotor_tensor_averaged_td": Q_td.tolist(), "rotor_tensor_equilibrium_td": Q_eq.tolist(),
            "rotational_constants_averaged": sorted(np.linalg.eigvalsh(Q_td).tolist(), reverse=True),
            "rotational_constants_equilibrium": sorted(np.linalg.eigvalsh(Q_eq).tolist(), reverse=True),
            "frame_rotation": fr, "well_vector_L": fr["well_vector_L"], "well_vector_R": fr["well_vector_R"],
            "td_alignment_rms_A": a["rms"]}


def rovibronic_levels(spec_dict: dict, J: list, ta: dict, *, variant: str = "R", sign: float | None = None,
                      rotor="averaged") -> dict:
    """First-principles levels {(N, irrep, k): E} on the blocks of the given build spec (rot x wells)."""
    import math
    from .. import build as B
    from ..build import Context, _C2Y, runtime_spaces
    from ..symmetry import su2_lift, wigner_D
    d = dict(spec_dict)
    d["J"] = list(J)
    d.setdefault("max_order", 10)
    d.setdefault("name", "fp")
    spec = B.BuildSpec.from_dict(d)
    bm = B.compile_build(spec)
    g = bm.group
    ctx = Context(g, runtime_spaces(spec, g))
    rot, well = ctx.spaces[0], ctx.spaces[1]
    if rot.kind != "rot" or well.kind != "well" or len(ctx.spaces) != 2:
        raise ValueError("first-principles levels need exactly a rotation space and a well space")
    V = np.array([s.vectors for s in spec.spaces if s.kind == "well"][0], float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    def find(v):
        dots = np.abs(V @ np.asarray(v))
        k = int(np.argmax(dots))
        if abs(dots[k] - 1) > 1e-3:
            raise ValueError(f"no well vector along {np.round(v, 3)} (best overlap {dots[k]:.3f})")
        return k
    iL, iR = find(ta["well_vector_L"]), find(ta["well_vector_R"])
    Q = np.array(ta["rotor_tensor_averaged_td" if rotor == "averaged" else "rotor_tensor_equilibrium_td"])
    Rth = np.array(ta["frame_rotation"]["R_theta"])
    if variant == "Rinv":
        Rth = Rth.T
    elif variant == "none":
        Rth = np.eye(3)
    nw, order = well.n, g.order

    def full(tj, s):
        J_ = rot.mats(tj)
        Jm = [np.asarray(J_[a].toarray() if hasattr(J_[a], "toarray") else J_[a]) for a in "xyz"]
        QJ = sum(Q[a, b] * 0.5 * (Jm[a] @ Jm[b] + Jm[b] @ Jm[a]) for a in range(3) for b in range(3))
        PL = np.zeros((nw, nw))
        PL[iL, iL] = 1
        hop = np.zeros((nw, nw))
        hop[iR, iL] = 1
        Dm = wigner_D(tj, _C2Y @ su2_lift(Rth) @ _C2Y.conj().T)
        O_rot = np.kron(QJ, PL)
        O_hop = np.kron(Dm, hop)
        O_hop = O_hop + O_hop.conj().T
        Hr = np.zeros_like(O_rot, dtype=complex)
        Hh = np.zeros_like(O_rot, dtype=complex)
        for gi in range(order):
            U = np.kron(rot.U(gi, tj).toarray(), well.U(gi, tj).toarray())
            Hr += U @ O_rot @ U.conj().T
            Hh += U @ O_hop @ U.conj().T
        return Hr * nw / order, Hh / order
    _, h0 = full(0, 1.0)
    c0 = float(np.max(np.abs(h0)))
    sig = ta["sigma_abs"]
    out = {}
    for s in ([sign] if sign is not None else [1.0, -1.0]):
        levels, leak = {}, 0.0
        for tj in sorted({b.two_j for b in bm.blocks}):
            Hr, Hh = full(tj, s)
            H = Hr + s * sig / c0 * Hh
            nrm = max(np.abs(H).max(), 1e-300)
            for b in [b for b in bm.blocks if b.two_j == tj]:
                W = b.W.toarray() if hasattr(b.W, "toarray") else np.asarray(b.W)
                Hb = W.conj().T @ H @ W
                leak = max(leak, float(np.abs(H @ W - W @ Hb).max() / nrm))
                Hb = (Hb + Hb.conj().T).real / 2
                for k, e in enumerate(np.linalg.eigvalsh(Hb)):
                    levels[(tj // 2, b.irrep, k)] = float(e)
        e0 = min(e for (N, _, _), e in levels.items() if N == 0)
        levels = {k: v - e0 for k, v in levels.items()}
        ground = min(((k, v) for k, v in levels.items() if k[0] == 0), key=lambda t: t[1])[0][1]
        out[s] = {"levels": levels, "symmetry_leakage": leak, "ground_irrep_N0": ground}
    if sign is None:
        pick = [s for s in out if out[s]["ground_irrep_N0"].startswith("F2")]
        if len(pick) != 1:
            raise ValueError(f"Ham's theorem cannot fix the sign of sigma: N=0 ground irreps "
                             f"{ {s: out[s]['ground_irrep_N0'] for s in out} }")
        s = pick[0]
    r = out[s]
    r.update(sign=s, variant=variant, wells={"L": iL + 1, "R": iR + 1}, hop_normalization=c0)
    return r


def levels_to_json(levels: dict) -> list:
    return [{"N": N, "irrep": ir, "k": k, "E_cm": e} for (N, ir, k), e in sorted(levels.items(), key=lambda t: t[1])]


def levels_from_json(rows: list) -> dict:
    return {(r["N"], r["irrep"], r["k"]): r["E_cm"] for r in rows}
