#!/usr/bin/env python3
"""H2O2 torsion-rotation from ab initio, built dynamically (no fixed model type).

  python scripts/case_h2o2_rph.py --scan DIR [--out DIR] [--samples 100]

Input: the relaxed point sets written by `case_h2o2_abinitio.py --relaxed --out DIR`
(MP2/cc-pVDZ geometries with frozen dihedral, MP2/cc-pVDZ and CCSD(T)/cc-pVTZ energies).

1. The actual HOOH dihedral of every relaxed geometry is used (not the scan label).
2. MP2/cc-pVDZ Hessians (finite differences of analytic gradients; cached in
   h2o2_hessians.json) -> frequencies of the 5 other modes projected on translations,
   rotations and the path tangent -> adiabatic zero-point energy dZPE(gamma).
3. Delta-GP surface V(gamma) = MP2 + GP(CCSD(T) - MP2) on the actual dihedrals.
4. rd.abinitio.rph: generalized inertia tensor along the path -> G(gamma) for p^2, J_a J_b and
   J_a p, Podolsky pseudopotential, each expanded in the Fourier functions the PI group G8(H2O2)
   allows (half-angle frame; K-odd states get antiperiodic torsional functions).
5. rd.build compiles torsion (x) rotation with certificates; J = 0..2.
6. GP samples -> propagated level sd. Variants switched off one at a time (fixed F, no ZPE,
   no pseudopotential, nominal angles) show what each piece contributes.
7. Bayesian update against the torsional levels of Hunt et al. (1965) with corrections
   dV1..dV3 to the potential: prior sd = GP sd (+) model-error tau_ab chosen by evidence;
   leave-one-out predictions; prior->posterior shift in prior-sd units.
Requires PySCF for step 2 (cached afterwards).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HUNT_1965 = [11.4, 254.6, 370.9, 569.5, 775.9]


def hessians(geoms, path: Path, step: float = 0.005):
    if path.exists():
        d = json.loads(path.read_text())
        if len(d["hessians"]) == len(geoms):
            return [np.array(h) for h in d["hessians"]], d
    from pyscf import gto, mp, scf
    t0 = time.time()
    out = []
    for geo in geoms:
        sym = [g[0] for g in geo]
        X0 = np.array([g[1:] for g in geo], float) / 0.529177210903
        n = len(sym)
        H = np.zeros((3 * n, 3 * n))

        def grad(X):
            mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(sym, X)], basis="cc-pvdz", unit="Bohr", verbose=0)
            mf = scf.RHF(mol).run(conv_tol=1e-11)
            return mp.MP2(mf).run().nuc_grad_method().kernel().ravel()
        for k in range(3 * n):
            Xp, Xm = X0.copy().ravel(), X0.copy().ravel()
            Xp[k] += step
            Xm[k] -= step
            H[:, k] = (grad(Xp.reshape(n, 3)) - grad(Xm.reshape(n, 3))) / (2 * step)
        out.append((H + H.T) / 2)
    import pyscf
    meta = {"method": "MP2/cc-pVDZ", "program": f"pyscf {pyscf.__version__}", "step_bohr": step,
            "seconds": round(time.time() - t0, 1), "hessians": [h.tolist() for h in out]}
    path.write_text(json.dumps(meta))
    return out, meta


def rotation_comparison(rph, B, gam, geoms, V0, dzpe, a, hunt):
    """n = 0 torsion-rotation levels (J <= jmax, Ka <= kamax) vs the JPL catalogue (Petkie et al. 1995,
    Camy-Peyret et al. 1992): prediction, then a joint update of V1..V3 and the three rotational
    G-matrix constants with weak priors; held-out J check."""
    from riss_drucker.data.catalog import levels_from_cat, read_cat
    from riss_drucker.fit import Observation, Prior, fit
    lv, rep = levels_from_cat(read_cat(a.jpl), max_spread=5e-3)
    # catalogue v: 0 -> n=0 lower torsional component, 3 -> n=0 upper component
    obs_lv = {(J, Ka, Kc, v): E for (J, Ka, Kc, v), (E, s) in lv.items()
              if v in (0, 3) and J <= a.jmax_rot and Ka <= a.kamax}
    spec, theta, _ = rph.build_h2o2_spec(np.degrees(gam), geoms, V0, dzpe, m_max=a.m_max,
                                         J=list(range(a.jmax_rot + 1)))
    bm = B.compile_build(spec)

    def assign(th):
        Lz = bm.operator_blocks("Jz Jz")
        pool = {}
        eigs = bm.eig(th)
        e0 = min(float(w[0]) for w, _ in eigs)
        for b, (w, V), L in zip(bm.blocks, eigs, Lz):
            for i in range(V.shape[1]):
                if w[i] - e0 > 200:     # n = 0 manifold only (n = 1 starts near 250 cm-1)
                    continue
                ka = int(round(np.sqrt(max(float(np.real(V[:, i] @ L @ V[:, i])), 0.0))))
                pool.setdefault((b.two_j // 2, ka), []).append((w[i], bm.model.blocks[[x for x in
                                                                range(len(bm.blocks)) if bm.blocks[x] is b][0]]["key"], i))
        table = {}
        for (J, ka), items in pool.items():
            items.sort()
            nd = 1 if ka == 0 else 2
            low, up = items[:nd], items[nd:2 * nd]
            kcs = sorted([kc for (JJ, KK, kc, v) in obs_lv if JJ == J and KK == ka and v == 0], reverse=True)
            for kc, it in zip(kcs, low):
                table[(J, ka, kc, 0)] = it
            kcs = sorted([kc for (JJ, KK, kc, v) in obs_lv if JJ == J and KK == ka and v == 3], reverse=True)
            for kc, it in zip(kcs, up):
                table[(J, ka, kc, 3)] = it
        return table
    table = assign(theta)
    ground = table[(0, 0, 0, 0)]
    common = sorted(k for k in obs_lv if k in table and k != (0, 0, 0, 0))

    def predicted(th):
        e = {}
        for b, (w, _) in zip(bm.model.blocks, bm.eig(th)):
            for i, x in enumerate(w):
                e[(b["key"], i)] = x
        g = e[(ground[1], ground[2])]
        return {k: e[(table[k][1], table[k][2])] - g for k in common}
    pr = predicted(theta)
    resid0 = {k: obs_lv[k] - obs_lv[(0, 0, 0, 0)] - pr[k] for k in common}
    # joint update: V1..V3 and the rotational constants (c0 terms of Gxx, Gyy, Gzz)
    free = [n for n in ("V_c2", "V_c4", "V_c6", "Gxx_c0", "Gyy_c0", "Gzz_c0", "Gzz_c2", "Gzz_c4") if n in theta]
    sd = {n: (30.0 if n.startswith("V") else 0.02 * abs(theta[n]) if n.endswith("_c0") else
              max(0.5 * abs(theta[n]), 0.05)) for n in free}
    jfit = max(2, a.jmax_rot - 3)

    def obs_for(keys):
        o = []
        for k in keys:
            E = obs_lv[k] - obs_lv[(0, 0, 0, 0)]
            o.append(Observation(table[k][1], table[k][2], E, 0.02, lower_block=ground[1], lower_index=ground[2]))
        return o
    fit_keys = [k for k in common if k[0] <= jfit]
    hold_keys = [k for k in common if k[0] > jfit]
    fr = fit(bm.model, obs_for(fit_keys), theta, prior=Prior({n: theta[n] for n in free}, sd),
             fixed=[n for n in bm.param_names if n not in free], method="dense")
    pp = predicted(fr.theta)
    resid1 = {k: obs_lv[k] - obs_lv[(0, 0, 0, 0)] - pp[k] for k in common}

    def rms(d, keys):
        return float(np.sqrt(np.mean([d[k] ** 2 for k in keys]))) if keys else None
    split = lambda d: {f"Ka={ka}": float(np.mean([d[k] for k in common if k[1] == ka and k[3] == 3 and k[0] <= 4]))
                       for ka in range(a.kamax + 1)}
    torsional_splitting = {}
    for J, ka in sorted({(k[0], k[1]) for k in common}):
        lo_ = [obs_lv[k] for k in common if k[0] == J and k[1] == ka and k[3] == 0]
        up_ = [obs_lv[k] for k in common if k[0] == J and k[1] == ka and k[3] == 3]
        mlo = [pr[k] for k in common if k[0] == J and k[1] == ka and k[3] == 0]
        mup = [pr[k] for k in common if k[0] == J and k[1] == ka and k[3] == 3]
        if lo_ and up_ and J == max(ka, 1):
            torsional_splitting[f"J={J},Ka={ka}"] = {"observed": float(np.mean(up_) - np.mean(lo_)),
                                                     "predicted_ab_initio": float(np.mean(mup) - np.mean(mlo))}
    split_after = {}
    for J, ka in sorted({(k[0], k[1]) for k in common}):
        if J != max(ka, 1):
            continue
        mlo = [pp[k] for k in common if k[0] == J and k[1] == ka and k[3] == 0]
        mup = [pp[k] for k in common if k[0] == J and k[1] == ka and k[3] == 3]
        if mlo and mup:
            torsional_splitting[f"J={J},Ka={ka}"]["after_update"] = float(np.mean(mup) - np.mean(mlo))
    return {"levels_compared": len(common), "catalogue": rep, "jmax": a.jmax_rot, "kamax": a.kamax,
            "prediction_rms_cm": rms(resid0, common),
            "prediction_rms_by_Ka_cm": {f"Ka={ka}": rms(resid0, [k for k in common if k[1] == ka])
                                        for ka in range(a.kamax + 1)},
            "torsional_splitting_vs_Ka": torsional_splitting,
            "update": {"free": free, "prior_sd": sd, "fit_J_max": jfit, "n_fit": len(fit_keys),
                       "n_held_out": len(hold_keys), "chi2_per_dof": fr.reduced_chi2,
                       "shift": {n: fr.theta[n] - theta[n] for n in free},
                       "shift_in_prior_sd": {n: (fr.theta[n] - theta[n]) / sd[n] for n in free},
                       "fit_rms_cm": rms(resid1, fit_keys), "held_out_rms_cm": rms(resid1, hold_keys),
                       "held_out_max_cm": float(max(abs(resid1[k]) for k in hold_keys)) if hold_keys else None,
                       "J0_levels_after": None}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="directory with h2o2_low.json / h2o2_high.json (relaxed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--m-max", type=int, default=40)
    ap.add_argument("--jpl", default=None, help="JPL catalogue c034004.cat for K-resolved torsion-rotation levels")
    ap.add_argument("--jmax-rot", type=int, default=8)
    ap.add_argument("--kamax", type=int, default=3)
    a = ap.parse_args()
    scan = Path(a.scan)
    out = Path(a.out or a.scan)
    out.mkdir(parents=True, exist_ok=True)

    import riss_drucker as rd
    from riss_drucker import build as B
    from riss_drucker.abinitio import io, multifidelity, rph
    from riss_drucker.fit import Observation, Prior, fit

    lo, hi = io.read(scan / "h2o2_low.json"), io.read(scan / "h2o2_high.json")
    geoms = [p["geometry"] for p in lo["points"]]
    gam = rph.actual_dihedrals(geoms)
    El = np.array([p["energy"] for p in lo["points"]]) * io.HARTREE_TO_CM
    hgeo = [p["geometry"] for p in hi["points"]]
    hgam = rph.actual_dihedrals(hgeo)
    Eh = np.array([p["energy"] for p in hi["points"]]) * io.HARTREE_TO_CM

    # 2. adiabatic ZPE
    Hs, hmeta = hessians(geoms, out / "h2o2_hessians.json")
    zpe, freqs = [], [None] * len(geoms)
    # tangent: finite difference of body coordinates along the path, rotated back to each lab frame
    body = []
    for geo in geoms:
        r, gg, Rb, m = rph.h2o2_body_frame(geo)
        if gg > math.pi:
            r = r @ np.diag([-1.0, 1, 1])
        body.append(r)
    body = np.array(body)
    for i, (geo, H) in enumerate(zip(geoms, Hs)):
        r, gg, Rb, m = rph.h2o2_body_frame(geo)
        flip = np.diag([-1.0, 1, 1]) if gg > math.pi else np.eye(3)
        j0, j1 = max(i - 1, 0), min(i + 1, len(geoms) - 1)
        t_body = (body[j1] - body[j0]) / max(gam[j1] - gam[j0], 1e-9)
        t_lab = (t_body @ flip) @ Rb                   # body (flipped branch) -> lab
        X = np.array([g[1:] for g in geo])
        w = rph.projected_frequencies(H, X, m, tangent=t_lab)
        freqs[i] = w.tolist()
        zpe.append(0.5 * float(np.sum(w[w > 0])))
    zpe = np.array(zpe)
    dzpe = zpe - zpe[int(np.argmin(El))]

    # 3. Delta-GP on actual dihedrals (mirrored to 2pi)
    def mirror(g, E):
        mk = (g > 1e-6) & (g < math.pi - 1e-6)
        return np.concatenate([g, 2 * math.pi - g[mk]]), np.concatenate([E, E[mk]])
    dm = multifidelity.DeltaModel.fit(*mirror(gam, El), *mirror(hgam, Eh), kind="periodic")
    Vmean, Vvar = dm.predict(gam)
    V0 = Vmean - Vmean.min()

    # 4./5. dynamic model and levels
    J = [0, 1, 2]
    t0 = time.time()
    spec, theta, info = rph.build_h2o2_spec(np.degrees(gam), geoms, V0, dzpe, m_max=a.m_max, J=J)
    bm = B.compile_build(spec)
    t_build = time.time() - t0

    def j0_levels(model, th):
        ev = [w for b, (w, _) in zip(model.blocks, model.eig(th)) if b.two_j == 0]
        e = np.sort(np.concatenate(ev))
        return e[1:6] - e[0], e

    lv, e_all = j0_levels(bm, theta)
    # rotational structure: lowest K=0/1 levels of J = 1
    jlev = {}
    for b, (w, _) in zip(bm.blocks, bm.eig(theta)):
        jlev.setdefault(b.two_j // 2, []).append((b.irrep, (w[:4] - e_all[0]).tolist()))

    # 6. propagation through GP samples (V only; kinetic terms are geometric)
    samp = dm.sample(gam, a.samples, seed=1)
    lv_s = []
    for s in samp:
        sp2, th2, _ = rph.build_h2o2_spec(np.degrees(gam), geoms, s - s.min(), dzpe, m_max=a.m_max, J=[0],
                                         include_rotation=False)
        bm2 = B.compile_build(sp2)
        lv_s.append(j0_levels(bm2, th2)[0])
    lv_s = np.array(lv_s)

    # ablations
    def variant(**kw):
        V = kw.pop("V", V0)
        z = kw.pop("zpe", dzpe)
        sp_, th_, _ = rph.build_h2o2_spec(np.degrees(gam), geoms, V, z, m_max=a.m_max, J=[0], include_rotation=False,
                                        **kw)
        return j0_levels(B.compile_build(sp_), th_)[0].tolist()
    ablate = {"no_zpe": variant(zpe=None), "no_pseudopotential": variant(include_pseudo=False),
              "low_fidelity_only": variant(V=(El - El.min()))}
    # fixed-F 1-D model on nominal angles (the previous treatment)
    nominal = np.radians([p["coordinate"] for p in lo["points"]])
    dmn = multifidelity.DeltaModel.fit(*mirror(nominal, El),
                                       *mirror(np.radians([p["coordinate"] for p in hi["points"]]), Eh),
                                       kind="periodic")
    grid = np.radians(np.arange(0, 360, 2.0))
    Vn = dmn.predict(grid)[0]
    m1 = rd.compile_torsion_rotor(n_fold=1, v_orders=[1, 2, 3, 4, 5, 6], m_max=60)
    from riss_drucker.abinitio import torsion
    old = torsion.propagate(m1, 36.40062908011135, grid, [Vn - Vn.min()], [1, 2, 3, 4, 5, 6], nlowest=4)["mean"][1:6]

    # 7. Bayesian update: dV1..dV3 against Hunt (transitions from the ground level)
    corr = ["dV1: cos1[t]", "dV2: cos2[t]", "dV3: cos3[t]"]
    spec0, theta0, _ = rph.build_h2o2_spec(np.degrees(gam), geoms, V0, dzpe, m_max=a.m_max, J=[0],
                                           include_rotation=False)
    have = {t.split(":", 1)[1].strip() for t in spec0.terms}
    corr = [c for c in corr if c.split(":", 1)[1].strip() not in have]
    # corrections duplicate existing cos terms of V: shift those parameters instead
    vmap = {f"dV{k}": f"V_c{2 * k}" for k in (1, 2, 3)}
    bmc = B.compile_build(spec0.with_terms(spec0.terms + corr)) if corr else B.compile_build(spec0)
    th_c = dict(theta0)
    # assign by energy order at the prior
    order = sorted([(w[i], b, i) for b, (w, _) in enumerate(bmc.eig(th_c)) for i in range(len(w))])[:6]
    keys = [bmc.model.blocks[b]["key"] for _, b, _ in order]
    g_key, g_idx = keys[0], order[0][2]
    sig_obs = 0.5

    def obs_list(idx):
        return [Observation(keys[k + 1], order[k + 1][2], HUNT_1965[k], sig_obs, lower_block=g_key,
                            lower_index=g_idx) for k in idx]
    # GP prior on cosine coefficients from the samples
    cs = []
    for s in samp:
        A_ = np.stack([np.cos(k * gam) for k in (1, 2, 3)] + [np.ones_like(gam)], -1)
        c, *_ = np.linalg.lstsq(A_, s - s.mean(), rcond=None)
        cs.append(c[:3])
    gp_sd = np.std(np.array(cs), axis=0)
    corr_names = [vmap[f"dV{k}"] for k in (1, 2, 3)]
    fixed = [n for n in bmc.param_names if n not in corr_names]
    # type-II maximum likelihood for tau_ab: linearized model y = y0 + A d, d ~ N(0, S),
    # y - y0 ~ N(0, sig^2 I + A S A^T) (exact Gaussian marginal of the linearization)
    y_obs = np.array(HUNT_1965)
    y0 = j0_levels(bmc, th_c)[0]
    A = np.zeros((5, 3))
    for k, n in enumerate(corr_names):
        tp, tm = dict(th_c), dict(th_c)
        tp[n] += 1.0
        tm[n] -= 1.0
        A[:, k] = (j0_levels(bmc, tp)[0] - j0_levels(bmc, tm)[0]) / 2
    best = None
    scan_tau = []
    for tau in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        sd = np.sqrt(gp_sd**2 + tau**2)
        C = sig_obs**2 * np.eye(5) + A @ np.diag(sd**2) @ A.T
        r = y_obs - y0
        sign, logdet = np.linalg.slogdet(C)
        lz = float(-0.5 * r @ np.linalg.solve(C, r) - 0.5 * logdet - 2.5 * math.log(2 * math.pi))
        scan_tau.append({"tau_ab": tau, "log_evidence": lz})
        if best is None or lz > best[0]:
            best = (lz, tau, None, sd)
    lz, tau, _, sd = best
    pri = Prior({n: th_c[n] for n in corr_names}, {n: float(sd[k]) for k, n in enumerate(corr_names)})
    fr = fit(bmc.model, obs_list(range(5)), th_c, prior=pri, fixed=fixed, method="dense")
    best = (lz, tau, fr, sd)
    lz, tau, fr, sd = best
    post = {f"dV{k + 1}": fr.theta[n] - th_c[n] for k, n in enumerate(corr_names)}
    shifts = {k: post[k] / float(s) for k, s in zip(("dV1", "dV2", "dV3"), sd)}
    post_lv = [lv_["calculated"] for lv_ in fr.levels]
    loo = []
    for k in range(5):
        keep = [i for i in range(5) if i != k]
        pri = Prior({n: th_c[n] for n in corr_names}, {n: float(sd[j]) for j, n in enumerate(corr_names)})
        frk = fit(bmc.model, obs_list(keep), th_c, prior=pri, fixed=fixed, method="dense")
        pred = j0_levels(bmc, frk.theta)[0][k]
        loo.append({"held_out": HUNT_1965[k], "predicted": float(pred), "residual": float(HUNT_1965[k] - pred)})

    res = {
        "inputs": {"geometries": "MP2/cc-pVDZ relaxed (frozen dihedral)", "low": lo["method"] + "/" + lo["basis"],
                   "high": hi["method"] + "/" + hi["basis"], "hessians": hmeta.get("method"),
                   "actual_dihedral_deg": np.degrees(gam).round(3).tolist(),
                   "nominal_scan_deg": [p["coordinate"] for p in lo["points"]]},
        "model": {"terms": len(spec.terms), "group": spec.group, "J": J, "compile_seconds": round(t_build, 2),
                  "certificates_passed": all(c["passed"] for c in bm.certificates),
                  "n_certificates": len(bm.certificates), **{k: v for k, v in info.items() if k != "actual_dihedral_deg"}},
        "zpe": {"dZPE_cm": dzpe.round(2).tolist(), "frequencies_cis": freqs[0], "frequencies_trans": freqs[-1]},
        "barriers_cm": {"trans_electronic": float(V0[np.argmin(np.abs(gam - math.pi))]),
                        "cis_electronic": float(V0[np.argmin(gam)]),
                        "trans_with_zpe": float((V0 + dzpe)[np.argmin(np.abs(gam - math.pi))] - (V0 + dzpe).min()),
                        "cis_with_zpe": float((V0 + dzpe)[np.argmin(gam)] - (V0 + dzpe).min())},
        "levels_J0_cm": {"rph_dynamic": lv.tolist(), "gp_sd": lv_s.std(axis=0).tolist(), **ablate,
                         "previous_1d_fixed_F_nominal_angles": old.tolist(), "experiment_hunt_1965": HUNT_1965},
        "rotational_levels_cm": {str(k): v for k, v in jlev.items()},
        "bayesian_update": {"sigma_obs": sig_obs, "tau_ab_scan": scan_tau, "tau_ab": tau, "gp_sd_dV": gp_sd.tolist(),
                            "posterior_dV": {k: post[k] for k in ("dV1", "dV2", "dV3")},
                            "posterior_sd_dV": {f"dV{k + 1}": fr.sigma[n] for k, n in enumerate(corr_names)},
                            "shift_in_prior_sd": shifts, "posterior_levels": post_lv, "chi2": fr.chi2,
                            "leave_one_out": loo},
    }
    if a.jpl:
        res["rotation"] = rotation_comparison(rph, B, gam, geoms, V0, dzpe, a, HUNT_1965)
    (out / "h2o2_rph_result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("levels_J0_cm", "barriers_cm", "bayesian_update")}, indent=1,
                     default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
