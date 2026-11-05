#!/usr/bin/env python3
"""Ab initio -> multi-fidelity -> torsional levels with propagated uncertainty (H2O2).

  python scripts/case_h2o2_abinitio.py --out DIR [--high ccsd(t)] [--high-basis cc-pvtz]

1. Rigid HOOH torsion scan at a fixed geometry close to experiment
   (r_OO 1.4556 A, r_OH 0.9672 A, angle OOH 102.32 deg): low fidelity HF/cc-pVDZ
   every 10 deg; high fidelity (default CCSD(T)/cc-pVTZ) at 0, 30, ..., 180 deg.
   Points are mirrored (V(phi) = V(-phi)). Written as riss_drucker.pointset/1 JSON.
2. Delta model: E_high = E_low + delta, delta ~ GP (periodic kernel).
3. 200 posterior surface samples -> cosine series V1..V6 -> torsion_rotor
   (n_fold = 1) levels; F = hbar^2 / (2 I_red), I_red = m_H (r_OH sin angle)^2 / 2.
4. Compare with the experimental torsional levels of Hunt et al. (1965):
   about 11.4, 254.6, 370.9, 569.5, 775.9 cm-1 above the ground level.
--relaxed: constrained optimizations (frozen dihedral) at MP2/cc-pVDZ (needs
geomeTRIC); low fidelity = MP2/cc-pVDZ at those geometries, high fidelity =
single points at the same geometries; F from the relaxed minimum. Without it
the scan is rigid, which overestimates the trans barrier.
Requires PySCF (run locally / offline; no external API).

This script produces the scan point sets and the 1-D fixed-F baseline. The full
torsion-rotation model built from them (exact G(gamma), pseudopotential, ZPE,
half-angle frame, Bayesian update) is scripts/case_h2o2_rph.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

R_OO, R_OH, ANG = 1.4556, 0.9672, 102.32
HUNT_1965 = [11.4, 254.6, 370.9, 569.5, 775.9]
M_H = 1.00782503


def geometry(phi_deg: float):
    t, p = math.radians(ANG), math.radians(phi_deg)
    h1 = (R_OH * math.sin(t), 0.0, R_OH * math.cos(t))
    h2 = (R_OH * math.sin(t) * math.cos(p), R_OH * math.sin(t) * math.sin(p), R_OO - R_OH * math.cos(t))
    return [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, R_OO], ["H", *h1], ["H", *h2]]


_RELAXED = {}


def relaxed_geometry(phi_deg: float):
    """Optimize all coordinates except the HOOH dihedral at MP2/cc-pVDZ (cached)."""
    if phi_deg in _RELAXED:
        return _RELAXED[phi_deg]
    import logging
    import os
    import tempfile

    from pyscf import gto, mp, scf
    from pyscf.geomopt.geometric_solver import optimize

    logging.getLogger("geometric").setLevel(logging.ERROR)
    mol = gto.M(atom=[[a, (x, y, z)] for a, x, y, z in geometry(phi_deg)], basis="cc-pvdz", verbose=0)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("$freeze\ndihedral 3 1 2 4\n")
    try:
        opt = optimize(mp.MP2(scf.RHF(mol)), constraints=fh.name, maxsteps=100, verbose=0)
    finally:
        os.unlink(fh.name)
    c = opt.atom_coords(unit="Angstrom")
    out = [[s, *map(float, xyz)] for s, xyz in zip(["O", "O", "H", "H"], c)]
    _RELAXED[phi_deg] = out
    return out


def energy(phi_deg, method, basis, relaxed=False):
    from pyscf import cc, gto, mp, scf

    geo = relaxed_geometry(phi_deg) if relaxed else geometry(phi_deg)
    mol = gto.M(atom=[[a, (x, y, z)] for a, x, y, z in geo], basis=basis, verbose=0)
    hf = scf.RHF(mol).run(conv_tol=1e-11)
    if method == "hf":
        return hf.e_tot
    if method == "mp2":
        return mp.MP2(hf).run().e_tot
    if method == "ccsd(t)":
        c = cc.CCSD(hf).run(conv_tol=1e-9)
        return c.e_tot + c.ccsd_t()
    raise ValueError(method)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--high", default="ccsd(t)")
    ap.add_argument("--high-basis", default="cc-pvtz")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--relaxed", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="read energies from existing h2o2_{low,high}.json in --out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import riss_drucker as rd
    from riss_drucker.abinitio import io, multifidelity, torsion

    t0 = time.time()
    low_deg = np.arange(0, 181, 10.0)
    high_deg = np.arange(0, 181, 30.0)
    low_method = "mp2" if a.relaxed else "hf"
    def cached(name, deg):
        f = out / f"h2o2_{name}.json"
        if not (a.reuse and f.exists()):
            return None
        pts = {round(p["coordinate"], 6): p for p in io.read(f)["points"]}
        if not all(round(float(x), 6) in pts for x in deg):
            return None
        if a.relaxed:
            for x in deg:
                _RELAXED.setdefault(float(x), pts[round(float(x), 6)]["geometry"])
        return [pts[round(float(x), 6)]["energy"] for x in deg]

    low = cached("low", low_deg) or [energy(x, low_method, "cc-pvdz", a.relaxed) for x in low_deg]
    t_low = time.time() - t0
    high = cached("high", high_deg) or [energy(x, a.high, a.high_basis, a.relaxed) for x in high_deg]
    t_high = time.time() - t0 - t_low
    import pyscf
    geo_of = relaxed_geometry if a.relaxed else geometry
    scan = "relaxed (MP2/cc-pVDZ, frozen dihedral)" if a.relaxed else "rigid"
    for name, deg, E, meth, basis in (("low", low_deg, low, low_method.upper(), "cc-pVDZ"),
                                      ("high", high_deg, high, a.high.upper(), a.high_basis)):
        io.write(out / f"h2o2_{name}.json", "H2O2", meth, basis,
                 [{"coordinate": float(x), "energy": float(e), "geometry": geo_of(x)} for x, e in zip(deg, E)],
                 program=f"pyscf {pyscf.__version__}", coordinate="HOOH dihedral / deg",
                 provenance={"scan": scan, "start_geometry": {"r_OO": R_OO, "r_OH": R_OH, "angle_OOH": ANG}})

    def mirror(deg, E):
        deg, E = np.asarray(deg), np.asarray(E)
        m = (deg > 0) & (deg < 180)
        return np.radians(np.concatenate([deg, 360 - deg[m]])), np.concatenate([E, E[m]])

    H2CM = io.HARTREE_TO_CM
    lx, ly = mirror(low_deg, np.array(low) * H2CM)
    hx, hy = mirror(high_deg, np.array(high) * H2CM)
    dm = multifidelity.DeltaModel.fit(lx, ly, hx, hy, kind="periodic")
    grid = np.radians(np.arange(0, 360, 2.0))
    mean, var = dm.predict(grid)
    V = mean - mean.min()
    i_trans, i_cis = np.argmin(np.abs(grid - np.pi)), 0
    # check: leave-one-out of the high points
    loo = []
    for k in range(len(high_deg)):
        keep = high_deg != high_deg[k]
        x2, y2 = mirror(high_deg[keep], np.array(high)[keep] * H2CM)
        d2 = multifidelity.DeltaModel.fit(lx, ly, x2, y2, kind="periodic")
        mu, v = d2.predict(np.radians([high_deg[k]]))
        loo.append(((high[k] * H2CM - mu[0]) / math.sqrt(v[0] + 1e-12)))

    if a.relaxed:  # geometry of the lowest computed point
        g = geo_of(float(low_deg[int(np.argmin(low))]))
        o1, o2, h1 = (np.array(x[1:]) for x in g[:3])
        axis = (o2 - o1) / np.linalg.norm(o2 - o1)
        v = h1 - o1
        r_perp = float(np.linalg.norm(v - (v @ axis) * axis))
    else:
        r_perp = R_OH * math.sin(math.radians(ANG))
    F = 16.857629 / (M_H * r_perp**2 / 2)
    model = rd.compile_torsion_rotor(n_fold=1, v_orders=[1, 2, 3, 4, 5, 6], m_max=60)
    orders = [1, 2, 3, 4, 5, 6]
    det = torsion.propagate(model, F, grid, [V], orders, nlowest=4)["mean"][1:6]
    samp = dm.sample(grid, a.samples, seed=1)
    prop = torsion.propagate(model, F, grid, samp, orders, nlowest=4)
    lv_mean, lv_sd = prop["mean"][1:6], prop["sd"][1:6]
    # low fidelity alone, for comparison
    vl = dm.low(grid)
    low_only = torsion.propagate(model, F, grid, [vl - vl.min()], orders, nlowest=4)["mean"][1:6]
    res = {
        "geometry": {"start": {"r_OO": R_OO, "r_OH": R_OH, "angle_OOH": ANG}, "scan": scan},
        "methods": {"low": f"{low_method.upper()}/cc-pVDZ", "high": f"{a.high.upper()}/{a.high_basis}"},
        "timing_s": {"low_19_points": round(t_low, 1), "high_7_points": round(t_high, 1)},
        "F_cm": F,
        "barriers_cm": {"trans": float(V[i_trans]), "cis": float(V[i_cis]), "trans_sd": float(math.sqrt(var[i_trans])),
                        "cis_sd": float(math.sqrt(var[i_cis]))},
        "loo_z_high_points": [float(z) for z in loo],
        "levels_cm": {"multifidelity_mean": lv_mean.tolist(), "multifidelity_sd": lv_sd.tolist(),
                      "low_fidelity_only": low_only.tolist(), "experiment_hunt_1965": HUNT_1965},
    }
    (out / "h2o2_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
