#!/usr/bin/env python3
"""Ab initio CH4+ (and CH4) for the methane-cation case study. Stages are cached as JSON in --out.

  python scripts/abinitio_ch4plus.py --out DIR [--stages min,ts,irc,hess,ccsdt,neutral] [--workers N]

Needs only PySCF, geomeTRIC, NumPy and SciPy (the riss_drucker helpers it uses are loaded from
the repository files; no build of the compiled core is required). Run from a repository checkout.

Levels: UMP2/cc-pVTZ geometries, IRC and Hessians (finite differences of analytic gradients);
UCCSD(T)/cc-pVTZ single points at the minimum, saddle and along the path (energy correction),
and for neutral CH4 (adiabatic ionization energy). PySCF + geomeTRIC, CPU only.

minimum : C2v structure of CH4+ (two long, two short C-H bonds)
ts      : Cs saddle connecting two adjacent minima (pseudorotation; guess = aligned midpoint of the
          minimum and its image with one long-bond H swapped with one short-bond H)
irc     : mass-weighted steepest descent from the saddle to the minimum (one side; the other side
          is equivalent by symmetry - checked: the end point must reproduce the minimum)
hess    : Cartesian Hessians at points along the IRC (mass independent: used for CH4+ and CD4+)
ccsdt   : UCCSD(T) energies at the minimum, saddle and selected IRC points
neutral : CH4 Td minimum, Hessian and UCCSD(T) energy
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

BOHR = 0.529177210903
WORKERS = 1


def _helpers():
    """rph.py and tunneling.py loaded straight from the repository (pure NumPy): the ab initio
    stages need neither an installed riss_drucker nor its compiled core."""
    import importlib.util
    base = Path(__file__).resolve().parents[1] / "python" / "riss_drucker" / "abinitio"
    mods = {}
    for name in ("rph", "tunneling"):
        spec = importlib.util.spec_from_file_location(f"_rd_{name}", base / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["rph"], mods["tunneling"]
SYM = ["C", "H", "H", "H", "H"]
BASIS = "cc-pvtz"


def mol_of(X, charge=1, spin=1, basis=BASIS):
    from pyscf import gto
    return gto.M(atom=[[s, tuple(x)] for s, x in zip(SYM, X)], basis=basis, charge=charge, spin=spin, verbose=0)


def mp2(X, grad=False, charge=1, spin=1):
    from pyscf import mp, scf
    m = mol_of(X, charge, spin)
    mf = (scf.UHF(m) if spin else scf.RHF(m)).run(conv_tol=1e-11)
    pt = mp.MP2(mf).run()
    if not grad:
        return pt.e_tot
    g = pt.nuc_grad_method().kernel()
    return pt.e_tot, np.asarray(g) / BOHR            # Eh / Angstrom


def ccsdt(X, charge=1, spin=1):
    from pyscf import cc, scf
    m = mol_of(X, charge, spin)
    mf = (scf.UHF(m) if spin else scf.RHF(m)).run(conv_tol=1e-11)
    c = (cc.UCCSD(mf) if spin else cc.CCSD(mf)).run(conv_tol=1e-9)
    return float(c.e_tot + c.ccsd_t())


def _grad_task(args):
    X, charge, spin = args
    return mp2(X, True, charge, spin)[1]


def hessian(X, charge=1, spin=1, step=0.005):
    """Cartesian Hessian (Eh/bohr^2) by central differences of analytic MP2 gradients.
    The 6N displaced gradients are independent: with --workers > 1 they run in parallel
    processes (OMP threads per process = OMP_NUM_THREADS / workers is the user's choice)."""
    Xb = np.asarray(X) / BOHR
    n = len(Xb)
    tasks = []
    for k in range(3 * n):
        d = np.zeros(3 * n)
        d[k] = step
        tasks.append((((Xb.ravel() + d).reshape(n, 3)) * BOHR, charge, spin))
        tasks.append((((Xb.ravel() - d).reshape(n, 3)) * BOHR, charge, spin))
    if WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(WORKERS) as ex:
            grads = list(ex.map(_grad_task, tasks))
    else:
        grads = [_grad_task(t) for t in tasks]
    H = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        H[:, k] = ((grads[2 * k] - grads[2 * k + 1]) * BOHR).ravel() / (2 * step)
    return (H + H.T) / 2


def optimize(X, transition=False, charge=1, spin=1):
    from pyscf import mp, scf
    from pyscf.geomopt.geometric_solver import optimize as gopt
    logging.getLogger("geometric").setLevel(logging.ERROR)
    m = mol_of(X, charge, spin)
    mf = (scf.UHF(m) if spin else scf.RHF(m)).run(conv_tol=1e-10)
    kw = {"transition": True, "hessian": "first"} if transition else {}
    out = gopt(mp.MP2(mf), maxsteps=150, verbose=0, **kw)
    return out.atom_coords(unit="Angstrom")


def stage_min(out):
    a1, a2 = math.radians(62.5), math.radians(27.5)
    X0 = np.array([[0, 0, 0], [1.18 * math.sin(a1), 0, 1.18 * math.cos(a1)], [-1.18 * math.sin(a1), 0, 1.18 * math.cos(a1)],
                   [0, 1.09 * math.sin(a2), -1.09 * math.cos(a2)], [0, -1.09 * math.sin(a2), -1.09 * math.cos(a2)]])
    X = optimize(X0)
    return {"xyz": X.tolist(), "e_ump2": mp2(X)}


def masses_of(iso="H"):
    MASS = _helpers()[0].MASS
    return np.array([MASS["C"]] + [MASS[iso]] * 4)


def optimize_constrained(X, constraints: str, charge=1, spin=1):
    import os
    import tempfile
    from pyscf import mp, scf
    from pyscf.geomopt.geometric_solver import optimize as gopt
    logging.getLogger("geometric").setLevel(logging.ERROR)
    m = mol_of(X, charge, spin)
    mf = (scf.UHF(m) if spin else scf.RHF(m)).run(conv_tol=1e-10)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(constraints)
    try:
        out = gopt(mp.MP2(mf), constraints=fh.name, maxsteps=150, verbose=0)
    finally:
        os.unlink(fh.name)
    return out.atom_coords(unit="Angstrom")


def stage_ts(out, mn):
    """Cs saddle between two adjacent minima. The bonds that exchange roles (one short-bond H
    becoming long, one long-bond H becoming short) are equal at the saddle by symmetry: minimize
    on the seam r(C-Ha) = r(C-Hb) = d for several d, refine the lowest seam point as a transition
    state, and require exactly one imaginary frequency."""
    rph, tun = _helpers()
    projected_frequencies, kabsch = rph.projected_frequencies, tun.kabsch
    X = np.array(mn["xyz"])
    d = [np.linalg.norm(X[i] - X[0]) for i in range(1, 5)]
    order = np.argsort(d)
    a, b = 1 + order[0], 1 + order[3]          # short-bond H a, long-bond H b
    Y = X.copy()
    Y[[a, b]] = Y[[b, a]]
    Y = kabsch(X, Y, masses_of())
    guess = (X + Y) / 2
    seam = []
    for dd in (1.10, 1.12, 1.14, 1.16):
        Z = optimize_constrained(guess, f"$set\ndistance 1 {a + 1} {dd}\ndistance 1 {b + 1} {dd}\n")
        seam.append((mp2(Z), dd, Z))
    e0, d0, Z0 = min(seam, key=lambda t: t[0])
    T = optimize(Z0, transition=True)
    H = hessian(T)
    w = projected_frequencies(H, T, masses_of())
    nimag = int(np.sum(w < -1.0))
    if nimag != 1:
        raise RuntimeError(f"saddle search gave {nimag} imaginary frequencies")
    return {"xyz": T.tolist(), "e_ump2": mp2(T), "frequencies": w.tolist(), "swap": [int(a), int(b)],
            "seam_scan": [[float(e), float(dd)] for e, dd, _ in seam], "hessian": H.tolist()}


def stage_irc(out, ts, mn, h=0.02, max_steps=400):
    """Mass-weighted steepest descent (Euler steps of h amu^1/2 A, with a corrector) from the saddle."""
    kabsch = _helpers()[1].kabsch
    m = masses_of()
    T = np.array(ts["xyz"])
    H = np.array(ts["hessian"])
    sm = np.repeat(np.sqrt(m), 3)
    w, V = np.linalg.eigh(H / np.outer(sm, sm))
    v = V[:, 0]                                           # imaginary mode (mass weighted)
    disp = (v / sm).reshape(5, 3)
    best = None
    for sgn in (1, -1):                                   # pick the side that leads to the given minimum
        X = T + sgn * 0.02 * disp / np.sqrt(np.sum(m[:, None] * disp**2))
        path, E = [T.copy(), X.copy()], [ts["e_ump2"]]
        e, g = mp2(X, True)
        E.append(e)
        for _ in range(max_steps):
            gm = (g / np.sqrt(m)[:, None])
            nrm = np.linalg.norm(gm)
            if nrm < 2e-4:
                break
            Xp = X - h * (gm / nrm) / np.sqrt(m)[:, None]
            ep, gp = mp2(Xp, True)
            gmp = gp / np.sqrt(m)[:, None]
            Xn = X - h * 0.5 * (gm / nrm + gmp / max(np.linalg.norm(gmp), 1e-12)) / np.sqrt(m)[:, None]
            en, gn = mp2(Xn, True)
            if en > e:                                    # passed the minimum
                break
            X, e, g = Xn, en, gn
            path.append(X.copy())
            E.append(e)
        endpoint = kabsch(np.array(mn["xyz"]), X, m)
        dev = float(np.max(np.abs(np.sort(np.linalg.norm(endpoint - endpoint[0], axis=1)) -
                                  np.sort(np.linalg.norm(np.array(mn["xyz"]) - np.array(mn["xyz"])[0], axis=1)))))
        cand = {"path": [p.tolist() for p in path], "e_ump2": E, "endpoint_bond_deviation_A": dev,
                "endpoint_minus_min_energy_cm": (E[-1] - mn["e_ump2"]) * 219474.6313705, "side": sgn}
        if best is None or dev < best["endpoint_bond_deviation_A"]:
            best = cand
        if dev < 5e-3:
            break
    return best


def stage_hess(out, irc, npts=12):
    path = irc["path"]
    idx = sorted(set(np.linspace(0, len(path) - 1, npts).round().astype(int).tolist()))
    return {"indices": idx, "hessians": [hessian(np.array(path[i])).tolist() for i in idx]}


def stage_ccsdt(out, irc, npts=6):
    path = irc["path"]
    idx = sorted(set(np.linspace(0, len(path) - 1, npts).round().astype(int).tolist()))
    return {"indices": idx, "e_ccsdt": [ccsdt(np.array(path[i])) for i in idx]}


def stage_neutral(out):
    r = 1.09 / math.sqrt(3)
    X0 = np.array([[0, 0, 0], [r, r, r], [r, -r, -r], [-r, r, -r], [-r, -r, r]])
    X = optimize(X0, charge=0, spin=0)
    H = hessian(X, charge=0, spin=0)
    return {"xyz": X.tolist(), "e_mp2": mp2(X, charge=0, spin=0), "e_ccsdt": ccsdt(X, charge=0, spin=0),
            "hessian": H.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="min,ts,irc,hess,ccsdt,neutral")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for finite-difference Hessians")
    a = ap.parse_args()
    global WORKERS
    WORKERS = max(1, a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyscf
    meta = {"program": f"pyscf {pyscf.__version__}", "basis": BASIS}

    def cached(name, fn, *args):
        f = out / f"ch4plus_{name}.json"
        if f.exists():
            return json.loads(f.read_text())
        t = time.time()
        r = fn(out, *args)
        r["meta"] = {**meta, "seconds": round(time.time() - t, 1)}
        f.write_text(json.dumps(r))
        print(f"stage {name}: {r['meta']['seconds']} s", flush=True)
        return r
    st = a.stages.split(",")
    mn = cached("min", stage_min)
    if "ts" in st:
        ts = cached("ts", stage_ts, mn)
    if "irc" in st:
        irc = cached("irc", stage_irc, ts, mn)
    if "hess" in st:
        cached("hess", stage_hess, irc)
    if "ccsdt" in st:
        cached("ccsdt", stage_ccsdt, irc)
    if "neutral" in st:
        cached("neutral", stage_neutral)
    return 0


if __name__ == "__main__":
    sys.exit(main())
