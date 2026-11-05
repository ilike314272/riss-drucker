#!/usr/bin/env python3
"""Ab initio CH4+ (and CH4) for the methane-cation case study. Stages are cached as JSON in --out.

  python scripts/abinitio_ch4plus.py --out DIR [--stages min,ts,irc,hess,ccsdt,neutral,plane2d] [--workers N]

Needs only PySCF, geomeTRIC, ASE (NEB), NumPy and SciPy (the riss_drucker helpers it uses are loaded from
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
plane2d : UMP2 energies on the Cartesian reaction plane of the double well (CH4+ and CD4+ planes),
          checkpointed; used by the first-principles tunneling (riss_drucker.abinitio.firstprinciples)
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
NEB_STEPS = 300


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


def mol_of(X, charge=1, spin=1, basis=None):
    from pyscf import gto
    return gto.M(atom=[[s, tuple(x)] for s, x in zip(SYM, X)], basis=basis or BASIS, charge=charge, spin=spin,
                 verbose=0)


def scf_ground(m, spin):
    """SCF converged to the lowest solution: for the open-shell cation an internal stability
    analysis follows any lower UHF solution (near the Jahn-Teller region a from-scratch guess
    can land on an excited electronic state)."""
    from pyscf import scf
    mf = (scf.UHF(m) if spin else scf.RHF(m)).run(conv_tol=1e-11)
    if spin:
        for _ in range(4):
            mo, _, stable, _ = mf.stability(return_status=True)
            if stable:
                break
            dm = mf.make_rdm1(mo, mf.mo_occ)
            mf = scf.UHF(m).run(dm, conv_tol=1e-11)
    return mf


def mp2(X, grad=False, charge=1, spin=1):
    from pyscf import mp
    m = mol_of(X, charge, spin)
    mf = scf_ground(m, spin)
    pt = mp.MP2(mf).run()
    if not grad:
        return pt.e_tot
    g = pt.nuc_grad_method().kernel()
    return pt.e_tot, np.asarray(g) / BOHR            # Eh / Angstrom


def ccsdt(X, charge=1, spin=1):
    from pyscf import cc
    m = mol_of(X, charge, spin)
    mf = scf_ground(m, spin)
    c = (cc.UCCSD(mf) if spin else cc.CCSD(mf)).run(conv_tol=1e-9)
    return float(c.e_tot + c.ccsd_t())


_POOL = None


def _pool():
    """One process pool for the whole run, started with 'spawn': forking a process whose OpenMP
    thread pool is already active can deadlock the children."""
    global _POOL
    if _POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"))
    return _POOL


def _grad_task(args):
    X, charge, spin, basis = args
    global BASIS
    BASIS = basis                      # spawned workers do not inherit the command-line basis
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
        tasks.append((((Xb.ravel() + d).reshape(n, 3)) * BOHR, charge, spin, BASIS))
        tasks.append((((Xb.ravel() - d).reshape(n, 3)) * BOHR, charge, spin, BASIS))
    if WORKERS > 1:
        grads = list(_pool().map(_grad_task, tasks))
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


def mirror_symmetrize(G, a, b, m):
    """Average G with its image under the reflection that swaps H_a and H_b: plane through the
    centre of mass with normal along r_a - r_b. Returns the symmetrized geometry and the 3x3
    reflection (the other atoms are pushed into the plane)."""
    com = (m[:, None] * G).sum(0) / m.sum()
    n = G[a] - G[b]
    n = n / np.linalg.norm(n)
    M = np.eye(3) - 2 * np.outer(n, n)
    Gm = (G - com) @ M.T + com
    Gm[[a, b]] = Gm[[b, a]]
    return (G + Gm) / 2, M


def _image_gradients(images):
    tasks = [(G, 1, 1, BASIS) for G in images]
    if WORKERS > 1:
        out = list(_pool().map(_eg_task, tasks))
    else:
        out = [_eg_task(t) for t in tasks]
    return [o[0] for o in out], [o[1] for o in out]


def _eg_task(args):
    X, charge, spin, basis = args
    global BASIS
    BASIS = basis
    return mp2(X, True, charge, spin)


def neb(chain, m, k_spring=0.5, iters=300, fmax=3e-3, ci_after=20, log=None, dt_max=0.15, step_max=0.02):
    """Climbing-image nudged elastic band (Henkelman tangents, FIRE optimizer) on an initial chain of
    aligned Cartesian geometries (Angstrom; forces Eh/Angstrom); the end points are the two minima
    and stay fixed. Returns images, energies, climbing index, iterations, final max force."""
    imgs = [np.array(g, float) for g in chain]
    n_images = len(imgs)
    A = imgs[0]
    v = [np.zeros_like(A) for _ in imgs]
    dt, alpha, nplus = 0.05, 0.1, 0
    E = [None] * n_images
    eA, eB = mp2(imgs[0]), mp2(imgs[-1])
    ci = None
    for it in range(iters):
        e_int, g_int = _image_gradients(imgs[1:-1])
        E = [eA] + list(e_int) + [eB]
        G = [None] + list(g_int) + [None]
        imax = int(np.argmax(E[1:-1])) + 1
        if it >= ci_after:
            ci = imax
        F = [np.zeros_like(A) for _ in imgs]
        worst = 0.0
        for i in range(1, n_images - 1):
            tp, tm = imgs[i + 1] - imgs[i], imgs[i] - imgs[i - 1]
            if E[i + 1] > E[i] > E[i - 1]:
                tau = tp
            elif E[i + 1] < E[i] < E[i - 1]:
                tau = tm
            else:
                dmax = max(abs(E[i + 1] - E[i]), abs(E[i - 1] - E[i]))
                dmin = min(abs(E[i + 1] - E[i]), abs(E[i - 1] - E[i]))
                tau = tp * dmax + tm * dmin if E[i + 1] > E[i - 1] else tp * dmin + tm * dmax
            tau = tau / np.linalg.norm(tau)
            g = G[i]
            gpar = np.sum(g * tau) * tau
            if i == ci:
                F[i] = -g + 2 * gpar
            else:
                fs = k_spring * (np.linalg.norm(tp) - np.linalg.norm(tm)) * tau
                F[i] = -(g - gpar) + fs
            worst = max(worst, float(np.max(np.abs(F[i] if i == ci else -(g - gpar)))))
        if log:
            log(f"  NEB {it:3d}: max force {worst:.2e}, barrier {(max(E) - eA) * 219474.63:.1f} cm-1, CI {ci}")
        if ci is not None and worst < fmax:
            break
        # FIRE
        P = sum(float(np.sum(F[i] * v[i])) for i in range(1, n_images - 1))
        fn = np.sqrt(sum(float(np.sum(F[i] ** 2)) for i in range(1, n_images - 1)))
        vn = np.sqrt(sum(float(np.sum(v[i] ** 2)) for i in range(1, n_images - 1)))
        for i in range(1, n_images - 1):
            v[i] = (1 - alpha) * v[i] + alpha * vn * F[i] / max(fn, 1e-12)
        if P > 0:
            nplus += 1
            if nplus > 5:
                dt, alpha = min(dt * 1.1, dt_max), alpha * 0.99
        else:
            nplus, dt, alpha = 0, dt * 0.5, 0.1
            v = [np.zeros_like(A) for _ in imgs]
        for i in range(1, n_images - 1):
            v[i] = v[i] + dt * F[i]
            step = dt * v[i]
            sn = np.max(np.abs(step))
            if sn > step_max:
                step *= step_max / sn
            imgs[i] = imgs[i] + step
    return imgs, E, ci, it, worst


HARTREE_EV = 27.211386245988


def neb_ase(chain, fmax_ev=0.05, steps=300, k=0.5, log=None):
    """Climbing-image NEB with ASE (improved tangents, FIRE). Interior images are evaluated in
    parallel (--workers) and handed to ASE as single-point results. Units: eV, Angstrom.
    Returns images (Angstrom), energies (Eh), climbing index, steps, final max force (Eh/Angstrom)."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.mep import NEB
    from ase.optimize import FIRE
    images = [Atoms("".join(SYM), positions=np.array(g)) for g in chain]
    e0, g0 = mp2(images[0].positions, True)
    e1, g1 = mp2(images[-1].positions, True)
    for img, e, g in ((images[0], e0, g0), (images[-1], e1, g1)):
        img.calc = SinglePointCalculator(img, energy=e * HARTREE_EV, forces=-np.asarray(g) * HARTREE_EV)

    class ParallelNEB(NEB):
        def get_forces(self):
            inner = self.images[1:-1]
            es, gs = _image_gradients([im.positions.copy() for im in inner])
            for im, e, g in zip(inner, es, gs):
                im.calc = SinglePointCalculator(im, energy=e * HARTREE_EV, forces=-np.asarray(g) * HARTREE_EV)
            return super().get_forces()
    band = ParallelNEB(images, k=k, climb=True, method="improvedtangent", allow_shared_calculator=True)
    opt = FIRE(band, logfile=None)
    nstep = 0
    for nstep, _ in enumerate(opt.irun(fmax=fmax_ev, steps=steps)):
        if log and nstep % 5 == 0:
            E = [im.get_potential_energy() / HARTREE_EV for im in images]
            log(f"  NEB {nstep:3d}: max force {np.max(np.abs(band.get_forces())) / HARTREE_EV:.2e} Eh/A, "
                f"barrier {(max(E) - E[0]) * 219474.63:.1f} cm-1")
    E = [im.get_potential_energy() / HARTREE_EV for im in images]
    ci = int(np.argmax(E[1:-1])) + 1
    fres = float(np.max(np.abs(band.get_forces()))) / HARTREE_EV
    return [im.positions.copy() for im in images], E, ci, nstep, fres


def relaxed_scan(X, a, b, nq, seam_d):
    """Constrained minimizations along q = r(C-Ha) - r(C-Hb) from the minimum X to q = 0."""
    d = [np.linalg.norm(X[i] - X[0]) for i in range(1, 5)]
    ra0, rb0 = d[a - 1], d[b - 1]
    q0, mid0 = ra0 - rb0, (ra0 + rb0) / 2

    def con(ra, rb):
        return f"$set\ndistance 1 {a + 1} {ra:.6f}\ndistance 1 {b + 1} {rb:.6f}\n"
    path, G = [X.copy()], X.copy()
    for k in range(1, nq + 1):
        f = k / nq
        q = q0 * (1 - f)
        mid = mid0 + (seam_d - mid0) * f
        G = optimize_constrained(G, con(mid + q / 2, mid - q / 2))
        path.append(G.copy())
    return path


def least_motion_neighbour(X, m):
    """The adjacent minimum reached by least motion. Every permutation of the four H, with and
    without a reflection (permutation-inversion), gives an equivalent minimum; keep those whose
    long-bond pair shares exactly one H with the reference (octahedral neighbours) and take the
    smallest aligned mass-weighted RMSD. A plain transposition of a short- and a long-bond H is
    not least-motion: it leads to a much higher saddle (~5100 cm^-1 at UMP2), which a NEB then
    converges to faithfully."""
    import itertools
    tun = _helpers()[1]
    d = [np.linalg.norm(X[i] - X[0]) for i in range(1, 5)]
    longA = {i + 1 for i in np.argsort(d)[2:]}
    best = None
    for p in itertools.permutations([1, 2, 3, 4]):
        perm = [0] + list(p)
        Yp = X[perm]
        dY = [np.linalg.norm(Yp[i] - Yp[0]) for i in range(1, 5)]
        if len(longA & {i + 1 for i in np.argsort(dY)[2:]}) != 1:
            continue
        for mirror in (1, -1):
            Z = tun.kabsch(X, Yp * np.array([1.0, 1.0, mirror]), m)
            rmsd = float(np.sqrt(np.sum(m[:, None] * (Z - X) ** 2) / m.sum()))
            if best is None or rmsd < best[3] - 1e-9:
                best = (Z, p, mirror, rmsd)
    return best


def stage_ts(out, mn, n_images=11):
    """Pseudorotation saddle and path: climbing-image NEB between the reference minimum and its
    least-motion neighbour (least_motion_neighbour), started from the straight line between the
    aligned minima. The climbing image converges to the saddle with no assumption about its
    symmetry. A geomeTRIC TS refinement from the climbing image is kept only if it stays close, has
    one imaginary frequency and lies above the minimum."""
    rph, tun = _helpers()
    m = masses_of()
    X = np.array(mn["xyz"])
    lines = []

    def log(msg):
        lines.append(msg)
        print(msg, flush=True)
    Y, perm, mirror, rmsd = least_motion_neighbour(X, m)
    log(f"  neighbour minimum: H permutation {perm}, reflection {mirror}, aligned mass-weighted RMSD {rmsd:.4f} A")
    chain = [X + (Y - X) * t for t in np.linspace(0, 1, n_images)]
    imgs, E, ci, nit, fres = neb_ase(chain, steps=NEB_STEPS, log=log)
    T = imgs[ci]
    refined = None
    try:
        R = optimize(T, transition=True)
        HR = hessian(R)
        wR = rph.projected_frequencies(HR, R, m)
        eR = mp2(R)
        if (int(np.sum(wR < -1.0)) == 1 and np.max(np.abs(tun.kabsch(T, R, m) - T)) < 0.15
                and eR > mn["e_ump2"] + 1e-6):
            refined = (R, HR, wR, eR)
        else:
            log("  TS refinement rejected (moved away, wrong curvature or fell to the minimum)")
    except Exception as exc:
        log(f"  TS refinement failed: {exc}")
    if refined is None:
        H = hessian(T)
        w = rph.projected_frequencies(H, T, m)
        eT, gT = mp2(T, True)
        src = "neb_climbing_image"
    else:
        T, H, w, eT = refined
        gT = mp2(T, True)[1]
        src = "geomeTRIC TS refinement from the climbing image"
    half = list(range(ci, -1, -1))
    return {"xyz": T.tolist(), "e_ump2": float(eT), "frequencies": w.tolist(), "n_imaginary": int(np.sum(w < -1.0)),
            "residual_gradient_rms": float(np.sqrt(np.mean(np.asarray(gT) ** 2))), "source": src,
            "neighbour": {"permutation": [int(x) for x in perm], "reflection": int(mirror), "rmsd_A": rmsd},
            "neb": {"iterations": int(nit), "final_max_force": float(fres),
                                              "energies": [float(e) for e in E], "climbing_index": int(ci),
                                              "images": [g.tolist() for g in imgs]},
            "half_path": {"path": [T.tolist()] + [imgs[i].tolist() for i in half[1:]],
                          "e_ump2": [float(eT)] + [float(E[i]) for i in half[1:]]},
            "hessian": H.tolist(), "log": lines}


def stage_irc(out, ts, mn, n_fine=24):
    """Reaction path, saddle -> minimum: the NEB half path, refined by inserting points along a
    cubic interpolation of the aligned images (energies recomputed at UMP2)."""
    from scipy.interpolate import CubicSpline
    tun = _helpers()[1]
    m = masses_of()
    hp = [np.array(p) for p in ts["half_path"]["path"]]
    for i in range(1, len(hp)):
        hp[i] = tun.kabsch(hp[i - 1], hp[i], m)
    s = tun.arc_length(hp, m)
    cs = CubicSpline(s, np.array(hp).reshape(len(hp), -1), axis=0)
    sf = np.linspace(0, s[-1], n_fine)
    path = [cs(x).reshape(5, 3) for x in sf]
    path[0], path[-1] = hp[0], hp[-1]
    E, _ = _image_gradients(path)
    E[0], E[-1] = ts["e_ump2"], ts["half_path"]["e_ump2"][-1]
    end = tun.kabsch(np.array(mn["xyz"]), path[-1], m)
    return {"path": [p.tolist() for p in path], "e_ump2": [float(e) for e in E],
            "type": "climbing-image NEB half path (cubic refinement)",
            "endpoint_deviation_A": float(np.max(np.abs(end - np.array(mn["xyz"]))))}


def stage_hess(out, irc, npts=12):
    path = irc["path"]
    idx = sorted(set(np.linspace(0, len(path) - 1, npts).round().astype(int).tolist()))
    return {"indices": idx, "hessians": [hessian(np.array(path[i])).tolist() for i in idx]}


def stage_ccsdt(out, irc, npts=6):
    path = irc["path"]
    idx = sorted(set(np.linspace(0, len(path) - 1, npts).round().astype(int).tolist()))
    return {"indices": idx, "e_ccsdt": [ccsdt(np.array(path[i])) for i in idx]}


def _e_task(args):
    X, charge, spin, basis = args
    global BASIS
    BASIS = basis
    return float(mp2(np.asarray(X), False, charge, spin))


def _rpsurface():
    import importlib.util
    f = Path(__file__).resolve().parents[1] / "python" / "riss_drucker" / "abinitio" / "rpsurface.py"
    spec = importlib.util.spec_from_file_location("_rd_rpsurface", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_plane2d(out, h=0.03, chunk=40):
    """UMP2 energies on the Cartesian reaction plane of the double well (one plane per isotopologue,
    since the plane lives in mass-weighted coordinates). Points: spacing h (amu^1/2 Angstrom), within
    0.3 of the path and below 5000 cm-1 in the harmonic estimate. Checkpointed every `chunk` points to
    ch4plus_plane2d.partial.json, so an interrupted run resumes where it stopped."""
    rp = _rpsurface()
    st = rp.load_stages(out)
    part = out / "ch4plus_plane2d.partial.json"
    done = json.loads(part.read_text()) if part.exists() else {}
    res = {"h": h, "method": "UMP2", "basis": BASIS}
    for iso in ("H", "D"):
        rs = rp.ReactionSurface(st, iso, use_grid=False)
        pl = rp.ReactionPlane(rs)
        pts, _ = pl.abinitio_points(h=h)
        # consistency points on the path itself (energies must match the reaction-path stage)
        idx = np.linspace(0, len(pl.curve) - 1, 5).round().astype(int)
        check_pts = pl.curve[idx]
        allp = np.r_[pts, check_pts]
        geoms = [g.tolist() for g in pl.geometry(allp)]
        prev = done.get(iso, {})
        E = list(prev.get("e_ump2", [])) if prev.get("n") == len(allp) and prev.get("h") == h else []
        if E:
            print(f"  plane {iso}: resuming at point {len(E)} of {len(allp)}", flush=True)
        t0, n0 = time.time(), len(E)
        for i in range(len(E), len(allp), chunk):
            tasks = [(G, 1, 1, BASIS) for G in geoms[i:i + chunk]]
            E += list(_pool().map(_e_task, tasks)) if WORKERS > 1 else [_e_task(x) for x in tasks]
            done[iso] = {"n": len(allp), "h": h, "e_ump2": E}
            part.write_text(json.dumps(done))
            rate = (time.time() - t0) / max(1, len(E) - n0)
            print(f"  plane {iso}: {len(E)}/{len(allp)} points ({(len(allp) - len(E)) * rate / 60:.0f} min left)",
                  flush=True)
        n = len(pts)
        sf = pl.fold(pl.s_signed[idx])
        e_path = (np.array(E[n:]) * rp.H2CM - rs.e_mp2_min)
        res[iso] = {"points": pts.tolist(), "e_ump2": E[:n], "n": n, "h": h,
                    "path_check": {"V_mp2_plane_cm": e_path.tolist(), "V_mp2_path_cm": rs._v_mp2(sf).tolist()}}
        dev = float(np.max(np.abs(e_path - rs._v_mp2(sf))))
        print(f"  plane {iso}: path consistency max |dE| = {dev:.1f} cm-1", flush=True)
    return res


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
    ap.add_argument("--stages", default="min,ts,irc,hess,ccsdt,neutral,plane2d")
    ap.add_argument("--plane-h", type=float, default=0.03, help="reaction-plane grid spacing (amu^1/2 Angstrom)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for finite-difference Hessians")
    ap.add_argument("--neb-steps", type=int, default=300, help="maximum NEB optimizer steps")
    ap.add_argument("--basis", default="cc-pvtz", help="basis set (cc-pvtz for the case study; small ones for tests)")
    a = ap.parse_args()
    global WORKERS, BASIS, NEB_STEPS
    NEB_STEPS = a.neb_steps
    WORKERS = max(1, a.workers)
    BASIS = a.basis
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyscf
    meta = {"program": f"pyscf {pyscf.__version__}", "basis": BASIS}
    mf = out / "ch4plus_min.json"
    if mf.exists() and json.loads(mf.read_text()).get("meta", {}).get("basis", BASIS) != BASIS:
        raise SystemExit(f"{mf} was computed with another basis; use a separate --out directory")

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
    if "plane2d" in st:
        cached("plane2d", stage_plane2d, a.plane_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
