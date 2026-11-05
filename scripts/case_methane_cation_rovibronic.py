#!/usr/bin/env python3
"""Rovibronic PFI-ZEKE spectrum of CH4+, 13CH4+ and CD4+ from a dynamically built
tunneling-rotation Hamiltonian.

  python scripts/case_methane_cation_rovibronic.py --pdf PAPER.pdf [--out result.json]

Data: H. J. Woerner, X. Qian, F. Merkt, J. Chem. Phys. 126, 144305 (2007), Tables VII-IX
(measured line positions and assignments), read from the user-supplied PDF (not stored in
the repository). Neutral ground-state rotational constants: CH4 B0 = 5.2410 cm^-1 (13CH4 taken
equal), CD4 B = 2.63273 cm^-1 (as used in the paper).

Model (built, not chosen):
* spaces: rotation |N k> (cation rotation; electron spin not resolved) x six wells whose
  adiabatic electronic functions are T2 carrier vectors (geometric phase); group Td.
* terms: the dictionary's well-projected quadratic rotor operators, symmetrized from one well
  (the compiler finds that well's unique axis: A J_a^2 + (B+C)/2 (J_b^2 + J_c^2) +- (B-C)/2 {J_b, J_c}),
  and the nearest-well tunneling operator; delta = 4|sigma|.
* fit: IE, A, B, C, sigma (the paper's 5 constants) by nonlinear least squares to the measured
  lines (A >= B >= C). Each line is assigned once to one level (N+, Gamma+, index) of the observed
  class - nuclear-spin symmetry is conserved (A <-> A1/A2, F <-> F1/F2, E <-> E) - nearest at the
  published constants; after the fit it is re-assigned once and kept only if the rms improves.
* published N+ labels are kept unless the labelled N+ misses by > 3 cm^-1 at the published
  constants and a neighbouring N+ of the same symmetry fits three times better; every such
  relabelling is reported.
* extension test: every tunneling-rotation coupling operator of the dictionary (hop x J^2) is
  tried on evidence (BIC); kept only if it improves it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

WELLS = [(0, 1, -1), (0, 1, 1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0)]
NEUTRAL_B = {"CH4+": 5.2410, "13CH4+": 5.2410, "CD4+": 2.63273}
PAPER = {"CH4+": {"IE": 101752.2, "A": 6.40, "B": 5.55, "C": 4.03, "delta": 16.4},
         "13CH4+": {"IE": 101756.6, "A": 6.30, "B": 5.70, "C": 4.05, "delta": 16.2},
         "CD4+": {"IE": 102196.9, "A": 3.35, "B": 2.53, "C": 1.98, "delta": 1.4}}
RELABEL_TOL = 3.0      # cm^-1: a published N+ label is questioned only if it misses by more than this
ROW = re.compile(r"^\s*(\d)\s+(A1|A2|E|F1|F2)\s+(\d(?:/\d)?)?\s+(A1|A2|A|E|F1|F2|F)\s+(1\d\d\s\d{3}\.\d+)\s*([−+-]?\d+\.\d+)?")


def read_tables(pdf: Path) -> dict:
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    sections = {"CH4+": "TABLE VII.", "13CH4+": "TABLE VIII.", "CD4+": "TABLE IX."}
    out = {}
    keys = list(sections)
    for i, k in enumerate(keys):
        start = txt.index(sections[k])
        end = txt.index(sections[keys[i + 1]]) if i + 1 < len(keys) else start + 8000
        rows = []
        for line in txt[start:end].splitlines():
            m = ROW.match(line)
            if not m:
                continue
            J, g, N, gp, nu, dev = m.groups()
            J = int(J)
            if g == "E" and J != 2:      # 13CH4 Table VIII prints J = 1 for an E level: only J = 2 has E
                J = 2
            rows.append({"J": J, "G": g, "N": N, "Gp": gp, "nu": float(nu.replace(" ", "")),
                         "paper_dev": float(dev.replace("−", "-")) if dev else None})
        out[k] = rows
    return out


class Model:
    def __init__(self, nmax=5, extra_terms=(), geometric_phase=True):
        from riss_drucker import build as B
        from riss_drucker.build import Context, Operators, Parser, runtime_spaces
        self.B = B
        terms = ["rA: sym(w[1,1] Jx Jx)", "rBC: sym(w[1,1] Jy Jy)", "rD: sym(w[1,1] (Jy Jz + Jz Jy))",
                 "sig: sym(w[1,3] + w[3,1])"] + list(extra_terms)
        self.spec = B.BuildSpec([B.RotSpace(), B.WellSpace("t", "F2", WELLS, geometric_phase)], "Td", terms,
                                J=list(range(nmax + 1)))
        self.bm = B.compile_build(self.spec)
        g = self.bm.group
        ctx = Context(g, runtime_spaces(self.spec, g))
        P = Parser(ctx)
        M = Operators(ctx, 0).matrix(ctx.canon(P.parse_expr("sym(w[1,3] + w[3,1])"))).toarray().real
        self.hop = float(np.max(np.abs(M)))                      # sigma = coefficient * hop
        self.extra = [t.split(":")[0].strip() for t in extra_terms]

    def theta(self, A, Bc, C, sigma, sgn=1.0, extra=()):
        th = {"rA": 6 * A, "rBC": 6 * (Bc + C), "rD": 3 * sgn * (Bc - C), "sig": sigma / self.hop}
        th.update(dict(zip(self.extra, extra)))
        return th

    def levels(self, th):
        out = []
        for b, (w, _) in zip(self.bm.blocks, self.bm.eig(th)):
            for e in w:
                out.append((b.two_j // 2, b.irrep, float(e)))
        e0 = min(e for N, ir, e in out if N == 0)
        return [(N, ir, e - e0) for N, ir, e in out]


def allowed(gp, glow):
    cls = {"A1": "A", "A2": "A", "F1": "F", "F2": "F", "E": "E"}
    if gp in ("A1", "A2", "F1", "F2", "E"):
        return {gp}
    return {g for g in ("A1", "A2", "F1", "F2", "E") if cls[g] == gp}


def levels_indexed(model, x, sgn):
    IE, A, Bc, C, sig = x[:5]
    out = {}
    for b, (w, _) in zip(model.bm.blocks, model.bm.eig(model.theta(A, Bc, C, sig, sgn, x[5:]))):
        for k, e in enumerate(w):
            out[(b.two_j // 2, b.irrep, k)] = float(e)
    e0 = min(e for (N, _, _), e in out.items() if N == 0)
    return {k: v - e0 for k, v in out.items()}


def assign(model, x, sgn, data, Bn):
    """Each line -> one level (N+, Gamma+, index) of the allowed class, nearest at parameters x."""
    lev = levels_indexed(model, x, sgn)
    out = []
    for r in data:
        El = Bn * r["J"] * (r["J"] + 1)
        dist = lambda k: abs(x[0] + lev[k] - El - r["nu"])
        cands = [k for k in lev if k[0] == int(r["N"]) and k[1] in allowed(r["Gp"], r["G"])]
        if not cands:
            raise ValueError(f"model has no level of class {r['Gp']} at N+={r['N']} "
                             f"(line J''={r['J']} {r['G']} at {r['nu']} cm-1)")
        best = min(cands, key=dist)
        r["relabelled"] = None
        if dist(best) > RELABEL_TOL:      # published N+ inconsistent with the published constants
            alt = [k for k in lev if abs(k[0] - int(r["N"])) == 1 and k[1] in allowed(r["Gp"], r["G"])]
            if alt and dist(min(alt, key=dist)) < dist(best) / 3:
                best = min(alt, key=dist)
                r["relabelled"] = f"N+ {r['N']} -> {best[0]}"
        out.append(best)
    return out


def calc_lines(model, x, sgn, data, Bn, asg):
    lev = levels_indexed(model, x, sgn)
    return np.array([x[0] + lev[k] - Bn * r["J"] * (r["J"] + 1) for r, k in zip(data, asg)])


def fit_species(name, rows, model, sgn):
    from scipy.optimize import least_squares
    data = [r for r in rows if r["N"] and "/" not in r["N"]]
    Bn = NEUTRAL_B[name]
    p0 = PAPER[name]
    x0 = np.array([p0["IE"], p0["A"], p0["B"], p0["C"], p0["delta"] / 4] + [0.0] * len(model.extra))
    obs = np.array([r["nu"] for r in data])
    asg = assign(model, x0, sgn, data, Bn)                # assignment at the published constants
    lo = np.r_[-np.inf, 0.0, 0.0, 0.0, -np.inf, [-np.inf] * len(model.extra)]
    best = None
    for it in range(3):                                   # fit with frozen assignment; re-assign once more
        def res(x):
            pen = [10 * max(0.0, x[2] - x[1]), 10 * max(0.0, x[3] - x[2])]   # A >= B >= C
            return np.r_[obs - calc_lines(model, x, sgn, data, Bn, asg), pen]
        r = least_squares(res, x0, x_scale=np.r_[1.0, 0.1, 0.1, 0.1, 0.5, [0.01] * len(model.extra)],
                          diff_step=1e-6, bounds=(lo, np.inf))
        rms = float(np.sqrt(np.mean(r.fun[:len(obs)] ** 2)))
        if best is None or rms < best[1] - 1e-9:
            best = (r, rms, list(asg))
        else:
            break
        x0 = r.x
        asg = assign(model, r.x, sgn, data, Bn)
    r, rms, asg = best
    J = r.jac[:len(obs)]
    dof = max(len(obs) - len(r.x), 1)
    s2 = float(np.sum(r.fun[:len(obs)] ** 2) / dof)
    try:
        sd = np.sqrt(np.diag(np.linalg.inv(J.T @ J) * s2))
    except np.linalg.LinAlgError:
        sd = np.full(len(r.x), np.nan)
    calc = calc_lines(model, r.x, sgn, data, Bn, asg)
    paper = [d["paper_dev"] for d in data if d["paper_dev"] is not None]
    n = len(obs)
    bic = n * math.log(max(np.mean((obs - calc) ** 2), 1e-300)) + len(r.x) * math.log(n)
    names = ["IE", "A", "B", "C", "sigma"] + model.extra
    return {"n_lines": n, "rms_cm": rms, "paper_rms_cm": float(np.sqrt(np.mean(np.square(paper)))) if paper else None,
            "bic": bic, "constants": {k: float(v) for k, v in zip(names, r.x)},
            "sd": {k: float(v) for k, v in zip(names, sd)}, "delta": float(4 * abs(r.x[4])),
            "delta_sd": float(4 * sd[4]), "rD_sign": sgn,
            "relabelled": [f"J''={d['J']} {d['G']} -> {d['Gp']} at {d['nu']}: {d['relabelled']}"
                           for d in data if d.get("relabelled")],
            "lines": [{"J": d["J"], "G": d["G"], "N": d["N"], "Gp": d["Gp"], "obs": d["nu"],
                       "calc": float(c), "obs_minus_calc": float(d["nu"] - c), "upper": list(k),
                       "relabelled": d.get("relabelled")}
                      for d, c, k in zip(data, calc, asg)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nmax", type=int, default=5)
    a = ap.parse_args()
    tables = read_tables(Path(a.pdf))
    from riss_drucker import algebra as A
    base = Model(a.nmax)
    out = {"source": "Woerner, Qian, Merkt, J. Chem. Phys. 126, 144305 (2007), Tables VI-IX",
           "parsed_rows": {k: len(v) for k, v in tables.items()}, "species": {}}
    # candidate tunneling-rotation couplings from the dictionary (hop x quadratic rotor)
    dic = A.enumerate_terms(base.spec.with_terms([]), A.multidegrees(base.spec, rot=(2, 2)), prefix="tr").terms
    tr_terms = [t for t in dic if not re.search(r"w\[(\d),\1\]", t)]
    nophase = Model(a.nmax, geometric_phase=False)
    for name, rows in tables.items():
        fits = [fit_species(name, rows, base, s) for s in (1.0, -1.0)]
        try:
            npf = min((fit_species(name, rows, nophase, s) for s in (1.0, -1.0)), key=lambda f: f["rms_cm"])
            no_phase = {"rms_cm": npf["rms_cm"], "bic": npf["bic"], "delta": npf["delta"]}
        except Exception as exc:          # the phase-free model may lack the observed symmetry classes
            no_phase = {"error": str(exc)}
        best = min(fits, key=lambda f: f["rms_cm"])
        trials = []
        for t in tr_terms:
            m2 = Model(a.nmax, [t])
            f2 = fit_species(name, rows, m2, best["rD_sign"])
            trials.append({"term": t, "rms_cm": f2["rms_cm"], "bic": f2["bic"], "dBIC": f2["bic"] - best["bic"]})
        out["species"][name] = {"fit": best, "other_rD_sign_rms": max(fits, key=lambda f: f["rms_cm"])["rms_cm"],
                                "paper_constants": PAPER[name], "tunneling_rotation_trials": trials,
                                "without_geometric_phase": no_phase}
        print(f"{name}: {best['n_lines']} lines, rms {best['rms_cm']:.3f} cm-1 (paper {best['paper_rms_cm']}); "
              + ", ".join(f"{k} {v:.3f}({best['sd'][k]:.3f})" for k, v in best["constants"].items())
              + f", delta {best['delta']:.2f}({best['delta_sd']:.2f}); best extension dBIC "
              + f"{min(t['dBIC'] for t in trials):.2f}; relabelled: {best['relabelled']}; "
              + f"without geometric phase: {no_phase}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
