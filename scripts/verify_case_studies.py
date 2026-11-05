#!/usr/bin/env python3
"""Re-run the case studies and check every headline number against its recorded value.

  python scripts/verify_case_studies.py [--pdf CH4PLUS_PAPER.pdf] [--data DIR] [--figures DIR]

Always run (no external input; seconds to a minute):
  * methane cation six-well tunneling from symmetry (T1+T2, delta = 4|sigma|, CH2D2+ pattern)
  * methane cation from the ab initio surface: the committed UMP2/UCCSD(T) stages in
    data/abinitio/ch4plus are re-analysed (rotational constants, barrier, tunneling splittings,
    isotope ratio, ionization energy) and figures are regenerated
With --pdf (Woerner, Qian & Merkt, JCP 126, 144305 (2007)):
  * rovibronic fits of CH4+, 13CH4+, CD4+ (rms and constants vs the published fit)
  * ab initio effective Hamiltonian vs the measured lines, Bayesian update
With --data DIR (catalogue downloads, fetched with --fetch):
  * CH3D from the JPL catalogue (fit and extrapolation)
Exit status 0 only if every executed check passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = []


def check(name, value, expected, tol, note=""):
    ok = value is not None and abs(value - expected) <= tol
    RESULTS.append((name, value, expected, tol, ok, note))
    return ok


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def run_json(args):
    out = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    t = out.stdout
    return json.loads(t[t.index("{"):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--figures", default=str(ROOT / "docs" / "case_studies" / "figures"))
    a = ap.parse_args()
    expected = json.loads((ROOT / "docs" / "case_studies" / "expected_results.json").read_text())

    # 1. symmetry-only methane cation
    m = load_script("case_methane_cation").analyse()
    check("CH4+ wells: T1+T2 (geometric phase)", float(m["CH4+"]["well_representation"] == {"F1": 1, "F2": 1}), 1, 0)
    check("CH4+ splitting / |sigma|", m["CH4+"]["splitting_over_sigma"], 4.0, 1e-9)
    check("CH2D2+ J=0 pattern A1, B1+B2, A2 reproduced", float(m["CH2D2+"]["pattern_reproduced"]), 1, 0)

    # 2. ab initio surface -> effective Hamiltonian
    args = [str(ROOT / "scripts" / "case_methane_cation_abinitio.py"), "--abinitio", str(ROOT / "data" / "abinitio" / "ch4plus"),
            "--figures", a.figures]
    if a.pdf:
        args += ["--pdf", a.pdf]
    r = run_json(args)
    e = expected["methane_cation_abinitio"]
    for iso in ("CH4+", "CD4+"):
        for k in ("A", "B", "C"):
            check(f"{iso} {k}_e (UMP2/cc-pVTZ)", r[iso]["rotational_constants_equilibrium"][k], e[iso][k], 1e-3)
        check(f"{iso} barrier, UCCSD(T)-corrected", r[iso]["barrier_electronic_cm"], e[iso]["barrier"], 0.5)
        check(f"{iso} tunneling splitting delta (ab initio)", r[iso]["delta_ab_initio"], e[iso]["delta"],
              max(1e-3, 1e-3 * e[iso]["delta"]))
    check("isotope ratio delta(CH4+)/delta(CD4+)", r["isotope_ratio_delta"], e["isotope_ratio"], 1e-2 * e["isotope_ratio"])
    check("adiabatic IE (UCCSD(T) + ZPE)", r["ionization_energy"]["adiabatic_cm"], e["IE"], 0.5)
    if a.pdf:
        for iso in ("CH4+", "CD4+"):
            eh = r["effective_hamiltonian"][iso]
            check(f"{iso} ab initio spectrum rms (IE offset only)", eh["prediction_no_fit"]["rms_cm"],
                  e[iso]["rms_no_fit"], 0.02)
            check(f"{iso} rms after Bayesian update", eh["bayesian_update"]["rms_cm"], e[iso]["rms_updated"], 0.02)
        rv = load_script("case_methane_cation_rovibronic")
        tables = rv.read_tables(Path(a.pdf))
        model = rv.Model(5)
        ref = expected["methane_cation_rovibronic"]
        for iso in ("CH4+", "13CH4+", "CD4+"):
            f = min((rv.fit_species(iso, tables[iso], model, s) for s in (1.0, -1.0)), key=lambda x: x["rms_cm"])
            check(f"{iso} rovibronic fit rms", f["rms_cm"], ref[iso]["rms"], 0.01, f"paper {ref[iso]['paper_rms']}")
            check(f"{iso} delta (fit)", f["delta"], ref[iso]["delta"], 0.05)

    # 3. CH3D (needs the catalogue)
    if a.data:
        c = run_json([str(ROOT / "scripts" / "case_ch3d.py"), "--data", a.data, "--fetch"])
        check("CH3D fit chi2/dof", c["fit"]["chi2_per_dof"], expected["ch3d"]["chi2"], 0.05)
        check("CH3D extrapolation rms J 13-23", c["extrapolation"]["rms_cm"], expected["ch3d"]["extrap_rms"], 2e-4)

    w = max(len(x[0]) for x in RESULTS)
    print(f"{'check':{w}}  {'value':>14} {'expected':>14} {'tol':>8}  result")
    for name, v, ex, tol, ok, note in RESULTS:
        vs = "None" if v is None else f"{v:14.6g}"
        print(f"{name:{w}}  {vs:>14} {ex:14.6g} {tol:8.2g}  {'PASS' if ok else 'FAIL'} {note}")
    nfail = sum(not x[4] for x in RESULTS)
    print(f"\n{len(RESULTS) - nfail}/{len(RESULTS)} checks passed; figures in {a.figures}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
