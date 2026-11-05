#!/usr/bin/env python3
"""Re-run the case studies and check every headline number against its recorded value.

This is a REPRODUCIBILITY check: expected_results.json holds values that this same code produced
earlier, so a pass shows the pipeline still gives the same numbers, not that they are correct.
Correctness evidence is in the case studies (held-out predictions, coverage, baselines) and in
scripts/review_followups.py.

  python scripts/verify_case_studies.py [--pdf CH4PLUS_PAPER.pdf] [--data DIR] [--figures DIR]

Always run (no external input; seconds to a minute):
  * methane cation six-well tunneling from symmetry (T1+T2, delta = 4|sigma|, CH2D2+ pattern)
  * methane cation from the ab initio surface: every committed UMP2/UCCSD(T) surface listed in
    expected_results.json (data/abinitio/ch4plus_<basis>) is re-analysed (rotational constants,
    barrier, tunneling splittings, isotope ratio, ionization energy) and its figures regenerated
    in <figures>/methane_cation_<basis>/
With --pdf (Woerner, Qian & Merkt, JCP 126, 144305 (2007)):
  * rovibronic fits of CH4+, 13CH4+, CD4+ (rms and constants vs the published fit)
  * ab initio effective Hamiltonian vs the measured lines, Bayesian update
With --firstprinciples (about two minutes per surface):
  * first-principles tunneling (Cartesian reaction plane), averaged rotor, frame rotation and, with --pdf,
    the validation of the first-principles levels against the measured lines
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
    ap.add_argument("--firstprinciples", action="store_true",
                    help="also recompute the first-principles levels (reaction-plane tunneling; ~2 min per surface)")
    a = ap.parse_args()
    expected = json.loads((ROOT / "docs" / "case_studies" / "expected_results.json").read_text())

    # 1. symmetry-only methane cation
    m = load_script("case_methane_cation").analyse()
    check("CH4+ wells: T1+T2 (geometric phase)", float(m["CH4+"]["well_representation"] == {"F1": 1, "F2": 1}), 1, 0)
    check("CH4+ splitting / |sigma|", m["CH4+"]["splitting_over_sigma"], 4.0, 1e-9)
    check("CH2D2+ J=0 pattern A1, B1+B2, A2 reproduced", float(m["CH2D2+"]["pattern_reproduced"]), 1, 0)

    # 2. ab initio surfaces -> effective Hamiltonian (every committed surface listed in expected_results)
    for label, e in expected["methane_cation_abinitio"].items():
        d = ROOT / e["data"]
        if not d.exists():
            print(f"skip {label}: {e['data']} not present")
            continue
        fig_dir = Path(a.figures) / f"methane_cation_{label}"
        args = [str(ROOT / "scripts" / "case_methane_cation_abinitio.py"), "--abinitio", str(d), "--figures", str(fig_dir)]
        if a.pdf:
            args += ["--pdf", a.pdf]
        r = run_json(args)
        t = f"[{label}] "
        for iso in ("CH4+", "CD4+"):
            for k in ("A", "B", "C"):
                check(f"{t}{iso} {k}_e", r[iso]["rotational_constants_equilibrium"][k], e[iso][k], 1e-3)
            check(f"{t}{iso} barrier, UCCSD(T)-corrected", r[iso]["barrier_electronic_cm"], e[iso]["barrier"], 0.5)
            check(f"{t}{iso} tunneling splitting delta", r[iso]["delta_ab_initio"], e[iso]["delta"],
                  max(1e-3, 1e-3 * e[iso]["delta"]))
        check(f"{t}isotope ratio delta(CH4+)/delta(CD4+)", r["isotope_ratio_delta"], e["isotope_ratio"],
              1e-2 * e["isotope_ratio"])
        check(f"{t}adiabatic IE (UCCSD(T) + ZPE)", r["ionization_energy"]["adiabatic_cm"], e["IE"], 0.5)
        if a.pdf:
            for iso in ("CH4+", "CD4+"):
                eh = r["effective_hamiltonian"][iso]
                check(f"{t}{iso} ab initio spectrum rms (IE offset only)", eh["prediction_no_fit"]["rms_cm"],
                      e[iso]["rms_no_fit"], 0.02)
                check(f"{t}{iso} rms after Bayesian update", eh["bayesian_update"]["rms_cm"], e[iso]["rms_updated"], 0.02)
                check(f"{t}{iso} delta after Bayesian update", eh["bayesian_update"]["delta"], e[iso]["delta_updated"], 0.05)
    if a.pdf:
        rv = load_script("case_methane_cation_rovibronic")
        tables = rv.read_tables(Path(a.pdf))
        model = rv.Model(5)
        ref = expected["methane_cation_rovibronic"]
        for iso in ("CH4+", "13CH4+", "CD4+"):
            f = min((rv.fit_species(iso, tables[iso], model, s) for s in (1.0, -1.0)), key=lambda x: x["rms_cm"])
            check(f"{iso} rovibronic fit rms", f["rms_cm"], ref[iso]["rms"], 0.01, f"paper {ref[iso]['paper_rms']}")
            check(f"{iso} delta (fit)", f["delta"], ref[iso]["delta"], 0.05)

    # 3. first principles (optional: slower)
    if a.firstprinciples:
        import tempfile
        for label, e in expected.get("ch4plus_firstprinciples", {}).items():
            d = ROOT / e["data"]
            if not d.exists():
                continue
            out = Path(tempfile.mkdtemp()) / "fp.json"
            args = [str(ROOT / "scripts" / "firstprinciples_ch4plus.py"), "--abinitio", str(d), "--out", str(out)]
            if a.pdf:
                args += ["--pdf", a.pdf]
            p = subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=ROOT)
            if p.returncode:
                raise RuntimeError(p.stderr[-2000:])
            r = json.loads(out.read_text())
            for iso in ("CH4+", "CD4+"):
                x, ex = r["isotopologues"][iso], e[iso]
                t = f"[fp {label}] {iso} "
                check(t + "delta (reaction plane)", x["tunneling"]["delta"], ex["delta"], 2e-3 * ex["delta"])
                check(t + "A vibrationally averaged", x["tunneling"]["rotational_constants_averaged"][0], ex["A_avg"], 1e-3)
                check(t + "frame rotation / deg", x["tunneling"]["frame_rotation"]["angle_deg"], ex["frame_rotation_deg"], 1e-3)
                if a.pdf:
                    check(t + "rms vs measured lines (IE shift only)", x["validation"]["rms_IE_shifted_cm"],
                          ex["rms_IE_shifted"], 0.02)

    # 4. CH3D (needs the catalogue)
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
