#!/usr/bin/env python3
"""CH3D ground state (C3v) built dynamically from the JPL catalogue (tag 17003).

  python scripts/case_ch3d.py --data DIR [--fetch] [--jfit 12]

Levels (J, K) are derived from the catalogue (E'' + nu, combination differences checked).
They are catalogue (SPCAT-model) energies built on the measured lines and IR ground-state
combination differences of the cited works, not raw measurements.
Start: rigid symmetric top from the operator dictionary in C3v (2 terms). autobuild adds
quartic and sextic dictionary operators on evidence, with the contact-transformation
filter. Fit to J <= jfit, predict J > jfit (extrapolation check).
Data: B. J. Drouin et al., JQSRT 110, 2077 (2009); C. Bray et al. (2017); see d017003.cat.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

URL = "https://spec.jpl.nasa.gov/ftp/pub/catalog/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--jfit", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    for f, sub in (("c017003.cat", ""), ("d017003.cat", "doc/")):
        if a.fetch or not (data / f).exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_data.py")), "url", URL + sub + f,
                            "--out", str(data), "--citation", "JPL catalogue CH3D tag 17003"], check=True)
    from riss_drucker import algebra as A, autobuild as AB, build as B
    from riss_drucker.data.catalog import levels_from_cat, read_cat

    lines = read_cat(data / "c017003.cat")
    levels, rep = levels_from_cat(lines)
    jmax = max(q[0] for q in levels)
    spec = B.BuildSpec([B.RotSpace()], "C3v", [], J=list(range(jmax + 1)))
    q2 = A.enumerate_terms(spec, A.multidegrees(spec, rot=(2, 2))).terms
    spec0 = spec.with_terms(q2)
    bm0 = B.compile_build(spec0)
    names = bm0.param_names
    # initial guess: the terms are (Jx^2 + Jy^2) and Jz^2 in some order
    guess = {}
    for t in q2:
        n = t.split(":")[0].strip()
        guess[n] = 3.88 if "Jx" in t else 5.25 / 2      # sym(Jz Jz) + h.c. = 2 Jz^2
    # assignment: nearest model level of the same J and K-symmetry class at the initial guess
    # assignment by quantum number: K of each model level from <Jz^2> (exact for the rigid top)
    Kz = bm0.operator_blocks("Jz Jz")
    slots = {}
    for b, (w, V), Lz in zip(bm0.blocks, bm0.eig(guess), Kz):
        for i in range(V.shape[1]):
            k = int(round(np.sqrt(max(float(np.real(V[:, i] @ Lz @ V[:, i])), 0.0))))
            slots.setdefault((b.two_j // 2, k), []).append((b.irrep, i))
    data_all = []
    used = set()
    for (J, K), (E, sig) in sorted(levels.items(), key=lambda kv: (kv[0][0], abs(kv[0][1]), -kv[0][1])):
        cand = [c for c in slots.get((J, abs(K)), []) if (J, c) not in used]
        if not cand:
            raise RuntimeError(f"no model level for J={J} K={K}")
        used.add((J, cand[0]))
        data_all.append(AB.LevelDatum(J, cand[0][0], cand[0][1], E, max(sig, 1e-4 / np.sqrt(12)),
                                    label=f"J={J} K={K}"))  # E'' printed to 1e-4 cm-1
    keys = {(d.J, d.irrep, d.index) for d in data_all}
    if len(keys) != len(data_all):
        raise RuntimeError("ambiguous assignment")
    fitset = [d for d in data_all if d.J <= a.jfit]
    hold = [d for d in data_all if d.J > a.jfit]
    spec_fit = B.BuildSpec(spec.spaces, spec.group, q2, J=list(range(a.jfit + 1)))
    res = AB.autobuild(spec_fit, fitset, guess,
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, rot=(4, 6)))],
                       redundancy_degrees=A.multidegrees(spec, rot=(3, 5)), max_steps=12,
                       new_term_sigma=1e-2, prior_sigma=10.0)
    final = B.compile_build(B.BuildSpec(spec.spaces, spec.group, res.spec.terms, J=list(range(jmax + 1))))
    pred = {}
    for b, (w, _) in zip(final.blocks, final.eig(res.fit.theta)):
        for i, e in enumerate(w):
            pred[(b.two_j / 2, b.irrep, i)] = e
    hold_res = [d.energy - pred[(d.J, d.irrep, d.index)] for d in hold]
    fit_res = [l["residual"] for l in res.fit.levels]
    out = {
        "data": {"lines": rep["n_lines"], "levels": rep["n_levels"], "combination_difference_max_spread_cm":
                 rep["max_spread_cm"], "jmax": jmax, "jfit": a.jfit},
        "steps": [{"term": s.term, "accepted": s.accepted, "dlogZ": s.delta_log_evidence, "reason": s.reason}
                  for s in res.steps if s.term],
        "final_terms": res.spec.terms, "theta": res.fit.theta,
        "sigma": {k: v for k, v in res.fit.sigma.items()},
        "fit": {"n": len(fitset), "chi2_per_dof": res.fit.reduced_chi2, "rms_cm": float(np.sqrt(np.mean(np.square(fit_res))))},
        "extrapolation": {"n": len(hold), "rms_cm": float(np.sqrt(np.mean(np.square(hold_res)))) if hold else None,
                          "max_abs_cm": float(np.max(np.abs(hold_res))) if hold else None,
                          "worst": sorted([(d.label, float(r)) for d, r in zip(hold, hold_res)],
                                          key=lambda x: -abs(x[1]))[:6]},
        "certificates_passed": all(c["passed"] for c in res.certificates),
        "redundancy": {k: res.redundancy[k] for k in ("n_terms", "redundant", "identifiable_max")},
    }
    dest = Path(a.out or data)
    (dest / "ch3d_result.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
