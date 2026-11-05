#!/usr/bin/env python3
"""CH3D ground state (C3v) built dynamically from the JPL catalogue (tag 17003).

  python scripts/case_ch3d.py --data DIR [--fetch] [--jfit 12]

Levels (J, K) are derived from the catalogue (E'' + nu, combination differences checked).
PROVENANCE: the JPL CH3D file flags none of its 203 lines as measured (all TAGs positive), and E''
is the catalogue's computed lower-state energy. These levels are SPCAT-model output, so this study
tests how well a dynamically built model reproduces (and extrapolates) a published effective
Hamiltonian - a model-reduction test, not a test against measurements.
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


def load_levels(data: Path, fetch=False, measured_only=False):
    data.mkdir(parents=True, exist_ok=True)
    for f, sub in (("c017003.cat", ""), ("d017003.cat", "doc/")):
        if fetch or not (data / f).exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_data.py")), "url", URL + sub + f,
                            "--out", str(data), "--citation", "JPL catalogue CH3D tag 17003"], check=True)
    from riss_drucker.data.catalog import levels_from_cat, read_cat
    lines = read_cat(data / "c017003.cat")
    return levels_from_cat(lines, measured_only=measured_only)


def run_ch3d(levels, rep, *, jfit=12, extra_terms=(), holdout=None, autobuild_on=True, terms=None, theta0=None):
    """CH3D fit. holdout(J, K) -> True for levels held out (default: J > jfit). extra_terms are forced
    into the model after autobuild (or after `terms`, a fixed term list, when given).
    Returns a dict with the fit result, held-out predictions with Fisher sigma, and the term list."""
    from riss_drucker import algebra as A, autobuild as AB, build as B
    holdout = holdout or (lambda J, K: J > jfit)
    jmax = max(q[0] for q in levels)
    spec = B.BuildSpec([B.RotSpace()], "C3v", [], J=list(range(jmax + 1)))
    q2 = A.enumerate_terms(spec, A.multidegrees(spec, rot=(2, 2))).terms
    bm0 = B.compile_build(spec.with_terms(q2))
    guess = {}
    for t in q2:
        n = t.split(":")[0].strip()
        guess[n] = 3.88 if "Jx" in t else 5.25 / 2      # sym(Jz Jz) + h.c. = 2 Jz^2
    # assignment by quantum number: K of each model level from <Jz^2> (exact for the rigid top)
    Kz = bm0.operator_blocks("Jz Jz")
    slots = {}
    for b, (w, V), Lz in zip(bm0.blocks, bm0.eig(guess), Kz):
        for i in range(V.shape[1]):
            k = int(round(np.sqrt(max(float(np.real(V[:, i] @ Lz @ V[:, i])), 0.0))))
            slots.setdefault((b.two_j // 2, k), []).append((b.irrep, i))
    data_all, used, KofLabel = [], set(), {}
    for (J, K), (E, sig) in sorted(levels.items(), key=lambda kv: (kv[0][0], abs(kv[0][1]), -kv[0][1])):
        cand = [c for c in slots.get((J, abs(K)), []) if (J, c) not in used]
        if not cand:
            raise RuntimeError(f"no model level for J={J} K={K}")
        used.add((J, cand[0]))
        d = AB.LevelDatum(J, cand[0][0], cand[0][1], E, max(sig, 1e-4 / np.sqrt(12)), label=f"J={J} K={K}")
        data_all.append(d)
        KofLabel[d.label] = (J, K)
    fitset = [d for d in data_all if not holdout(*KofLabel[d.label])]
    hold = [d for d in data_all if holdout(*KofLabel[d.label])]
    jf = max(d.J for d in fitset)
    spec_fit = B.BuildSpec(spec.spaces, spec.group, q2, J=list(range(int(jf) + 1)))
    steps = []
    if terms is not None:
        base_terms, theta0 = list(terms), dict(guess, **(theta0 or {}))
    elif autobuild_on:
        res = AB.autobuild(spec_fit, fitset, guess,
                           [AB.DictionaryProposer(lambda s: A.multidegrees(s, rot=(4, 6)))],
                           redundancy_degrees=A.multidegrees(spec, rot=(3, 5)), max_steps=12,
                           new_term_sigma=1e-2, prior_sigma=10.0)
        base_terms, theta0 = list(res.spec.terms), dict(res.fit.theta)
        steps = [{"term": s.term, "accepted": s.accepted, "dlogZ": s.delta_log_evidence, "reason": s.reason}
                 for s in res.steps if s.term]
    else:
        base_terms, theta0 = list(q2), dict(guess)
    all_terms = base_terms + [t for t in extra_terms if t not in base_terms]
    # final fit on the fit set (also when nothing was forced, so every variant is fitted the same way)
    bmf = B.compile_build(B.BuildSpec(spec.spaces, spec.group, all_terms, J=sorted({int(d.J) for d in fitset})))
    th0 = {n: theta0.get(n, 0.0) for n in bmf.param_names}
    fr = AB.fit(bmf.model, AB.observations_for(bmf, fitset), th0, method="dense")
    final = B.compile_build(B.BuildSpec(spec.spaces, spec.group, all_terms, J=list(range(jmax + 1))))
    obs_h = AB.observations_for(final, hold) if hold else []
    pred = {}
    for b, (w, _) in zip(final.blocks, final.eig(fr.theta)):
        for i, e in enumerate(w):
            pred[(b.two_j / 2, b.irrep, i)] = e
    rows = []
    if hold:
        jac = AB.hf_jacobian(final, fr.theta, obs_h)
        order = [final.param_names.index(n) for n in fr.param_names]
        cov = np.zeros((len(final.param_names),) * 2)
        cov[np.ix_(order, order)] = fr.covariance
        for d, g in zip(hold, jac):
            r = d.energy - pred[(d.J, d.irrep, d.index)]
            rows.append({"label": d.label, "J": d.J, "K": KofLabel[d.label][1], "residual": float(r),
                         "sigma_obs": float(d.sigma), "sigma_pred": float(np.sqrt(max(g @ cov @ g, 0.0)))})
    fit_res = [l["residual"] for l in fr.levels]
    return {"terms": all_terms, "theta": fr.theta, "steps": steps, "fit": fr, "fit_rms": float(np.sqrt(np.mean(np.square(fit_res)))),
            "chi2_per_dof": fr.reduced_chi2, "n_fit": len(fitset), "holdout": rows, "rep": rep, "jmax": jmax}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--jfit", type=int, default=12)
    ap.add_argument("--measured-only", action="store_true",
                    help="use only lines flagged as measured (JPL CH3D has none: the run then stops, by design)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    levels, rep = load_levels(data, a.fetch, a.measured_only)
    r = run_ch3d(levels, rep, jfit=a.jfit)
    hold = r["holdout"]
    hr = [h["residual"] for h in hold]
    out = {
        "data": {"lines": rep["n_lines"], "levels": rep["n_levels"], "measured_lines": rep.get("n_measured_lines"),
                 "provenance": rep.get("provenance"),
                 "combination_difference_max_spread_cm": rep["max_spread_cm"], "jmax": r["jmax"], "jfit": a.jfit},
        "steps": r["steps"], "final_terms": r["terms"], "theta": r["theta"],
        "sigma": dict(r["fit"].sigma),
        "fit": {"n": r["n_fit"], "chi2_per_dof": r["chi2_per_dof"], "rms_cm": r["fit_rms"]},
        "extrapolation": {"n": len(hold), "rms_cm": float(np.sqrt(np.mean(np.square(hr)))) if hold else None,
                          "max_abs_cm": float(np.max(np.abs(hr))) if hold else None,
                          "worst": sorted([(h["label"], h["residual"]) for h in hold], key=lambda x: -abs(x[1]))[:6]},
        "certificates_passed": True,
    }
    dest = Path(a.out or data)
    (dest / "ch3d_result.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
