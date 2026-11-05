#!/usr/bin/env python3
"""Train the symbolic language model on the case studies' accept/reject decisions.

  python scripts/train_symbolic_lm.py --ch3d DIR/ch3d_result.json [--write]

Sources:
* CH3D (real JPL data): every dictionary term autobuild accepted or rejected.
* Synthetic problems with known truth: asymmetric-top distortion (redundant terms present),
  a Fermi-resonance molecule, and a three-mode molecule with one Darling-Dennison resonance
  among detuned cubic couplings.
Evaluation: leave-one-source-out AUC (does a model trained on the other sources rank the held-out
source's accepted terms above its rejected ones?). --write stores the weights in
python/riss_drucker/_slm_weights.py (used by SymbolicLM.default()) only if their mean
leave-one-source-out AUC beats the physical prior; otherwise the prior stays the default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    return float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))


def synthetic_sources():
    from riss_drucker import algebra as A, autobuild as AB, build as B
    from riss_drucker.symbolic_lm import SymbolicLM
    rng = np.random.default_rng(11)
    recs = []

    def levels(bm, th, sig):
        return [AB.LevelDatum(b.two_j / 2, b.irrep, i, e + sig * rng.standard_normal(), sig, b.polyad)
                for b, (w, _) in zip(bm.blocks, bm.eig(th)) for i, e in enumerate(w)]
    # 1. asymmetric top, reduced quartic + sextic truth
    rs = B.BuildSpec([B.RotSpace()], "D2", [], J=list(range(9)))
    q = A.enumerate_terms(rs, A.multidegrees(rs, rot=(2, 6))).terms
    truth = B.compile_build(rs.with_terms(q[:3] + q[3:8] + q[9:12]))
    tt = dict(zip(truth.param_names, [1.5, 0.5, 1.4] + list(-3e-5 * (1 + rng.random(5))) + list(1e-8 * rng.standard_normal(3))))
    res = AB.autobuild(rs.with_terms(q[:3]), levels(truth, tt, 1e-4), dict(zip(truth.param_names[:3], [1.5, 0.5, 1.4])),
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, rot=(4, 6)))],
                       redundancy_degrees=A.multidegrees(rs, rot=(3, 5)), max_steps=10, new_term_sigma=1e-2)
    recs += SymbolicLM.records_from_model(rs, res.spec.terms, A.multidegrees(rs, rot=(4, 6)), source="rotor")
    # 2. Fermi resonance (omega1 ~ 2 omega2) with detuned alternatives
    modes = [B.Mode("1", "A", 2), B.Mode("2", "A", 1)]
    sp = [B.VibSpace(modes, polyads=[0, 1, 2, 3, 4])]
    tr = B.compile_build(B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2", "x22: n2*n2", "k: a1+ a2 a2 + h.c."]))
    data = levels(tr, {"w1": 3000.0, "w2": 1495.0, "x22": -4.0, "k": 30.0}, 0.01)
    res = AB.autobuild(B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2"]), data, {"w1": 3000.0, "w2": 1495.0},
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, vib=(3, 4)))], max_steps=4,
                       new_term_sigma=100.0)
    recs += SymbolicLM.records_from_model(B.BuildSpec(sp, "C1", []), res.spec.terms,
                                          A.multidegrees(B.BuildSpec(sp, "C1", []), vib=(3, 4)),
                                          {"1": 3000.0, "2": 1495.0}, source="fermi")
    # 3. three modes, Darling-Dennison between 1 and 3 (omega1 ~ omega3), mode 2 far away
    modes = [B.Mode("1", "A", 1), B.Mode("2", "A", 1), B.Mode("3", "A", 1)]
    sp = [B.VibSpace(modes, nmax=3)]
    tr = B.compile_build(B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2", "w3: n3", "x11: n1 n1", "dd: a1+ a1+ a3 a3 + h.c."]))
    th = {"w1": 3650.0, "w2": 1600.0, "w3": 3750.0, "x11": -40.0, "dd": -20.0}
    res = AB.autobuild(B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2", "w3: n3"]), levels(tr, th, 0.05),
                       {"w1": 3650.0, "w2": 1600.0, "w3": 3750.0},
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, vib=(4, 4)))], max_steps=4,
                       new_term_sigma=100.0, trial_top=20)
    recs += SymbolicLM.records_from_model(B.BuildSpec(sp, "C1", []), res.spec.terms,
                                          A.multidegrees(B.BuildSpec(sp, "C1", []), vib=(4, 4)),
                                          {"1": 3650.0, "2": 1600.0, "3": 3750.0}, source="darling_dennison")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch3d", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    from riss_drucker.symbolic_lm import FEATURES, PRIOR, SymbolicLM, term_features
    from riss_drucker import algebra as A, build as B
    ch = json.loads(Path(a.ch3d).read_text())
    cspec = B.BuildSpec([B.RotSpace()], "C3v", [], J=[0])
    recs = SymbolicLM.records_from_model(cspec, ch["final_terms"], A.multidegrees(cspec, rot=(4, 6)),
                                         source="ch3d_real")
    recs += synthetic_sources()
    sources = sorted({r[2] for r in recs})
    report = {"n_records": len(recs), "by_source": {s: {"n": sum(r[2] == s for r in recs),
                                                         "accepted": sum(r[1] for r in recs if r[2] == s)}
                                                     for s in sources}, "leave_one_source_out_auc": {}}
    for s in sources:
        train = [(r[0], r[1]) for r in recs if r[2] != s]
        test = [r for r in recs if r[2] == s]
        lm = SymbolicLM().fit(train)
        prior = SymbolicLM()
        report["leave_one_source_out_auc"][s] = {
            "trained": auc([lm.logit(r[0]) for r in test], [r[1] for r in test]),
            "prior_only": auc([prior.logit(r[0]) for r in test], [r[1] for r in test])}
    lm = SymbolicLM(trained_on=sources).fit([(r[0], r[1]) for r in recs])
    report["weights"] = lm.weights
    print(json.dumps(report, indent=1))
    tr = [v["trained"] for v in report["leave_one_source_out_auc"].values() if v["trained"] is not None]
    pr = [v["prior_only"] for v in report["leave_one_source_out_auc"].values() if v["prior_only"] is not None]
    report["mean_auc"] = {"trained": float(np.mean(tr)), "prior_only": float(np.mean(pr))}
    better = report["mean_auc"]["trained"] > report["mean_auc"]["prior_only"]
    print(json.dumps(report["mean_auc"]), "-> ship trained weights" if better else "-> keep the physical prior")
    if a.write and not better:
        print("not writing: trained weights do not beat the prior out of sample")
    if a.write and better:
        p = Path(__file__).resolve().parents[1] / "python" / "riss_drucker" / "_slm_weights.py"
        p.write_text('"""Weights of the symbolic language model, written by scripts/train_symbolic_lm.py."""\n'
                     f"WEIGHTS = {json.dumps({k: round(v, 6) for k, v in lm.weights.items()}, indent=1)}\n"
                     f"TRAINED_ON = {json.dumps(sources)}\n")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
