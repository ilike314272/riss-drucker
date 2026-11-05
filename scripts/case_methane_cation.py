#!/usr/bin/env python3
"""Methane cation: Jahn-Teller tunneling between six C2v wells, built from symmetry.

  python scripts/case_methane_cation.py [--out DIR]

Source (open-access repository copy): M. Gruetter, H. J. Woerner, F. Merkt, "Rovibronic analysis
of the Jahn-Teller effect in CH2D2+ at low energies", J. Chem. Phys. 131, 024309 (2009);
ETH Research Collection, hdl:20.500.11850/182792. Numbers used: Table III (Delta = 123.6(5),
Delta' = 243.2(5), sigma = -1.7(4) cm^-1) and the CH4+ ground-state tunneling splitting of about
16 cm^-1 quoted there (Woerner et al., J. Chem. Phys. 126, 144305 (2007)).

* CH4+ (Td): six wells whose adiabatic electronic functions are vectors of the T2 carrier
  space. The signed permutation representation (geometric phase) decomposes as T1 + T2 by
  characters; the operator dictionary has exactly two well operators (energy, nearest-well
  tunneling), and sym() of one hop generates the whole signed tunneling matrix. delta = 4|sigma|.
* CH2D2+ (C2v subgroup keeping the D pair): wells split into orbits {1}, {6}, {2-5}; with the
  published Delta, Delta', sigma the J = 0 levels of the four-well isomer are A1 (Delta + 2 sigma),
  B1 + B2 (Delta), A2 (Delta - 2 sigma), as reported - labels included.
Rotational-tunneling structure (each well with its own rotor axes) is not built here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WELLS = [(0, 1, -1), (0, 1, 1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0)]
PUBLISHED = {"Delta": 123.6, "Delta_prime": 243.2, "sigma_CH2D2": -1.7, "delta_CH4": 16.0}


def hop_norm(ctx, P, Operators, text):
    M = Operators(ctx, 0).matrix(ctx.canon(P.parse_expr(text.split(":", 1)[1]))).toarray().real
    return float(np.max(np.abs(M)))


def analyse() -> dict:
    from riss_drucker import algebra as A, build as B, symmetry as S
    from riss_drucker.build import Context, Operators, Parser, runtime_spaces
    from riss_drucker.molecule import _mulliken_names
    td = S.get_group("Td")
    out = {}
    # CH4+
    spec = B.BuildSpec([B.WellSpace("t", "F2", WELLS)], "Td", [])
    w = B._Well(td, spec.spaces[0])
    rep = td.decompose([w.char(g, 0) for g in range(td.order)])
    dic = A.enumerate_terms(spec, [(("well",),)])
    terms = ["E: sym(w[1,1])", "sigma: sym(w[1,3] + w[3,1])"]
    bm = B.compile_build(spec.with_terms(terms))
    ctx = Context(td, runtime_spaces(spec, td))
    P = Parser(ctx)
    hn, en = hop_norm(ctx, P, Operators, terms[1]), hop_norm(ctx, P, Operators, terms[0])
    sig = -PUBLISHED["delta_CH4"] / 4
    lev = {b.irrep: float(e[0]) for b, (e, _) in zip(bm.blocks, bm.eig({"E": 0.0, "sigma": sig / hn}))}
    out["CH4+"] = {"well_representation": rep, "dictionary_operators": dic.by_degree[0]["count"],
                   "molien_certificate": dic.certificates[0]["passed"],
                   "certificates_passed": all(c["passed"] for c in bm.certificates),
                   "sigma_from_delta_16": sig, "levels": lev,
                   "splitting": abs(lev["F2"] - lev["F1"]), "splitting_over_sigma": abs(lev["F2"] - lev["F1"]) / abs(sig)}
    # CH2D2+
    sub = td.subgroup(lambda el: set(el.perm[:2]) == {0, 1}, "C2v(CH2D2)")
    _mulliken_names(sub, None)
    spec2 = B.BuildSpec([B.WellSpace("t", "F2", WELLS)], sub, [])
    w2 = B._Well(sub, spec2.spaces[0])
    orbits = sorted({tuple(sorted({w2.perm[g][k] + 1 for g in range(sub.order)})) for k in range(6)})
    dic2 = A.enumerate_terms(spec2, [(("well",),)])
    t2 = ["E1: w[1,1]", "E6: w[2,2]", "D: sym(w[3,3])", "s35: sym(w[3,5] + w[5,3])", "s36: sym(w[3,6] + w[6,3])"]
    bm2 = B.compile_build(spec2.with_terms(t2))
    ctx2 = Context(sub, runtime_spaces(spec2, sub))
    P2 = Parser(ctx2)
    n = {t.split(":")[0]: hop_norm(ctx2, P2, Operators, t) for t in t2}
    s = PUBLISHED["sigma_CH2D2"]
    th = {"E1": 0.0, "E6": PUBLISHED["Delta_prime"], "D": PUBLISHED["Delta"] / n["D"], "s35": s / n["s35"],
          "s36": s / n["s36"]}
    lev2 = sorted((float(e), b.irrep) for b, (es, _) in zip(bm2.blocks, bm2.eig(th)) for e in es)
    four = [x for x in lev2 if 100 < x[0] < 200]
    expected = {"A1": PUBLISHED["Delta"] + 2 * s, "B": PUBLISHED["Delta"], "A2": PUBLISHED["Delta"] - 2 * s}
    out["CH2D2+"] = {"well_orbits": orbits, "dictionary_operators": dic2.by_degree[0]["count"],
                     "certificates_passed": all(c["passed"] for c in bm2.certificates),
                     "levels_J0": lev2, "four_well_isomer": four, "published_pattern": expected,
                     "pattern_reproduced": (abs(four[0][0] - expected["A1"]) < 1e-9 and four[0][1] == "A1"
                                            and four[-1][1] == "A2" and abs(four[-1][0] - expected["A2"]) < 1e-9
                                            and {four[1][1], four[2][1]} == {"B1", "B2"}),
                     "ratio_sigma_CH4_over_CH2D2": abs(sig / s)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = analyse()
    txt = json.dumps(out, indent=1, default=float)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
