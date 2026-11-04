#!/usr/bin/env python3
"""Open-shell validation / gold entry: O2 X 3Sigma-g (v=0) from CDMS (Drouin et
al. 2010; tag 032508), reproduced with the open_rotor model in its linear limit.

  python scripts/gold_cdms_o2.py --data DIR [--fetch] [--nmax 12] [--out gold/cdms-o2-v0]

Mapping (SPFIT linear-molecule parameters, MHz): B -> B = C (A large, K = 0 states),
-D -> DeltaN, 4 lambda -> D_ss = 2 lambda, 4 lambda_D -> D_ss_N = 2 lambda_D,
gamma -> eps (isotropic), gamma_D -> DeltaN_S. Sextic H, lambda_H, gamma_H are
not in the model (listed as unmapped); they limit agreement at high N.
Levels are matched by (N, J) assigned from <N^2> (riss_drucker.open_shell).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = "https://cdms.astro.uni-koeln.de/classic/predictions/daten/O2/3Sigma/1616/"
CITATION = ("B. J. Drouin, S. Yu, C. E. Miller, H. S. P. Mueller, F. Lewen, S. Bruenken, H. Habara, "
            "JQSRT 111, 1167 (2010); CDMS entry 032508 (O2, X 3Sigma-g, v=0)")
MHZ = 29979.2458


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--out", default="gold/cdms-o2-v0")
    a = ap.parse_args()
    data = Path(a.data)
    for f in ("v0.var", "c032508.egy"):
        if a.fetch or not (data / f).exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_data.py")), "url", BASE + f,
                            "--out", str(data), "--citation", CITATION], check=True)
    import riss_drucker as rd
    from riss_drucker.data import spfit

    var = spfit.read_var(data / "v0.var")
    p = {q.id: q for q in var.parameters}
    mapped = {0: "E(0) (energy origin)", 100: "B", 200: "-D", 110010000: "4lambda", 110010100: "4lambdaD",
              10000000: "gamma", 10000100: "gammaD"}
    th_mhz = {"B": p[100].value, "C": p[100].value, "DeltaN": -p[200].value, "D_ss": p[110010000].value / 2,
              "D_ss_N": p[110010100].value / 2, "eps_aa": p[10000000].value, "eps_bb": p[10000000].value,
              "eps_cc": p[10000000].value, "DeltaN_S": p[10000100].value}
    m = rd.compile_open_rotor(two_s=2, n_max=a.nmax, order=4, spin_rotation=True, spin_spin=True)
    th = {n: 0.0 for n in m.param_names} | {k: v / MHZ for k, v in th_mhz.items()} | {"A": 2.0e4}
    sol = rd.solve(m, th, nlowest=6, method="dense", keep_vectors=True)
    lab = rd.open_shell.assign(m, sol)
    ours = {}
    for b, r in zip(sol.blocks, sol.results):
        for e, l in zip(r["eigenvalues"], lab[b["key"]]):
            if l["Ka"] == 0:
                ours[(l["N"], int(l["J"]))] = e
    egy = {}
    for line in (data / "c032508.egy").read_text().splitlines():
        h, q = line.split(":")
        N, J = map(int, q.split())
        egy[(N, J)] = float(h.split()[2])
    keys = sorted(k for k in egy if k[0] <= a.nmax - 1)
    e0 = ours[(1, 0)]
    d = np.array([ours[k] - e0 - egy[k] for k in keys])
    rms, worst = float(np.sqrt(np.mean(d**2))), float(np.abs(d).max())
    ok = sol.certified and worst < 5e-6
    rec = {
        "schema_version": 1, "kind": "constants", "molecule": {"formula": "O2", "isotopologue": "16O2",
                                                               "electronic_state": "X 3Sigma-g, v=0"},
        "source": "measured",
        "provenance": {"citation": CITATION, "file": BASE + "v0.var",
                       "sha256": hashlib.sha256((data / "v0.var").read_bytes()).hexdigest(),
                       "extraction_method": "parser", "extraction_confidence": 1.0, "dataset": "CDMS 032508"},
        "units": {"energy": "MHz"}, "uncertainty_definition": "1sigma",
        "constants": {str(q.id): {"value": q.value, "uncertainty": q.error, "fixed": False}
                      for q in var.parameters if q.id in mapped},
        "convention": {"model": "open_rotor (linear limit)", "reduction": "n/a", "representation": "Ir",
                       "operator_convention": "SPFIT linear 3Sigma; mapping in scripts/gold_cdms_o2.py",
                       "unmapped": [f"{q.id} {q.label}" for q in var.parameters if q.id not in mapped]},
        "validation": {"status": "validated" if ok else "quarantined", "round_trip_rms": rms,
                       "detail": f"{len(keys)} SPCAT levels N <= {a.nmax - 1} matched by (N, J); max |diff| "
                                 f"{worst:.2e} cm-1 (print precision 5e-7)"},
    }
    try:
        rd.data.records.validate(rec)
    except ImportError:
        pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"O2: {len(keys)} levels, rms {rms:.2e}, max {worst:.2e} cm-1 -> {rec['validation']['status']}; wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
