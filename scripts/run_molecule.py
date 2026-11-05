#!/usr/bin/env python3
"""Run any molecule through riss_drucker.

  python scripts/run_molecule.py molecule.xyz [--charge 0] [--multiplicity 1] [--electronic F2]
                                 [--harmonic hf/6-31g] [--J 0,1,2] [--out result.json]

Prints the analysis (point group, rotor type, rotational constants, Gamma_vib, spin weights,
Jahn-Teller activity), the zeroth-order effective Hamiltonian (grammar terms and initial values),
its certificates, the lowest levels per block, and the next terms the symbolic language model
would propose. --harmonic computes harmonic frequencies with PySCF (method/basis) and uses them
for polyad weights and harmonic terms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--electronic", default=None, help="irrep of a degenerate electronic ground state")
    ap.add_argument("--harmonic", default=None, help="method/basis for PySCF harmonic frequencies")
    ap.add_argument("--J", default=None)
    ap.add_argument("--nmax", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from riss_drucker.molecule import Molecule, harmonic_frequencies, run
    from riss_drucker.symbolic_lm import SymbolicLM, default_degrees
    mol = Molecule.from_xyz(Path(a.xyz).read_text(), charge=a.charge, multiplicity=a.multiplicity,
                            electronic_irrep=a.electronic)
    info = mol.analyse()
    info.pop("character_table", None)
    freqs = None
    if a.harmonic:
        method, basis = a.harmonic.split("/")
        freqs = harmonic_frequencies(mol, method, basis)
        info["harmonic_frequencies_cm"] = freqs
    J = [float(x) if "." in x else int(x) for x in a.J.split(",")] if a.J else None
    out = run(mol, None, frequencies=freqs, J=J, nmax=a.nmax)
    p, bm = out["plan"], out["model"]
    lm = SymbolicLM.default()
    nxt = lm.generate(p["spec"], default_degrees(p["spec"]), top=8,
                      frequencies={n: w for n, _, w in p["modes"] if w} if freqs else None)
    res = {"analysis": info, "terms": p["spec"].terms, "theta": p["theta"],
           "certificates_passed": all(c["passed"] for c in bm.certificates), "n_certificates": len(bm.certificates),
           "blocks": len(bm.blocks),
           "lowest_levels": [{k: (v[:4] if k == "energies" else v) for k, v in lv.items()} for lv in out["levels"][:12]],
           "symbolic_lm_next_terms": nxt}
    txt = json.dumps(res, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
