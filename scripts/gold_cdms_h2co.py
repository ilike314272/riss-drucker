#!/usr/bin/env python3
"""Gold-set entry: H2CO ground-state constants from CDMS (Mueller & Lewen 2017),
round-trip validated against SPCAT's own energy levels.

  python scripts/gold_cdms_h2co.py --data DIR [--fetch] [--out gold/cdms-h2co-v3]

Fetches H2CO.var (SPFIT parameters) and c030501.egy (SPCAT levels) from CDMS,
builds the constants record (schemas/record.schema.json), and validates it by
reproducing the .egy levels with J <= min(KNMAX, 40) (beyond KNMAX the SPCAT
energies depend on its K-basis truncation). Also checks that a sign-flipped
copy is quarantined. Writes record.json (numbers + citation only).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = "https://cdms.astro.uni-koeln.de/classic/predictions/daten/H2CO/H2CO/"
CITATION = ("H. S. P. Mueller and F. Lewen, J. Mol. Spectrosc. 331, 28 (2017); "
            "CDMS entry 030501 (H2CO, v=0), version 3 (Feb. 2017)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="gold/cdms-h2co-v3")
    a = ap.parse_args()
    data = Path(a.data)
    for f in ("H2CO.var", "c030501.egy"):
        if a.fetch or not (data / f).exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_data.py")), "url", BASE + f,
                            "--out", str(data), "--citation", CITATION], check=True)

    import riss_drucker as rd
    from riss_drucker.data import records, roundtrip, spfit

    var = spfit.read_var(data / "H2CO.var")
    sha = hashlib.sha256((data / "H2CO.var").read_bytes()).hexdigest()
    rec = spfit.constants_record(var, citation=CITATION, doi="10.1016/j.jms.2016.10.004", url=BASE + "H2CO.var",
                                 sha256=sha, formula="H2CO",
                                 extra_provenance={"dataset": "CDMS 030501 v3"})
    egy = spfit.read_egy(data / "c030501.egy")
    jmax = min(var.knmax, 40)
    refs = [{"schema_version": 1, "kind": "level", "molecule": {"formula": "H2CO"}, "source": "ehamil_derived",
             "provenance": {"citation": "CDMS c030501.egy (SPCAT)", "extraction_method": "parser"},
             "units": {"energy": "cm-1"}, "uncertainty_definition": "1sigma", "value": e, "uncertainty": 1e-6,
             "assignment": {"label": lab, "J": int(lab.split("(")[0])}}
            for lab, e in egy.items() if int(lab.split("(")[0]) <= jmax]
    model = rd.compile_asym_rotor(j_max=jmax, order=8, reduction="S", axis="Ir")
    out = roundtrip.validate_constants(rec, refs, model=model)
    flipped = copy.deepcopy(rec)
    flipped["constants"]["40199"]["value"] *= -1  # d1 sign flip
    q = roundtrip.validate_constants(flipped, refs, model=model)
    try:
        records.validate(out)
        schema_ok = True
    except ImportError:
        schema_ok = None
    print(f"round trip vs SPCAT ({len(refs)} levels, J <= {jmax}): {out['validation']['status']}, "
          f"rms {out['validation']['round_trip_rms']:.2e} cm-1; sign-flipped d1: {q['validation']['status']}; "
          f"schema valid: {schema_ok}")
    print("unmapped:", rec["convention"]["unmapped"])
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest / "record.json")
    return 0 if out["validation"]["status"] == "validated" and q["validation"]["status"] == "quarantined" else 1


if __name__ == "__main__":
    sys.exit(main())
