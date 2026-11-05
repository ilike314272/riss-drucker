#!/usr/bin/env python3
"""Case study: H2(32)S MARVEL (Chubb et al. 2018), independent network inversion
and a ground-state Watson fit to our own inverted levels.

  python scripts/case_h2s.py --data DIR [--fetch] [--jcut 12] [--order 8] [--out results.json]

Data: Mendeley Data 10.17632/mx6skzhxp4.1 (four files: ortho/para input
transitions and output energy levels). --fetch downloads them through the
public API and checks the SHA-256 published by Mendeley.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

API = "https://data.mendeley.com/public-api/datasets/mx6skzhxp4/files?folder_id=root&version=1"
CITATION = "Chubb et al., JQSRT 218, 178-186 (2018); Mendeley Data 10.17632/mx6skzhxp4.1"


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = json.loads(urllib.request.urlopen(API, timeout=60).read())
    manifest = []
    for f in files:
        data = urllib.request.urlopen(f["content_details"]["download_url"], timeout=300).read()
        h = hashlib.sha256(data).hexdigest()
        if h != f["content_details"]["sha256_hash"]:
            raise SystemExit(f"{f['filename']}: SHA-256 mismatch with the published hash")
        (out / f["filename"]).write_bytes(data)
        manifest.append({"file": f["filename"], "sha256": h, "bytes": len(data), "doi": "10.17632/mx6skzhxp4.1"})
        print("fetched", f["filename"], len(data), "bytes, hash verified")
    (out / "manifest.json").write_text(json.dumps({"files": manifest, "citation": CITATION}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--jcut", type=int, default=12)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--out", default="h2s_case_study.json")
    a = ap.parse_args()
    data = Path(a.data)
    if a.fetch or not (data / "H2S_para_input_transitions.txt").exists():
        fetch(data)

    import riss_drucker as rd
    from riss_drucker.data import marvel, network
    from riss_drucker.rotor import label_index

    out = {"molecule": "H2(32)S", "citation": CITATION, "network": {}, "fit": {}}
    ground = []
    for spin in ("para", "ortho"):
        tr = marvel.read_transitions(data / f"H2S_{spin}_input_transitions.txt", n_qn=7)
        ref = {e["qn"]: e for e in marvel.read_energies(data / f"H2S_{spin}_energylevels_output.txt", n_qn=7)}
        anchor = min(ref, key=lambda k: ref[k]["energy"])
        res = network.invert(tr, anchors={anchor: ref[anchor]["energy"]})
        lv = res.levels
        common = [k for k in ref if k in lv and lv[k]["uncertainty"] is not None
                  and res.components[lv[k]["component"]]["anchored"]]
        d = np.array([lv[k]["energy"] - ref[k]["energy"] for k in common])
        u = np.array([ref[k]["uncertainty"] for k in common])
        out["network"][spin] = {
            "n_transitions": len(tr), "n_marvel_deleted": int(sum(t.excluded for t in tr)),
            "n_outliers_flagged": len(res.outliers), "n_components": len(res.components),
            "n_levels_marvel": len(ref), "n_compared": len(common),
            "abs_diff_median": float(np.median(np.abs(d))), "abs_diff_max": float(np.abs(d).max()),
            "fraction_within_marvel_unc": float(np.mean(np.abs(d) <= u + 1e-9)),
            "unc_median_ours": float(np.median([lv[k]["uncertainty"] for k in common])),
            "unc_median_marvel": float(np.median(u)),
        }
        print(f"{spin}: {len(common)}/{len(ref)} MARVEL levels reproduced; median |diff| {np.median(np.abs(d)):.1e}, "
              f"max {np.abs(d).max():.1e} cm-1; {np.mean(np.abs(d) <= u + 1e-9):.1%} within MARVEL unc; "
              f"{len(res.outliers)} lines flagged")
        for k in common:
            if k[:3] == ("0", "0", "0") and int(k[3]) <= a.jcut and lv[k]["confirmed"]:
                ground.append((int(k[3]), int(k[4]), int(k[5]), lv[k]["energy"], lv[k]["uncertainty"]))

    # Ground-state Watson fit to our own inverted levels (energies relative to 0_00).
    orders = [o for o in (4, 6, 8) if o <= a.order]
    models = [rd.compile_asym_rotor(j_max=a.jcut, order=o, axis="IIIr") for o in orders]
    idx = label_index(models[-1])
    obs = [rd.Observation(*idx[f"{j}({p},{q})"], e, max(s, 1e-6), label=f"{j}({p},{q})") for j, p, q, e, s in ground]
    fits = rd.fit_by_order(models, obs, {"A": 10.36, "B": 9.02, "C": 4.73}, max_iter=200)
    rob = rd.robust.fit_robust(models[-1], obs, fits[-1].theta, max_iter=200)
    f = rob.fit
    out["fit"] = {"model": f"asym_rotor Watson A, IIIr, order {orders[-1]}", "jcut": a.jcut, "n_levels": len(obs),
                  "reduced_chi2_by_order": {o: r.reduced_chi2 for o, r in zip(orders, fits)},
                  "robust_reduced_chi2": f.reduced_chi2, "outlier_fraction": rob.outlier_fraction,
                  "robust_warnings": rob.warnings, "certified": f.certified, "rank": f.rank,
                  "constants_cm": f.theta, "sigma_cm": f.sigma}
    print(f"ground-state fit: {len(obs)} levels J<={a.jcut}; chi2/dof by order "
          + ", ".join(f"{o}: {r.reduced_chi2:.3g}" for o, r in zip(orders, fits))
          + f"; robust {f.reduced_chi2:.3g}, outliers {rob.outlier_fraction:.3f}, warnings {rob.warnings}")
    print("  A, B, C =", ", ".join(f"{f.theta[n]:.6f}" for n in "ABC"), "cm-1")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
