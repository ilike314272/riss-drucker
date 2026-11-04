#!/usr/bin/env python3
"""Case study: H2(12C)(16O) ground state from ExoMol AYTY (MARVEL-tagged levels).

  python scripts/case_h2co.py --data DIR [--fetch] [--jcut 30] [--out results.json]

1. (--fetch) download AYTY .def.json/.states.bz2/.pf with provenance (fetch_data.py)
2. read ground-vibrational-state levels; keep MARVEL-tagged ones (SourceType = Ma)
3. fit the Watson A-reduced sextic Hamiltonian (Ir) with the outlier mixture
4. calculated-vs-empirical error prior over all MARVEL-tagged states (README 6.4)
5. write a JSON summary (derived numbers + citations; no raw data)

Data: ExoMol AYTY (Al-Refaie et al. 2015; MARVEL-refined, Al-Derzi et al. 2021,
JQSRT 266, 107563); dataset DOI 10.5281/zenodo.5716868. Check the license
before redistributing the raw files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

STEM = "1H2-12C-16O__AYTY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--jcut", type=int, default=30)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--out", default="h2co_case_study.json")
    a = ap.parse_args()
    data = Path(a.data)
    if a.fetch or not (data / f"{STEM}.states.bz2").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_data.py")), "exomol",
                        "H2CO/1H2-12C-16O/AYTY", "--out", str(data)], check=True)

    import riss_drucker as rd
    from riss_drucker.data import error_prior, exomol
    from riss_drucker.rotor import label_index

    fields = exomol.read_def(data / f"{STEM}.def.json")["states_fields"]
    vcols = [fields.index(f"Herzberg:v{k}") for k in range(1, 7)]
    src = fields.index("Auxiliary:SourceType")
    ma = exomol.read_states(data / f"{STEM}.states.bz2", fields, keep=lambda t: t[src] == "Ma")
    vib = np.array(["".join(x) for x in zip(*[ma[f"Herzberg:v{k}"].astype(int).astype(str) for k in range(1, 7)])])

    # ---- ground-state fit -------------------------------------------------
    g = (vib == "000000") & (ma["J"] <= a.jcut)
    J = ma["J"][g].astype(int)
    Ka, Kc = ma["Herzberg:Ka"][g].astype(int), ma["Herzberg:Kb"][g].astype(int)
    if not np.all((Ka + Kc == J) | (Ka + Kc == J + 1)):
        raise SystemExit("unexpected rotational labels: Ka + Kb is not J or J+1")
    model = rd.compile_asym_rotor(j_max=a.jcut, order=a.order)
    idx = label_index(model)
    obs = [rd.Observation(*idx[f"{j}({p},{q})"], float(e), max(float(u), 1e-6), label=f"{j}({p},{q})")
           for j, p, q, e, u in zip(J, Ka, Kc, ma["E"][g], ma["unc"][g])]
    start = {n: 0.0 for n in model.param_names} | {"A": 9.4, "B": 1.3, "C": 1.13}
    plain = rd.fit(model, obs, start)
    rob = rd.robust.fit_robust(model, obs, plain.theta)
    f = rob.fit
    MHZ = rd.data.records.MHZ_PER_CM
    p = np.asarray(rob.outlier_probability)
    flagged = [{"label": o.label, "residual_cm": lv["residual"], "sigma_cm": o.sigma}
               for o, lv, pp in zip(obs, f.levels, p) if pp > 0.5]

    # ---- error prior ------------------------------------------------------------
    s = error_prior.summarize(ma["E"], ma["Auxiliary:Ecal"], ma["J"], band=vib, symmetry=ma["Grve"])
    d = ma["E"] - ma["Auxiliary:Ecal"]
    bands = {b: {"n": int((vib == b).sum()), "mean": float(d[vib == b].mean()),
                 "robust_std": error_prior._robust_std(d[vib == b])} for b in sorted(set(vib))}

    manifest = json.loads((data / "manifest.json").read_text()) if (data / "manifest.json").exists() else {}
    out = {
        "molecule": "H2(12C)(16O)", "dataset": "ExoMol AYTY", "dataset_doi": "10.5281/zenodo.5716868",
        "citations": ["Al-Derzi et al., JQSRT 266, 107563 (2021) [MARVEL]", "Al-Refaie et al., MNRAS 448, 1704 (2015) [AYTY]"],
        "files": manifest.get("files", []),
        "fit": {"model": f"asym_rotor Watson A, Ir, order {a.order}", "jcut": a.jcut, "n_levels": len(obs),
                "plain_reduced_chi2": plain.reduced_chi2, "robust_reduced_chi2": f.reduced_chi2,
                "outlier_fraction": rob.outlier_fraction, "certified": f.certified, "rank": f.rank,
                "constants_cm": f.theta, "sigma_cm": f.sigma,
                "constants_MHz": {k: v * MHZ for k, v in f.theta.items()},
                "flagged_levels": flagged},
        "error_prior": {"n": s["n"], "mean": s["mean"], "robust_std": s["robust_std"],
                        "sigma_model": s["sigma_model"], "bands": bands},
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"fit: {len(obs)} levels, robust chi2/dof {f.reduced_chi2:.2f}, outliers {rob.outlier_fraction:.3f}; "
          f"A,B,C = {f.theta['A'] * MHZ:.2f}, {f.theta['B'] * MHZ:.3f}, {f.theta['C'] * MHZ:.3f} MHz")
    print(f"error prior: {s['n']} MARVEL states, mean {s['mean']:+.4f}, robust std {s['robust_std']:.4f} cm-1")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
