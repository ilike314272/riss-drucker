#!/usr/bin/env python3
"""CH4+ / CD4+ rovibronic levels and spectra from first principles (no measured line enters).

  python scripts/firstprinciples_ch4plus.py --abinitio data/abinitio/ch4plus_ccpvtz [--pdf PAPER.pdf]
      [--out docs/case_studies/ch4plus_firstprinciples_ccpvtz.json] [--figures DIR] [--nmax 8]

Uses riss_drucker.abinitio.firstprinciples: Cartesian reaction-plane tunneling (ab initio plane grid
if the 'plane2d' stage exists, harmonic valley otherwise), vibrationally averaged rotor tensor, frame
rotation along the path, Ham's theorem for the sign of sigma, and the adiabatic IE from UCCSD(T) +
harmonic ZPE. The output JSON holds the first-principles levels (the target of the data-free Claude
loop) and, with --pdf, a validation against the measured lines that changes nothing in the model:
'absolute' uses the ab initio IE; 'IE shifted' reports the rms after the single offset only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ISOS = (("CH4+", "H"), ("CD4+", "D"))


def assign(levels, IE, rows, Bn):
    """Each measured line -> nearest first-principles level of the allowed class at the published N+;
    the published N+ is questioned only as in the case study (miss > 3 cm-1 and a neighbouring N+ fits
    three times better), and every relabelling is reported."""
    import case_methane_cation_rovibronic as R
    out, relabel = [], []
    for r in rows:
        El = Bn * r["J"] * (r["J"] + 1)
        dist = lambda k: abs(IE + levels[k] - El - r["nu"])
        cands = [k for k in levels if k[0] == int(r["N"]) and k[1] in R.allowed(r["Gp"], r["G"])]
        if not cands:
            raise ValueError(f"no first-principles level of class {r['Gp']} at N+={r['N']}")
        best = min(cands, key=dist)
        if dist(best) > R.RELABEL_TOL:
            alt = [k for k in levels if abs(k[0] - int(r["N"])) == 1 and k[1] in R.allowed(r["Gp"], r["G"])]
            if alt and dist(min(alt, key=dist)) < dist(best) / 3:
                best = min(alt, key=dist)
                relabel.append(f"{r['nu']:.2f}: N+ {r['N']} -> {best[0]}")
        out.append(best)
    return out, relabel


def validate(levels, IE, rows, Bn):
    """Measured lines vs first-principles levels. Nothing is fitted; 'IE shifted' applies the one
    offset that the ab initio IE error requires (assignment redone at the shifted IE)."""
    obs = np.array([r["nu"] for r in rows])
    El = np.array([Bn * r["J"] * (r["J"] + 1) for r in rows])
    asg, _ = assign(levels, IE, rows, Bn)
    off = float(np.mean(obs - (IE + np.array([levels[k] for k in asg]) - El)))
    for _ in range(5):                                      # assignment at the shifted IE, to convergence
        asg, relabel = assign(levels, IE + off, rows, Bn)
        calc = IE + np.array([levels[k] for k in asg]) - El
        new = float(np.mean(obs - calc))
        if abs(new - off) < 1e-9:
            break
        off = new
    return {"n_lines": len(obs), "rms_absolute_cm": float(np.sqrt(np.mean((obs - calc) ** 2))),
            "IE_offset_cm": off, "rms_IE_shifted_cm": float(np.sqrt(np.mean((obs - calc - off) ** 2))),
            "relabelled": relabel, "residuals_IE_shifted": (obs - calc - off).tolist(), "obs": obs.tolist(),
            "assignment": [list(k) for k in asg]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abinitio", required=True)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--no-grid", action="store_true", help="ignore the plane2d stage (harmonic valley)")
    a = ap.parse_args()
    from riss_drucker.abinitio import firstprinciples as FP
    from riss_drucker.abinitio.rpsurface import load_stages
    import case_methane_cation_abinitio as CA
    import case_methane_cation_rovibronic as R
    abd = Path(a.abinitio)
    st = load_stages(abd)
    spec = R.Model(5).spec.to_dict()
    spec = {k: spec[k] for k in ("group", "spaces", "terms")}
    tables = R.read_tables(Path(a.pdf)) if a.pdf else None
    res = {"basis": st["min"].get("meta", {}).get("basis"), "uses_plane_grid": ("plane2d" in st) and not a.no_grid,
           "nmax": a.nmax, "model_space": spec["spaces"], "isotopologues": {}}
    for name, iso in ISOS:
        t0 = time.time()
        print(f"{name}: tunneling on the reaction plane ...", file=sys.stderr, flush=True)
        ta = FP.tunneling_analysis(st, iso, use_grid=not a.no_grid)
        print(f"  delta = {ta['delta']:.3f} cm-1 ({ta['surface']}); averaged A B C = "
              f"{np.round(ta['rotational_constants_averaged'], 3)}; frame rotation "
              f"{ta['frame_rotation']['angle_deg']:.2f} deg", file=sys.stderr, flush=True)
        main_lv = FP.rovibronic_levels(spec, range(a.nmax + 1), ta, variant="R")
        conv = {}
        for var in ("Rinv", "none"):
            conv[var] = FP.rovibronic_levels(spec, range(6), ta, variant=var, sign=main_lv["sign"])["levels"]
        IE = CA.ionization_energy(abd, iso)["adiabatic_cm"]
        entry = {"tunneling": ta, "sign_sigma_by_Ham": main_lv["sign"], "symmetry_leakage": main_lv["symmetry_leakage"],
                 "wells_LR": main_lv["wells"], "ionization_energy_abinitio_cm": IE,
                 "levels": FP.levels_to_json(main_lv["levels"])}
        if tables is not None:
            rows = [r for r in tables[name] if r["N"] and "/" not in r["N"]]
            Bn = R.NEUTRAL_B[name]
            entry["validation"] = validate(main_lv["levels"], IE, rows, Bn)
            entry["validation_convention_checks"] = {
                "frame rotation inverted": validate(conv["Rinv"], IE, rows, Bn)["rms_IE_shifted_cm"],
                "no frame rotation": validate(conv["none"], IE, rows, Bn)["rms_IE_shifted_cm"]}
            v = entry["validation"]
            print(f"  validation ({v['n_lines']} measured lines, nothing fitted): rms {v['rms_absolute_cm']:.1f} cm-1 "
                  f"absolute (ab initio IE {IE:.0f}); {v['rms_IE_shifted_cm']:.2f} cm-1 after shifting the IE by "
                  f"{v['IE_offset_cm']:+.0f}", file=sys.stderr, flush=True)
        entry["seconds"] = round(time.time() - t0, 1)
        res["isotopologues"][name] = entry
    txt = json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if a.out:
        Path(a.out).write_text(txt)
    if a.figures:
        figures(Path(a.figures), res, tables)
    summary = {n: {"delta_cm": e["tunneling"]["delta"], "A_B_C_averaged": e["tunneling"]["rotational_constants_averaged"],
                   "frame_rotation_deg": e["tunneling"]["frame_rotation"]["angle_deg"],
                   **({k: e["validation"][k] for k in ("rms_absolute_cm", "rms_IE_shifted_cm", "IE_offset_cm")}
                      if "validation" in e else {})} for n, e in res["isotopologues"].items()}
    print(json.dumps(summary, indent=1))
    return 0


def figures(fd: Path, res, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_ch4plus_sticks as PS
    import case_methane_cation_rovibronic as R
    from riss_drucker.abinitio.firstprinciples import levels_from_json
    fd.mkdir(parents=True, exist_ok=True)
    for name, e in res["isotopologues"].items():
        lev = levels_from_json(e["levels"])
        shift = e.get("validation", {}).get("IE_offset_cm", 0.0)
        IE = e["ionization_energy_abinitio_cm"] + shift
        st = [s for s in PS.sticks_from_levels(lev, IE, name, R.NEUTRAL_B[name], 6, 7.0, True) if s[1] > 1e-3]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.vlines([s[0] for s in st], 0, [s[1] for s in st], color="C2", lw=1.2,
                  label=f"first principles (IE shifted by {shift:+.0f} cm$^{{-1}}$, nothing else fitted)")
        if tables is not None:
            obs = [r["nu"] for r in tables[name] if r["N"] and "/" not in r["N"]]
            ax.plot(obs, np.full(len(obs), 1.06), "kv", ms=4, label="measured lines (Wörner, Qian & Merkt 2007)")
        v = e.get("validation")
        ax.set_title(f"{name}: first-principles rovibronic stick spectrum (jet), δ = {e['tunneling']['delta']:.2f} cm$^{{-1}}$"
                     + (f"; rms {v['rms_IE_shifted_cm']:.2f} cm$^{{-1}}$ on {v['n_lines']} lines" if v else ""))
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("wave number / cm$^{-1}$")
        ax.set_ylabel("relative intensity")
        ax.legend(fontsize=8, loc="center right")
        fig.tight_layout()
        stem = f"{name.replace('+', 'p')}_firstprinciples_jet"
        fig.savefig(fd / f"{stem}.png", dpi=150)
        plt.close(fig)
        with open(fd / f"{stem}.csv", "w") as fh:
            fh.write("wavenumber_cm,relative_intensity,J_neutral,sym_neutral,N_cation,sym_cation,index\n")
            for s in st:
                fh.write(f"{s[0]:.3f},{s[1]:.5f},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]}\n")


if __name__ == "__main__":
    sys.exit(main())
