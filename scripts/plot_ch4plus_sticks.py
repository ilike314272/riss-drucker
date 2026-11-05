#!/usr/bin/env python3
"""Full predicted rovibronic stick spectra of CH4+ and CD4+ (photoionization of CH4 / CD4).

  python scripts/plot_ch4plus_sticks.py --model docs/case_studies/ch4plus_llm_model_ccpvtz.json \
      [--pdf PAPER.pdf] [--T 7] [--jet] [--nmax 8] [--jmax 6] [--out-dir DIR]

--model is a model file written by scripts/llm_build_ch4plus.py (Claude's selected model or the
reference): its spec, the ab initio coefficients (IE offset only) and the coefficients after the
Bayesian update. Both are plotted for each isotopologue: ab initio sticks below the axis, the final
model above it; with --pdf the measured line positions are marked.

Spectrum model (as in the paper's simulations, intensities approximate):
* neutral CH4 / CD4: spherical top E'' = B J(J+1), rovibrational symmetry of each J from Td;
* nuclear-spin weights from Td (H: I = 1/2, D: I = 1); the nuclear-spin class is conserved
  (A <-> A1/A2, E <-> E, F <-> F1/F2);
* populations: Boltzmann at --T over J'' <= --jmax; with --jet each nuclear-spin species keeps
  its own lowest level only (no interconversion in a supersonic expansion: A J=0, F J=1, E J=2),
  which is what the PFI-ZEKE experiment sees;
* |N+ - J''| <= 2 with C_0 = 1, C_1 = 0.75, C_2 = 0.5, times (2N+1), for the total into each N+;
  that total is shared among the N+ sub-levels of the allowed class in proportion to their
  number of states (no photoionization dynamics beyond the Delta N propensities).
Writes one figure per isotopologue and the full assigned line lists (CSV).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
C2 = 1.4387769              # cm K
CLS = {"A1": "A", "A2": "A", "E": "E", "F1": "F", "F2": "F"}
CDN = {0: 1.0, 1: 0.75, 2: 0.5}
SPIN = {"CH4+": 0.5, "CD4+": 1.0}


def neutral_levels(B, jmax, T, jet):
    from riss_drucker import symmetry as S
    td = S.get_group("Td")
    out = []
    for J in range(jmax + 1):
        for g, mult in td.level_counts(2 * J).items():
            out.append((J, g, mult, B * J * (J + 1)))
    if jet:
        keep = {}
        for J, g, m, E in out:
            c = CLS[g]
            if c not in keep or E < keep[c][3]:
                keep[c] = (J, g, m, E)
        out = list(keep.values())
    return out, td


def sticks(cand, IE, coeffs, name, B, jmax, T, jet):
    import llm_build_ch4plus as L
    lev = L.levels(cand, coeffs)
    lower, td = neutral_levels(B, jmax, T, jet)
    gns = td.spin_statistics([SPIN[name]] * 4)
    dim = {ir: cand.dims.get(ir, 1) for ir in cand.dims}
    out = []
    for J, g, mult, El in lower:
        pop = gns[g] * mult * (2 * J + 1) * math.exp(-C2 * El / T)
        for N in range(max(0, J - 2), J + 3):
            targets = [(key, e) for key, e in lev.items() if key[0] == N and CLS.get(key[1].split("+")[0]) == CLS[g]]
            tot = sum(dim[key[1]] for key, _ in targets)
            for (n, ir, k), e in targets:        # the N+ total is shared over its sub-levels by number of states
                out.append((IE + e - El, pop * CDN[abs(N - J)] * (2 * N + 1) * dim[ir] / tot, J, g, n, ir, k))
    imax = max(s[1] for s in out)
    return sorted((p, i / imax, *rest) for p, i, *rest in out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model JSON from llm_build_ch4plus.py")
    ap.add_argument("--pdf", default=None, help="paper PDF: mark the measured line positions")
    ap.add_argument("--T", type=float, default=7.0, help="rotational temperature / K (experiment: about 7 K)")
    ap.add_argument("--jet", action="store_true", help="only the lowest level of each nuclear-spin species")
    ap.add_argument("--nmax", type=int, default=8, help="highest cation N+")
    ap.add_argument("--jmax", type=int, default=6, help="highest neutral J''")
    ap.add_argument("--min-rel", type=float, default=1e-3, help="drop sticks weaker than this (relative)")
    ap.add_argument("--out-dir", default="docs/case_studies/figures")
    a = ap.parse_args()
    import llm_build_ch4plus as L
    import case_methane_cation_rovibronic as R
    m = json.loads(Path(a.model).read_text())
    spec = dict(m["spec"])
    spec["J"] = list(range(a.nmax + 1))
    cand = L.Candidate({k: spec[k] for k in ("group", "spaces", "terms", "J")}, m["roles"])
    tables = R.read_tables(Path(a.pdf)) if a.pdf else None
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = f"{'jet' if a.jet else 'T' + format(a.T, 'g') + 'K'}"
    label = m.get("selected", "model")
    for name in ("CH4+", "CD4+"):
        sc = m["scores"][name]
        B = R.NEUTRAL_B[name]
        ai = [s for s in sticks(cand, sc["IE_offset"], sc["ab_initio_coefficients"], name, B, a.jmax, a.T, a.jet)
              if s[1] >= a.min_rel]
        up = [s for s in sticks(cand, sc["posterior"]["IE"], sc["posterior_coefficients"], name, B, a.jmax, a.T, a.jet)
              if s[1] >= a.min_rel]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.vlines([s[0] for s in up], 0, [s[1] for s in up], color="C1", lw=1.0,
                  label=f"final model ({label}; rms {sc['updated_rms_cm']:.2f} cm$^{{-1}}$ on measured lines)")
        ax.vlines([s[0] for s in ai], 0, [-s[1] for s in ai], color="C0", lw=1.0,
                  label=f"ab initio constants, IE offset only (rms {sc['prediction_rms_IE_only_cm']:.2f})")
        if tables is not None:
            obs = [r["nu"] for r in tables[name] if r["N"] and "/" not in r["N"]]
            ax.plot(obs, np.full(len(obs), 1.06), "kv", ms=4, label="measured lines (Wörner, Qian & Merkt 2007)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.12, 1.15)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["1", "0.5", "0", "0.5", "1"])
        ax.set_ylabel("relative intensity")
        ax.set_xlabel("wave number / cm$^{-1}$")
        pops = "nuclear-spin species in their lowest levels (jet)" if a.jet else f"T = {a.T:g} K, J'' ≤ {a.jmax}"
        ax.set_title(f"{name} ← {name[:-1]} rovibronic stick spectrum, N$^+$ ≤ {a.nmax}; {pops}")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        stem = f"{name.replace('+', 'p')}_sticks_{tag}"
        fig.savefig(od / f"{stem}.png", dpi=150)
        plt.close(fig)
        with open(od / f"{stem}.csv", "w") as fh:
            fh.write("model,wavenumber_cm,relative_intensity,J_neutral,sym_neutral,N_cation,sym_cation,index\n")
            for lab, ss in (("final", up), ("abinitio", ai)):
                for s in ss:
                    fh.write(f"{lab},{s[0]:.3f},{s[1]:.5f},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]}\n")
        print(f"{name}: {len(up)} final / {len(ai)} ab initio sticks -> {od / stem}.png, .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
