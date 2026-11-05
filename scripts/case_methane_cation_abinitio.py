#!/usr/bin/env python3
"""From the ab initio CH4+ surface to the effective tunneling-rotation Hamiltonian and spectrum.

  python scripts/case_methane_cation_abinitio.py --abinitio DIR [--pdf PAPER.pdf] [--figures DIR] [--out JSON]

Input: the cached stages of scripts/abinitio_ch4plus.py (UMP2 minimum, saddle, reaction path, Hessians;
UCCSD(T) single points; neutral CH4) in the basis recorded in the stage files. Everything below runs in seconds.

1. Rotational constants of the C2v minimum (CH4+ and CD4+): A, B, C of the localized rotor.
2. Reaction path: mass-weighted arc length s, V(s) = UMP2 + [UCCSD(T) - UMP2] (interpolated
   between the CCSD(T) points), ZPE of the 8 orthogonal modes from projected Hessians; the same
   path and Hessians with deuterium masses for CD4+ (path geometry kept - stated approximation).
3. 1D reaction-path tunneling between adjacent minima -> pair splitting 2|sigma| -> delta = 4|sigma|.
4. Adiabatic ionization energy from UCCSD(T) and harmonic ZPEs.
5. Effective Hamiltonian = the symmetry-built model of case_methane_cation_rovibronic with the ab
   initio A, B, C, sigma; its spectrum is predicted with only IE adjusted (one offset), compared
   with the measured lines of Woerner, Qian & Merkt (2007) when --pdf is given, then updated with
   the ab initio values as a Gaussian prior (Bayesian update; shifts in prior-sd units).
6. Figures: path profile with tunneling levels; predicted vs measured spectra; residuals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
H2CM = 219474.6313705
NEUTRAL_B = {"CH4+": 5.2410, "CD4+": 2.63273}


def load(d: Path, name):
    return json.loads((d / f"ch4plus_{name}.json").read_text())


def rotational_constants(xyz, iso):
    from riss_drucker.molecule import Molecule
    m = Molecule(["C"] + [iso] * 4, np.array(xyz))
    c = sorted(m.rotational_constants().values(), reverse=True)
    return {"A": c[0], "B": c[1], "C": c[2]}


def saddle_tangent(H, m):
    """Cartesian displacement of the imaginary normal mode (mass-weighted eigenvector of the Hessian,
    translations/rotations are zero modes there and do not mix with a negative eigenvalue)."""
    sm = np.repeat(np.sqrt(m), 3)
    w, V = np.linalg.eigh(H / np.outer(sm, sm))
    return (V[:, 0] / sm).reshape(-1, 3)


def path_profile(abd: Path, iso: str):
    from riss_drucker.abinitio.rph import MASS, projected_frequencies
    from riss_drucker.abinitio.tunneling import arc_length, kabsch
    irc, hs, cc = load(abd, "irc"), load(abd, "hess"), load(abd, "ccsdt")
    m = np.array([MASS["C"]] + [MASS[iso]] * 4)
    path = [np.array(p) for p in irc["path"]]
    s = arc_length(path, m)
    E = np.array(irc["e_ump2"]) * H2CM
    # CCSD(T) correction interpolated along s (it is smooth; few points suffice)
    ci = cc["indices"]
    corr = np.array(cc["e_ccsdt"]) * H2CM - E[ci]
    Vc = E + np.interp(s, s[ci], corr)
    # ZPE of the orthogonal modes from projected Hessians
    zs, zp = [], []
    for k, H in zip(hs["indices"], hs["hessians"]):
        j0, j1 = max(k - 1, 0), min(k + 1, len(path) - 1)
        tan = kabsch(path[k], path[j1], m) - kabsch(path[k], path[j0], m)
        if k == 0:
            tan = saddle_tangent(np.array(H), m)      # exact IRC direction at the saddle
        if np.linalg.norm(tan) < 1e-10:
            tan = None
        w = projected_frequencies(np.array(H), path[k], m, tangent=tan)
        zs.append(s[k])
        zp.append(0.5 * float(np.sum(w[w > 0])))
    zpe = np.interp(s, zs, zp)
    return {"s": s, "V_mp2": E - E[-1], "V": Vc - Vc[-1], "zpe": zpe - zpe[-1], "zpe_abs_min": zp[-1],
            "zpe_abs_ts": zp[0], "ccsdt_points": ci}


def tunneling(prof, scale_barrier=1.0, with_zpe=True):
    from riss_drucker.abinitio.tunneling import splitting
    s = prof["s"]
    U = prof["V"] * scale_barrier + (prof["zpe"] if with_zpe else 0.0)
    return splitting(s, U)


def ionization_energy(abd: Path, iso="H"):
    from riss_drucker.abinitio.rph import MASS, projected_frequencies
    mn, neu, cc, hs = load(abd, "min"), load(abd, "neutral"), load(abd, "ccsdt"), load(abd, "hess")
    m = np.array([MASS["C"]] + [MASS[iso]] * 4)
    e_cat = cc["e_ccsdt"][-1]                    # last CCSD(T) point = IRC end = minimum
    w_cat = projected_frequencies(np.array(hs["hessians"][-1]), np.array(load(abd, "irc")["path"][-1]), m)
    w_neu = projected_frequencies(np.array(neu["hessian"]), np.array(neu["xyz"]), m)
    zc, zn = 0.5 * np.sum(w_cat[w_cat > 0]), 0.5 * np.sum(w_neu[w_neu > 0])
    return {"electronic_cm": (e_cat - neu["e_ccsdt"]) * H2CM, "zpe_cation_cm": float(zc), "zpe_neutral_cm": float(zn),
            "adiabatic_cm": float((e_cat - neu["e_ccsdt"]) * H2CM + zc - zn)}


def simulate(model_mod, model, x, sgn, name, T=7.0, fwhm=0.55):
    """Stick spectrum: lowest neutral level of each nuclear-spin species (J=0 A1, J=1 F1, J=2 E)
    to every cation level of the same nuclear-spin class, intensity ~ g_ns (2J+1) exp(-E''/kT) C_dN
    with C_0 = 1, C_+-1 = 0.75, C_+-2 = 0.5 (as in the paper's simulations)."""
    from riss_drucker import symmetry as S
    lev = model_mod.levels_indexed(model, x, sgn)
    td = S.get_group("Td")
    gns = td.spin_statistics([0.5 if name.startswith("CH") else 1.0] * 4)
    Bn = NEUTRAL_B[name]
    cls = {"A1": "A", "A2": "A", "F1": "F", "F2": "F", "E": "E"}
    lower = [(0, "A1"), (1, "F1"), (2, "E")]
    Cdn = {0: 1.0, 1: 0.75, -1: 0.75, 2: 0.5, -2: 0.5}
    sticks = []
    for J, g in lower:
        El = Bn * J * (J + 1)
        pop = gns[g] * (2 * J + 1) * math.exp(-1.4387769 * El / T)
        for (N, ir, k), e in lev.items():
            dn = N - J
            if cls[ir] != cls[g] or abs(dn) > 2:
                continue
            sticks.append((x[0] + e - El, pop * Cdn[dn] * (2 * N + 1), J, g, N, ir, k))
    grid = np.linspace(min(s[0] for s in sticks) - 5, max(s[0] for s in sticks) + 5, 4000)
    sig = fwhm / 2.3548
    spec = sum(s[1] * np.exp(-0.5 * ((grid - s[0]) / sig) ** 2) for s in sticks)
    return sticks, grid, spec / max(spec.max(), 1e-300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abinitio", required=True)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    abd = Path(a.abinitio)
    import case_methane_cation_rovibronic as R

    mn, ts = load(abd, "min"), load(abd, "ts")
    basis = mn.get("meta", {}).get("basis", "?")
    res = {"levels_of_theory": {"geometries_hessians": f"UMP2/{basis}",
                                "energies": f"UCCSD(T)/{basis} at the minimum, saddle and reaction-path points",
                                "program": mn.get("meta", {}).get("program")},
           "basis": basis}
    res["stationary_points"] = {
        "min_CH_bonds_A": sorted(float(np.linalg.norm(np.array(mn["xyz"])[i] - np.array(mn["xyz"])[0])) for i in range(1, 5)),
        "ts_imaginary_frequency_cm": float(min(ts["frequencies"])),
        "ts_n_imaginary": int(sum(1 for w in ts["frequencies"] if w < 0))}
    for iso, name in (("H", "CH4+"), ("D", "CD4+")):
        prof = path_profile(abd, iso)
        rc = rotational_constants(mn["xyz"], iso)
        t0 = tunneling(prof)
        sens = {f"barrier_x{f}": 4 * tunneling(prof, f)["sigma_abs"] for f in (0.9, 1.1)}
        res[name] = {"rotational_constants_equilibrium": rc,
                     "barrier_electronic_cm": float(prof["V"].max()), "barrier_mp2_cm": float(prof["V_mp2"].max()),
                     "barrier_zpe_corrected_cm": float((prof["V"] + prof["zpe"]).max()),
                     "path_length_amu_half_A": float(prof["s"][-1]),
                     "tunneling": t0, "delta_ab_initio": 4 * t0["sigma_abs"],
                     "delta_without_zpe": 4 * tunneling(prof, with_zpe=False)["sigma_abs"],
                     "delta_sensitivity": sens}
        res[name]["_profile"] = prof
    res["ionization_energy"] = ionization_energy(abd)
    res["isotope_ratio_delta"] = res["CH4+"]["delta_ab_initio"] / max(res["CD4+"]["delta_ab_initio"], 1e-300)

    # effective Hamiltonian from ab initio + comparison with the measured spectrum
    model = R.Model(5)
    comparison = {}
    tables = R.read_tables(Path(a.pdf)) if a.pdf else None
    for name in ("CH4+", "CD4+"):
        rc = res[name]["rotational_constants_equilibrium"]
        sig = res[name]["delta_ab_initio"] / 4
        x_ai = np.array([0.0, rc["A"], rc["B"], rc["C"], sig])
        entry = {"effective_hamiltonian_ab_initio": {"A": rc["A"], "B": rc["B"], "C": rc["C"], "sigma": sig,
                                                     "delta": 4 * sig}}
        if tables is not None:
            rows = [r for r in tables[name] if r["N"] and "/" not in r["N"]]
            Bn = NEUTRAL_B[name]
            obs = np.array([r["nu"] for r in rows])
            best = None
            for sgn in (1.0, -1.0):
                x = x_ai.copy()
                x[0] = R.PAPER[name]["IE"]
                asg = R.assign(model, x, sgn, rows, Bn)
                calc = R.calc_lines(model, x, sgn, rows, Bn, asg)
                x[0] += float(np.mean(obs - calc))             # one offset: IE
                calc = R.calc_lines(model, x, sgn, rows, Bn, R.assign(model, x, sgn, rows, Bn))
                rms = float(np.sqrt(np.mean((obs - calc) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, sgn, x.copy(), calc)
            rms, sgn, x, calc = best
            entry["prediction_no_fit"] = {"rms_cm": rms, "IE_offset_fitted": x[0], "rD_sign": sgn,
                                          "n_lines": len(obs)}
            # Bayesian update: ab initio prior (model-error sd: 5% on A, B, C; factor 2 on sigma)
            from scipy.optimize import least_squares
            prior_mu = x.copy()
            prior_sd = np.array([np.inf, 0.05 * rc["A"], 0.05 * rc["B"], 0.05 * rc["C"], 0.7 * abs(sig) + 0.3])
            # MAP estimate with re-assignment: assign at the current parameters, solve, re-assign,
            # until the assignment is stable (assignments are not frozen at the prior prediction)
            p_cur, asg, seen = x.copy(), None, []
            for _ in range(10):
                asg_new = R.assign(model, p_cur, sgn, rows, Bn)
                key = json.dumps(asg_new, sort_keys=True, default=str)
                if key in seen:
                    break
                seen.append(key)
                asg = asg_new

                def r_(p, asg=asg):
                    c = R.calc_lines(model, p, sgn, rows, Bn, asg)
                    pri = [(p[i] - prior_mu[i]) / prior_sd[i] for i in range(1, 5)]
                    return np.r_[(obs - c) / 0.5, pri]           # sigma_line = 0.5 cm-1 (resolution)
                sol = least_squares(r_, p_cur, x_scale=np.r_[1.0, 0.1, 0.1, 0.1, 0.5], diff_step=1e-6)
                p_cur = sol.x
            calc_u = R.calc_lines(model, sol.x, sgn, rows, Bn, asg)
            entry["bayesian_update"] = {
                "posterior": dict(zip(["IE", "A", "B", "C", "sigma"], map(float, sol.x))),
                "delta": float(4 * abs(sol.x[4])), "assignment_iterations": len(seen), "rms_cm": float(np.sqrt(np.mean((obs - calc_u) ** 2))),
                "shift_in_prior_sd": {k: float((sol.x[i] - prior_mu[i]) / prior_sd[i])
                                      for i, k in zip(range(1, 5), ["A", "B", "C", "sigma"])},
                "paper_constants": R.PAPER[name]}
            entry["_plot"] = {"obs": obs, "calc_ai": calc, "calc_up": calc_u, "x_ai": x, "x_up": sol.x, "sgn": sgn}
        comparison[name] = entry
    res["effective_hamiltonian"] = comparison

    if a.figures:
        make_figures(Path(a.figures), res, R, model)
    clean = json.loads(json.dumps({k: v for k, v in res.items()}, default=lambda o: None if isinstance(o, np.ndarray)
                                  else str(o)))
    for name in ("CH4+", "CD4+"):
        clean[name].pop("_profile", None)
        clean["effective_hamiltonian"][name].pop("_plot", None)
    txt = json.dumps(clean, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


def make_figures(fd: Path, res, R, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd.mkdir(parents=True, exist_ok=True)
    # 1. path profiles
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name in zip(axes, ("CH4+", "CD4+")):
        p = res[name]["_profile"]
        ax.plot(np.r_[-p["s"][::-1], p["s"][1:]], np.r_[p["V"][::-1], p["V"][1:]], label="V (UCCSD(T)-corrected)")
        ax.plot(np.r_[-p["s"][::-1], p["s"][1:]], np.r_[(p["V"] + p["zpe"])[::-1], (p["V"] + p["zpe"])[1:]],
                label="V + ZPE$_\\perp$")
        lv = res[name]["tunneling"]["levels"]
        base = float((p["V"] + p["zpe"]).min())
        for e in lv[:2]:
            ax.axhline(base + e, color="k", lw=0.6, ls=":")
        ax.set_title(f"{name}: δ = {res[name]['delta_ab_initio']:.2f} cm$^{{-1}}$")
        ax.set_xlabel("mass-weighted path s / amu$^{1/2}$ Å (saddle at 0)")
    axes[0].set_ylabel("energy / cm$^{-1}$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "methane_cation_path.png", dpi=150)
    plt.close(fig)
    # 2. spectra and 3. residuals
    if "_plot" not in res["effective_hamiltonian"]["CH4+"]:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    for ax, name in zip(axes, ("CH4+", "CD4+")):
        pl = res["effective_hamiltonian"][name]["_plot"]
        for x, lab, col in ((pl["x_ai"], "ab initio effective H (IE offset only)", "C0"),
                            (pl["x_up"], "after Bayesian update", "C1")):
            st, grid, spec = simulate(R, model, x, pl["sgn"], name)
            ax.plot(grid, spec, color=col, lw=1.0, label=lab)
            tag = "abinitio" if "ab initio" in lab else "updated"
            imax = max(s[1] for s in st)
            with open(fd / f"methane_cation_{name.replace('+', 'p')}_lines_{tag}.csv", "w") as fh:
                fh.write("wavenumber_cm,relative_intensity,J_neutral,sym_neutral,N_cation,sym_cation,index\n")
                for s in sorted(st):
                    fh.write(f"{s[0]:.3f},{s[1] / imax:.5f},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]}\n")
        for o in pl["obs"]:
            ax.axvline(o, color="k", lw=0.6, alpha=0.6)
        ax.set_xlim(min(pl["obs"]) - 8, max(pl["obs"]) + 8)
        ax.set_title(f"{name}: simulated PFI-ZEKE spectra (FWHM 0.55 cm$^{{-1}}$); black: measured line positions")
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("wave number / cm$^{-1}$")
    fig.tight_layout()
    fig.savefig(fd / "methane_cation_spectra.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, name, mk in zip(axes, ("CH4+", "CD4+"), ("o", "s")):
        pl = res["effective_hamiltonian"][name]["_plot"]
        eh = res["effective_hamiltonian"][name]
        ax.plot(pl["obs"], pl["obs"] - pl["calc_ai"], mk, mfc="none",
                label=f"ab initio (rms {eh['prediction_no_fit']['rms_cm']:.2f})")
        ax.plot(pl["obs"], pl["obs"] - pl["calc_up"], mk,
                label=f"after Bayesian update (rms {eh['bayesian_update']['rms_cm']:.2f})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("line position / cm$^{-1}$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("obs - calc / cm$^{-1}$")
    fig.tight_layout()
    fig.savefig(fd / "methane_cation_residuals.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
