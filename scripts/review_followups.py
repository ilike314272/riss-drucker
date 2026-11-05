#!/usr/bin/env python3
"""Follow-ups to the method review (docs/case_studies/review_followups.md), items 1-5.

  python scripts/review_followups.py all --data DATA_DIR --pdf JCP_126_144305.pdf [--llm] [--quick]
  python scripts/review_followups.py provenance --data DATA_DIR
  python scripts/review_followups.py coverage   --data DATA_DIR --pdf PAPER.pdf
  python scripts/review_followups.py baselines  --pdf PAPER.pdf [--llm]
  python scripts/review_followups.py splits     --data DATA_DIR --pdf PAPER.pdf
  python scripts/review_followups.py budget     [--quick]

DATA_DIR holds the catalogue downloads: ch3d_jpl/c017003.cat, h2o2_jpl/c034004.cat, h2co_cdms/c030501.cat
(CH3D is fetched when missing). Results go to docs/case_studies/review_followups.json (merged per item)
and a generated summary docs/case_studies/review_followups_results.md.

1 provenance  How many catalogue lines are measured; levels from measured lines only (network inversion,
              one catalogue E'' anchor per component) next to the catalogue levels.
2 coverage    Held-out predictions against their predicted uncertainty: fraction within 1 and 2 sigma
              (68% / 95% if calibrated). CH3D: Fisher sigma_pred of the J > 12 extrapolation and of
              whole held-out K stacks. CH4+: leave-one-position-out residuals against the per-isotopologue
              line sigma estimated from the free fit.
3 baselines   CH4+: the operator dictionary without a language model (every symmetry-allowed first- and
              second-order rotation-tunneling term, single additions and greedy forward selection to
              the same number of extra terms as Claude's best), scored exactly like Claude's proposals,
              against experiment and against the first-principles levels. Hidden-truth synthetic
              molecule (water with undisclosed distortion terms): dictionary baseline and, with --llm,
              Claude through the generic LevelsAdapter.
4 splits      CH3D: hold out one whole K stack at a time. CH4+: hold out one N+ group at a time;
              per-isotopologue line sigma from the free fit (chi2/dof = 1) instead of a fixed 0.5.
5 budget      Distribution of the first-principles delta: CCSD(T) correction x0 / x1 / x2 (it is applied
              along the path only), ZPE of the remaining modes x0.9 / x1.1, harmonic valley vs ab initio
              plane, cc-pVDZ vs cc-pVTZ; UHF <S^2> at the minimum and saddle (spin contamination).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs" / "case_studies" / "review_followups.json"
ISOS = (("CH4+", "H"), ("CD4+", "D"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def coverage(z):
    z = np.abs(np.asarray(z, float))
    return {"n": int(len(z)), "within_1sigma": float(np.mean(z < 1)), "within_2sigma": float(np.mean(z < 2)),
            "rms_z": float(np.sqrt(np.mean(z ** 2))) if len(z) else None,
            "calibrated_would_be": {"within_1sigma": 0.683, "within_2sigma": 0.954, "rms_z": 1.0}}


# ----------------------------------------------------------------------------- 1 provenance
def item_provenance(a):
    from riss_drucker.data.catalog import levels_from_cat, levels_from_measured, read_cat
    data = Path(a.data)
    out = {}
    for label, f in (("CH3D (JPL 17003)", "ch3d_jpl/c017003.cat"), ("H2O2 (JPL 34004)", "h2o2_jpl/c034004.cat"),
                     ("H2CO (CDMS 30501)", "h2co_cdms/c030501.cat")):
        p = data / f
        if not p.exists():
            out[label] = {"missing": str(p)}
            continue
        lines = read_cat(p)
        r = {"n_lines": len(lines), "n_measured": sum(l.measured for l in lines)}
        try:
            lv, rep = levels_from_cat(lines, max_spread=1.0)
            r["catalogue_levels"] = rep["n_levels"]
        except ValueError as e:
            r["catalogue_levels_error"] = str(e)
        if r["n_measured"]:
            lm, repm = levels_from_measured(lines)
            common = [k for k in lm if k in lv]
            d = np.array([lm[k][0] - lv[k][0] for k in common])
            r["measured_network"] = {k: repm[k] for k in ("n_levels", "n_components", "component_sizes",
                                                         "n_unconfirmed", "outliers", "provenance")}
            r["measured_minus_catalogue_cm"] = {"n": len(common), "rms": float(np.sqrt(np.mean(d ** 2))),
                                                "max_abs": float(np.max(np.abs(d)))}
        r["verdict"] = ("no measured lines: every level is catalogue (model) output -> model-reduction test only"
                        if not r["n_measured"] else
                        f"{r['n_measured']} measured lines give {r['measured_network']['n_levels']} levels from "
                        "measurements (network); the remaining catalogue levels are model output")
        out[label] = r
        log("provenance", label, r["verdict"])
    return out


# ----------------------------------------------------------------------------- 2 + 4 CH3D
def _ch3d(a):
    import case_ch3d as C
    return C, C.load_levels(Path(a.data) / "ch3d_jpl")


def item_coverage(a):
    out = {}
    C, (levels, rep) = _ch3d(a)
    r = C.run_ch3d(levels, rep)
    h = r["holdout"]
    out["CH3D J>12 extrapolation (Fisher sigma_pred)"] = coverage([x["residual"] / max(x["sigma_pred"], 1e-30) for x in h])
    out["CH3D J>12 extrapolation (sigma_pred (+) sigma_obs)"] = coverage(
        [x["residual"] / np.hypot(x["sigma_pred"], x["sigma_obs"]) for x in h])
    out["CH3D J>12 median sigma_pred / median |residual|"] = float(np.median([x["sigma_pred"] for x in h]) /
                                                                     np.median([abs(x["residual"]) for x in h]))
    log("coverage CH3D", out["CH3D J>12 extrapolation (Fisher sigma_pred)"])
    if a.pdf:
        sig = per_isotopologue_sigma(a)
        L, tables, facts = _ch4(a)
        for name, _ in ISOS:
            ref = L.reference_proposal()
            cand = L.Candidate(ref["spec"], ref["roles"])
            import case_methane_cation_rovibronic as R
            rows = [x for x in tables[name] if x["N"] and "/" not in x["N"]]
            ev = L.evaluate(cand, name, facts, rows, R.NEUTRAL_B[name], R.PAPER[name]["IE"], cv=True)
            res = np.array(ev["cv_residuals"])
            out[f"{name} leave-one-position-out, sigma 0.5 (as used)"] = coverage(res / 0.5)
            out[f"{name} leave-one-position-out, sigma from free fit ({sig[name]['sigma_line']:.2f})"] = \
                coverage(res / sig[name]["sigma_line"])
            log("coverage", name, out[f"{name} leave-one-position-out, sigma 0.5 (as used)"])
    return out


def per_isotopologue_sigma(a):
    import case_methane_cation_rovibronic as R
    tables = R.read_tables(Path(a.pdf))
    model = R.Model(5)
    out = {}
    for name in ("CH4+", "13CH4+", "CD4+"):
        f = min((R.fit_species(name, tables[name], model, s) for s in (1.0, -1.0)), key=lambda x: x["rms_cm"])
        n, p = f["n_lines"], 5
        s = f["rms_cm"] * np.sqrt(n / (n - p))
        out[name] = {"n_lines": n, "free_fit_rms": f["rms_cm"], "sigma_line": float(s),
                     "chi2_per_dof_at_0.5": float((f["rms_cm"] / 0.5) ** 2 * n / (n - p))}
    return out


def item_splits(a):
    out = {}
    C, (levels, rep) = _ch3d(a)
    Ks = sorted({abs(K) for (_, K) in levels})
    rows = []
    for K0 in Ks[:10]:
        try:
            r = C.run_ch3d(levels, rep, holdout=lambda J, K, K0=K0: abs(K) == K0, autobuild_on=True)
        except Exception as e:
            rows.append({"held_out_K": K0, "error": str(e)[:200]})
            continue
        h = r["holdout"]
        res = np.array([x["residual"] for x in h])
        z = [x["residual"] / max(np.hypot(x["sigma_pred"], x["sigma_obs"]), 1e-30) for x in h]
        rows.append({"held_out_K": K0, "n_held": len(h), "n_terms": len(r["terms"]), "rms_cm": float(np.sqrt(np.mean(res ** 2))),
                     "max_abs_cm": float(np.max(np.abs(res))), "coverage": coverage(z)})
        log("split CH3D K stack", K0, rows[-1].get("rms_cm"))
    out["CH3D hold out one K stack (all J)"] = rows
    if a.pdf:
        out["CH4+ per-isotopologue line sigma (free 5-constant fit, chi2/dof = 1)"] = per_isotopologue_sigma(a)
        L, tables, facts = _ch4(a)
        import case_methane_cation_rovibronic as R
        props = {"reference": L.reference_proposal()}
        props.update(_claude_proposals(L, "experiment"))
        res = {}
        for lab, prop in props.items():
            cand = L.Candidate(prop["spec"], prop["roles"])
            res[lab] = {}
            for name, _ in ISOS:
                rows = [x for x in tables[name] if x["N"] and "/" not in x["N"]]
                Bn, IE0 = R.NEUTRAL_B[name], R.PAPER[name]["IE"]
                err = []
                for Ng in sorted({x["N"] for x in rows}):
                    fit = [x for x in rows if x["N"] != Ng]
                    test = [x for x in rows if x["N"] == Ng]
                    ev = L.evaluate(cand, name, facts, fit, Bn, IE0, cv=False)
                    IE, co = ev["posterior"]["IE"], ev["posterior_coefficients"]
                    asg = L.assign(cand, IE, co, test, Bn)
                    err += list(np.array([x["nu"] for x in test]) - L.calc(cand, IE, co, test, Bn, asg))
                res[lab][name] = {"leave_one_N_group_out_rms": float(np.sqrt(np.mean(np.square(err)))), "n": len(err)}
            log("split CH4+", lab, res[lab])
        out["CH4+ leave one N+ group out"] = res
    return out


# ----------------------------------------------------------------------------- 3 baselines
def _ch4(a):
    import llm_build_ch4plus as L
    import case_methane_cation_rovibronic as R
    tables = R.read_tables(Path(a.pdf))
    facts = L.abinitio_facts(ROOT / "data" / "abinitio" / f"ch4plus_{a.basis}")
    return L, tables, facts


def _claude_proposals(L, mode):
    """Claude's proposals from the committed transcripts (experiment run, cc-pVTZ; first-principles run)."""
    out = {}
    files = {"experiment": ["ch4plus_llm_transcript_ccpvtz.json", "ch4plus_llm_transcript_ccpvtz_r2.json"],
             "firstprinciples": ["ch4plus_llm_fp_transcript_ccpvtz.json"]}[mode]
    for f in files:
        p = ROOT / "docs" / "case_studies" / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rounds"]:
            try:
                prop = L.parse_answer(r["response"])
                prop["spec"]["terms"] = [t for t in prop["spec"]["terms"] if not t.startswith("dBbar")]
                prop["roles"].pop("dBbar", None)
                L.Candidate(prop["spec"], prop["roles"])
                out[f"claude {f.split('_')[-1].split('.')[0]} round {r['round']}"] = prop
            except Exception:
                continue
    return out


def item_baselines(a):
    from riss_drucker import algebra as A
    out = {}
    if a.pdf:
        L, tables, facts = _ch4(a)
        ref = L.reference_proposal()
        cand_ref = L.Candidate(ref["spec"], ref["roles"])
        pool = []
        for deg in ((1, 1), (2, 2)):
            pool += A.enumerate_terms(cand_ref.spec, A.multidegrees(cand_ref.spec, rot=deg), prefix=f"b{deg[0]}").terms
        exp = L.ExperimentAdapter(facts, tables, cv=True)
        adapters = {"experiment (mean CV rms, cm-1)": exp}
        fpf = ROOT / "docs" / "case_studies" / f"ch4plus_firstprinciples_{a.basis}.json"
        if fpf.exists():
            adapters["first principles (mean predicted rms N+>5, cm-1)"] = L.FirstPrinciplesAdapter(
                facts, json.loads(fpf.read_text()), tables)
        silent = lambda *x, **k: None
        for lab, ad in adapters.items():
            def score(terms):
                prop = {"spec": dict(ref["spec"], terms=ref["spec"]["terms"] + terms),
                        "roles": dict(ref["roles"], **{t.split(":")[0].strip(): "extra" for t in terms})}
                try:
                    return ad.evaluate(prop, silent)[1]["score"]
                except Exception as e:
                    return None
            base = score([])
            singles = {t: score([t]) for t in pool}
            log("baseline singles", lab, {k.split(':')[0]: v for k, v in singles.items()})
            # greedy forward selection to 4 extra terms (Claude's richest model has 4)
            chosen, path = [], []
            for _ in range(4 if not a.quick else 2):
                best = None
                for t in pool:
                    if t in chosen:
                        continue
                    s = score(chosen + [t])
                    if s is not None and (best is None or s < best[1]):
                        best = (t, s)
                if best is None:
                    break
                chosen.append(best[0])
                path.append({"terms": list(chosen), "score": best[1]})
                log("baseline greedy", lab, len(chosen), best)
            claude = {}
            for k, prop in _claude_proposals(L, "experiment").items():
                try:
                    claude[k] = ad.evaluate(prop, silent)[1]["score"]
                except Exception as e:
                    claude[k] = f"rejected: {str(e)[:80]}"
            out[f"CH4+ {lab}"] = {"reference": base, "dictionary_single_additions": singles,
                                 "dictionary_greedy_forward": path, "claude_proposals_same_scoring": claude}
    out["hidden-truth synthetic molecule"] = synthetic(a)
    return out


H2O = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n"


def synthetic(a):
    """Water-like asymmetric top whose levels contain undisclosed quartic/sextic terms."""
    from riss_drucker import algebra as A, llm_loop as LL
    from riss_drucker.build import compile_build
    from riss_drucker.llm_adapters import LevelsAdapter
    from riss_drucker.molecule import Molecule
    mol = Molecule.from_xyz(H2O)
    p = mol.plan(nmax=0, J=list(range(10)))
    spec = p["spec"]
    q4 = A.enumerate_terms(spec, A.multidegrees(spec, rot=(4, 4)), prefix="h").terms
    q6 = A.enumerate_terms(spec, A.multidegrees(spec, rot=(6, 6)), prefix="s").terms
    rng = np.random.default_rng(7)
    hidden = [q4[1], q4[3], q6[2]]
    hv = {hidden[0].split(":")[0]: -1.3e-3, hidden[1].split(":")[0]: 4.0e-4, hidden[2].split(":")[0]: 2.0e-6}
    sp = spec.with_terms(list(spec.terms) + hidden)
    bm = compile_build(sp)
    th = dict(p["theta"], **hv)
    rows = []
    for b, (w, _) in zip(bm.blocks, bm.eig(th)):
        rows += [{"J": b.two_j / 2, "irrep": b.irrep, "E": float(e + rng.normal(0, 1e-5))} for e in w]
    e0 = min(r["E"] for r in rows)
    target = [dict(r, E=r["E"] - e0) for r in rows]
    ad = LevelsAdapter(mol, target, jfit=6, plan_kw={"nmax": 0})
    ref = ad.reference()
    silent = lambda *x, **k: None
    pool = q4 + q6

    def score(extra):
        prop = {"spec": {"spaces": ref["spec"]["spaces"], "terms": list(ref["spec"]["terms"]) + extra}}
        try:
            return ad.evaluate(prop, silent)[1]["score"]
        except Exception:
            return None
    chosen, path = [], []
    for _ in range(4):
        best = None
        for t in pool:
            if t in chosen:
                continue
            s = score(chosen + [t])
            if s is not None and (best is None or s < best[1]):
                best = (t, s)
        if best is None:
            break
        chosen.append(best[0])
        path.append({"terms": [c.split(":")[0] for c in chosen], "predict_rms": best[1]})
    res = {"hidden_terms": hidden, "noise_cm": 1e-5, "jfit": 6, "reference_predict_rms": score([]),
           "truth_predict_rms": score(hidden), "dictionary_greedy_forward": path}
    if a.llm:
        tr = ROOT / "docs" / "case_studies" / "synthetic_llm_transcript.json"
        run = LL.run_loop(ad, lambda bl: LL.claude_cli_complete(ad.system, a.model, budget_left=bl), rounds=a.rounds,
                          transcript=str(tr), max_budget_usd=a.budget)
        label, cand, sc, prop = run["best"]
        res["claude"] = {"selected": label, "predict_rms": sc["score"], "terms": cand["spec"].terms if label != "reference" else None,
                         "usage": run["usage"], "stop_reason": run["stop_reason"]}
    log("synthetic", {k: v for k, v in res.items() if k != "hidden_terms"})
    return res


# ----------------------------------------------------------------------------- 5 budget
def item_budget(a):
    from riss_drucker.abinitio import firstprinciples as FP
    from riss_drucker.abinitio.rpsurface import load_stages
    hs = (0.014, 0.010) if a.quick else (0.014, 0.010, 0.007)

    def cc(f):
        def m(rs):
            rs.V = rs._vmp2 + f * (rs.V - rs._vmp2)
        return m

    def zpe(f):
        def m(rs):
            s, z = rs._zpe
            rs._zpe = (s, f * z)
        return m
    out = {}
    for basis in ("ccpvdz", "ccpvtz"):
        d = ROOT / "data" / "abinitio" / f"ch4plus_{basis}"
        if not d.exists():
            continue
        st = load_stages(d)
        variants = {"base": dict(), "CCSD(T) correction x0 (MP2)": dict(modify=cc(0.0)),
                    "CCSD(T) correction x2": dict(modify=cc(2.0)), "ZPE(remaining modes) x0.9": dict(modify=zpe(0.9)),
                    "ZPE(remaining modes) x1.1": dict(modify=zpe(1.1)), "harmonic valley (no plane grid)": dict(use_grid=False)}
        for name, iso in ISOS:
            row = {}
            for lab, kw in variants.items():
                ta = FP.tunneling_analysis(st, iso, hs=hs, **kw)
                row[lab] = ta["delta"]
                log("budget", basis, name, lab, round(ta["delta"], 3))
            vals = np.array(list(row.values()))
            row["range"] = [float(vals.min()), float(vals.max())]
            out[f"{basis} {name}"] = row
    out["spin contamination"] = spin_contamination(a)
    return out


def spin_contamination(a):
    try:
        from pyscf import gto, scf
    except ImportError:
        return {"skipped": "PySCF not installed"}
    out = {}
    for basis in ("ccpvdz", "ccpvtz") if not a.quick else ("ccpvdz",):
        d = ROOT / "data" / "abinitio" / f"ch4plus_{basis}"
        if not d.exists():
            continue
        for stg in ("min", "ts"):
            X = np.array(json.loads((d / f"ch4plus_{stg}.json").read_text())["xyz"])
            mol = gto.M(atom=[[s, x] for s, x in zip(["C", "H", "H", "H", "H"], X)], basis=basis.replace("ccp", "cc-p"),
                        charge=1, spin=1, unit="Angstrom", verbose=0)
            mf = scf.UHF(mol).run()
            s2, _ = mf.spin_square()
            out[f"{basis} {stg}"] = {"S2": float(s2), "ideal": 0.75}
            log("<S^2>", basis, stg, round(s2, 5))
    return out


# ----------------------------------------------------------------------------- driver
def write_summary(res):
    res = {k: v for k, v in res.items() if isinstance(v, dict) and "error" not in v}
    L = ["# Method-review follow-ups: results", "",
         "Generated by `scripts/review_followups.py`; numbers in `review_followups.json`.", ""]
    if "provenance" in res:
        L += ["## 1 Data provenance", "", "| Catalogue | lines | measured | verdict |", "|---|---|---|---|"]
        for k, v in res["provenance"].items():
            if not isinstance(v, dict):
                continue
            L.append(f"| {k} | {v.get('n_lines', '—')} | {v.get('n_measured', '—')} | {v.get('verdict', v.get('missing', ''))} |")
        L.append("")
    if "coverage" in res:
        L += ["## 2 Coverage of held-out predictions", "", "| Prediction | n | within 1σ | within 2σ | rms z |",
              "|---|---|---|---|---|"]
        for k, v in res["coverage"].items():
            if isinstance(v, dict) and "within_1sigma" in v:
                L.append(f"| {k} | {v['n']} | {v['within_1sigma']:.2f} | {v['within_2sigma']:.2f} | {v['rms_z']:.3g} |")
        L += ["", "Calibrated: 0.68, 0.95, 1.", ""]
    if "baselines" in res:
        L += ["## 3 Baselines", ""]
        for k, v in res["baselines"].items():
            L += [f"### {k}", "", "```", json.dumps(v, indent=1, default=str)[:4000], "```", ""]
    if "splits" in res:
        L += ["## 4 Splits", "", "```", json.dumps(res["splits"], indent=1, default=str)[:6000], "```", ""]
    if "budget" in res:
        L += ["## 5 Uncertainty budget for δ (cm⁻¹)", ""]
        for k, v in res["budget"].items():
            if k == "spin contamination":
                L += [f"Spin contamination: {json.dumps(v)}", ""]
                continue
            L.append(f"- **{k}**: " + ", ".join(f"{lab} {val:.3f}" if isinstance(val, float) else f"{lab} {val}"
                                                 for lab, val in v.items()))
        L.append("")
    (OUT.parent / "review_followups_results.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=["provenance", "coverage", "baselines", "splits", "budget", "all"])
    ap.add_argument("--data", default=str(ROOT / "data" / "catalogues"))
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--basis", default="ccpvtz", help="ab initio surface for the CH4+ items (ccpvdz or ccpvtz)")
    ap.add_argument("--llm", action="store_true", help="also run Claude on the hidden-truth synthetic molecule")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--budget", type=float, default=2.0, help="USD cap for --llm")
    ap.add_argument("--quick", action="store_true", help="coarser grids / fewer steps (for testing)")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    items = ["provenance", "coverage", "baselines", "splits", "budget"] if a.item == "all" else [a.item]
    fn = {"provenance": item_provenance, "coverage": item_coverage, "baselines": item_baselines,
          "splits": item_splits, "budget": item_budget}
    for it in items:
        t0 = time.time()
        log("==== item", it)
        try:
            res[it] = fn[it](a)
        except Exception as e:
            res[it] = {"error": f"{type(e).__name__}: {e}"}
            log("item", it, "failed:", e)
        res.setdefault("_meta", {})[it] = {"seconds": round(time.time() - t0, 1), "date": time.strftime("%F %T"),
                                           "basis": a.basis, "quick": a.quick}
        OUT.write_text(json.dumps(res, indent=1, default=float))
        write_summary(res)
    print(json.dumps({k: (v if k == "_meta" else "see review_followups.json") for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
