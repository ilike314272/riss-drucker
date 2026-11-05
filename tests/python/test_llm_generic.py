"""Generic (molecule-independent) language-model loop: LevelsAdapter + run_loop with replayed answers."""
import json

import numpy as np

H2O = """3
water
O 0.000000 0.000000 0.117300
H 0.000000 0.757200 -0.469200
H 0.000000 -0.757200 -0.469200
"""


def _target(mol, dk=-0.0015):
    """Synthetic levels: the plan's rigid rotor plus a quartic distortion term."""
    from riss_drucker.build import compile_build
    p = mol.plan(nmax=0, J=list(range(8)))
    spec = p["spec"].with_terms(list(p["spec"].terms) + ["dk: Jz Jz Jz Jz"])
    th = dict(p["theta"], dk=dk)
    bm = compile_build(spec)
    rows = []
    for b, (w, _) in zip(bm.blocks, bm.eig(th)):
        rows += [{"J": b.two_j / 2, "irrep": b.irrep, "E": float(e)} for e in w]
    e0 = min(r["E"] for r in rows)
    return [dict(r, E=r["E"] - e0) for r in rows], p


def test_levels_adapter_selects_the_term_that_extrapolates(tmp_path):
    from riss_drucker import llm_loop as LL
    from riss_drucker.llm_adapters import LevelsAdapter
    from riss_drucker.molecule import Molecule
    mol = Molecule.from_xyz(H2O)
    target, p = _target(mol)
    ad = LevelsAdapter(mol, target, jfit=4, plan_kw={"nmax": 0})
    ref = ad.reference()
    plus = json.loads(json.dumps(ref, default=str))
    plus["spec"]["terms"] = list(plus["spec"]["terms"]) + ["dk: Jz Jz Jz Jz"]
    bad = json.loads(json.dumps(ref, default=str))
    bad["spec"]["terms"] = list(bad["spec"]["terms"]) + ["dk: Jz Jz Jz"]          # time-reversal odd: refused
    answers = ["```json\n" + json.dumps({"spec": {"spaces": bad["spec"]["spaces"], "terms": bad["spec"]["terms"]}}) + "\n```",
               "```json\n" + json.dumps({"spec": {"spaces": plus["spec"]["spaces"], "terms": plus["spec"]["terms"]},
                                         "initial": {"dk": 0.0}}) + "\n```"]
    rp = tmp_path / "replay.json"
    rp.write_text(json.dumps({"rounds": [{"response": a} for a in answers]}))
    ad.prompt([])                                         # the prompt builds for any molecule
    run = LL.run_loop(ad, lambda bl: LL.replay_complete(rp), rounds=2, transcript=str(tmp_path / "t.json"),
                      log=lambda *a, **k: None)
    label, cand, sc, prop = run["best"]
    assert "error" in run["rounds"][0] and "time" in run["rounds"][0]["error"].lower()
    assert label == "llm round 2"
    assert sc["predict_rms_cm"] < 1e-4 < run["reference"][1]["predict_rms_cm"]
    assert abs(sc["coefficients"]["dk"] + 0.0015) < 1e-6
