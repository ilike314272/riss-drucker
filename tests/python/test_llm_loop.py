"""Language-model-in-the-loop CH4+ builder (scripts/llm_build_ch4plus.py), without calling any model."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
DZ = ROOT / "data" / "abinitio" / "ch4plus_ccpvdz"


@pytest.fixture(scope="module")
def L():
    import llm_build_ch4plus
    return llm_build_ch4plus


def _alternative(ref):
    """Same physics, other parameter names and operator normalization."""
    alt = json.loads(json.dumps(ref))
    alt["spec"]["terms"] = ["a: 2 sym(w[1,1] Jx Jx)", "b: sym(w[1,1] Jy Jy)", "c: sym(w[1,1] (Jy Jz + Jz Jy))",
                            "t: sym(w[1,3] + w[3,1])"]
    alt["roles"] = {"a": "rotor", "b": "rotor", "c": "rotor", "t": "tunneling"}
    return alt


def test_calibration_is_normalization_free_and_geometry_constrained(L):
    facts = L.abinitio_facts(DZ)
    st = facts["structure"]
    assert st["minimum_point_group"] == "C2v" and st["saddle_point_group"] == "Cs" and st["n_equivalent_minima"] == 6
    assert facts["isotopologues"]["CH4+"]["constant_about_C2_axis"] == "A"
    ref = L.reference_proposal()
    rc = facts["isotopologues"]["CH4+"]["rigid_rotor_C2v_minimum_cm"]
    abc = np.array([rc["A"], rc["B"], rc["C"]])
    out = []
    for prop in (ref, _alternative(ref)):
        c = L.Candidate(prop["spec"], prop["roles"])
        rms, branches = c.calibrate(*abc)
        assert rms < 1e-8 and len(branches) == 6               # 3! axis assignments, distinct once tunneling is on
        axis = c.well_c2_axis()
        phys = []
        for br in branches:
            vals, vecs = c.well_quadratic_form(br["coeffs"])
            assert np.allclose(np.sort(vals)[::-1], abc, atol=1e-6)     # well-1 rotor = ab initio rigid rotor
            k = int(np.argmax(np.abs(vecs.T @ axis)))
            if abs(vals[k] - abc[0]) < 1e-4:
                phys.append(br)
        assert len(phys) == 2                                   # A on the C2 axis; B/C swap left to the data
        out.append((c, [b["coeffs"] | {c.tun: 1.7 / c.hop} for b in phys]))
    # the two proposals give the same set of spectra on their physical branches
    (c1, k1), (c2, k2) = out
    for a in k1:
        assert any(L.equivalent(c1, a, c2, b) for b in k2)


def test_rejections_are_explanatory(L):
    ref = L.reference_proposal()
    with pytest.raises(ValueError, match="no JSON object"):
        L.parse_answer("A rigid rotor should do.")
    bad = json.loads(json.dumps(ref))
    bad["roles"] = {"rA": "rotor", "rBC": "rotor", "rD": "rotor", "sig": "rotor"}
    with pytest.raises(ValueError, match="tunneling"):
        L.Candidate(bad["spec"], bad["roles"])
    odd = json.loads(json.dumps(ref))
    odd["spec"]["terms"][0] = "rA: sym(w[1,1] Jx)"            # time-reversal odd: the compiler must refuse
    with pytest.raises(Exception):
        L.Candidate(odd["spec"], odd["roles"])


def test_claude_cli_backend_with_fake_executable(L, tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env python3\nimport json, sys\nsys.stdin.read()\n"
                    "assert '-p' in sys.argv and '--tools' in sys.argv\n"
                    "print(json.dumps({'result': '```json\\n{\"spec\": {}, \"roles\": {}}\\n```'}))\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text = L.claude_cli_complete()("prompt")
    assert L.parse_answer(text) == {"spec": {}, "roles": {}}


def test_dry_run_prompt():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "llm_build_ch4plus.py"), "--abinitio", str(DZ),
                          "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    p = json.loads(out.stdout)
    assert "rA:" not in out.stdout                              # the reference model is not shown to the language model
    assert p["ab_initio"]["structure"]["n_equivalent_minima"] == 6 and "answer_format" in p


@pytest.mark.skipif(not os.environ.get("RISS_CH4PLUS_PDF"),
                    reason="needs the PDF of Woerner, Qian & Merkt, JCP 126, 144305 (2007) in RISS_CH4PLUS_PDF")
def test_replayed_loop_rejects_phase_free_model_and_accepts_equivalent(L, tmp_path):
    ref = L.reference_proposal()
    nophase = json.loads(json.dumps(ref))
    nophase["spec"]["spaces"][1]["geometric_phase"] = False
    answers = ["no json here", "```json\n" + json.dumps(nophase) + "\n```", "```json\n" + json.dumps(_alternative(ref)) + "\n```"]
    rp = tmp_path / "replay.json"
    rp.write_text(json.dumps({"rounds": [{"response": r} for r in answers]}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "llm_build_ch4plus.py"), "--abinitio", str(DZ),
                          "--pdf", os.environ["RISS_CH4PLUS_PDF"], "--llm", "replay", "--replay", str(rp),
                          "--rounds", "3", "--no-cv", "--out", str(tmp_path / "m.json"),
                          "--transcript", str(tmp_path / "t.json")], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    tr = json.loads((tmp_path / "t.json").read_text())["rounds"]
    assert "no JSON object" in tr[0]["error"] and "A1 at N+=1" in tr[1]["error"]
    assert tr[2]["equivalent_to_reference"] is True
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["selected"] == "llm round 3"
    assert abs(m["scores"]["CD4+"]["prediction_rms_IE_only_cm"] - 0.536) < 0.01
    assert abs(m["scores"]["CH4+"]["delta_posterior_cm"] - 16.7) < 0.2
