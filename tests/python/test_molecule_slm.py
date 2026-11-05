"""Any molecule -> analysis -> certified plan -> (data) -> autobuild with the symbolic LM."""
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import autobuild as AB, build as B
from riss_drucker.molecule import Molecule, run
from riss_drucker.symbolic_lm import SymbolicLM, term_features

H2O = "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"
NH3 = "N 0 0 0.1162\nH 0 0.9377 -0.2712\nH 0.8121 -0.4689 -0.2712\nH -0.8121 -0.4689 -0.2712"
CH4 = "C 0 0 0\nH 0.6276 0.6276 0.6276\nH 0.6276 -0.6276 -0.6276\nH -0.6276 0.6276 -0.6276\nH -0.6276 -0.6276 0.6276"


def benzene():
    c = [f"C {1.39 * math.cos(k * math.pi / 3):.5f} {1.39 * math.sin(k * math.pi / 3):.5f} 0" for k in range(6)]
    h = [f"H {2.47 * math.cos(k * math.pi / 3):.5f} {2.47 * math.sin(k * math.pi / 3):.5f} 0" for k in range(6)]
    return "\n".join(c + h)


def test_point_groups_gamma_vib_and_spin_weights():
    w = Molecule.from_xyz(H2O)
    assert w.schoenflies() == "C2v" and w.rotor_type() == "asymmetric"
    assert sum(w.gamma_vib().values()) == 3 and w.gamma_vib()["A1"] == 2
    sw = w.spin_weights()
    assert sorted(sw.values()) == [1, 1, 3, 3]                    # para : ortho = 1 : 3
    n = Molecule.from_xyz(NH3)
    assert n.schoenflies() == "C3v" and n.gamma_vib() == {"A1": 2, "E": 2}
    c = Molecule.from_xyz(CH4)
    assert c.schoenflies() == "Td" and c.gamma_vib() == {"A1": 1, "E": 1, "F2": 2}
    assert c.spin_weights() == {"A1": 5, "A2": 5, "E": 2, "F1": 3, "F2": 3}
    b = Molecule.from_xyz(benzene())
    assert b.schoenflies() == "D6h"
    assert b.gamma_vib() == {"A1g": 2, "A2g": 1, "B2g": 2, "E1g": 1, "E2g": 4, "A2u": 1, "B1u": 2, "B2u": 2,
                             "E1u": 3, "E2u": 2}
    assert Molecule.from_xyz("C 0 0 0\nO 0 0 1.16\nO 0 0 -1.16").schoenflies() == "D∞h"


def test_plan_compiles_and_reproduces_rigid_rotor_from_geometry():
    w = Molecule.from_xyz(H2O)
    p = w.plan(frequencies={"A1": [1595.0, 3657.0], "B1": [3756.0]}, J=[0, 1, 2, 3], polyad_max=1)
    bm = B.compile_build(p["spec"])
    assert all(c["passed"] for c in bm.certificates)
    rc = w.rotational_constants()
    m = rd.compile_asym_rotor(j_max=3, order=2)
    A_, B_, C_ = sorted(rc.values(), reverse=True)
    ref = np.sort(rd.solve(m, {"A": A_, "B": B_, "C": C_}, window=(-1e9, 1e9), method="dense").energies())
    ours = np.sort(np.concatenate([w_ for b, (w_, _) in zip(bm.blocks, bm.eig(p["theta"])) if b.polyad == 0]))
    np.testing.assert_allclose(ours, ref, atol=1e-8)
    # serialization of the detected group: same hash, same irrep names
    s1 = rd.ModelSpec.built(p["spec"])
    s2 = rd.ModelSpec.built(B.BuildSpec.from_dict(p["spec"].to_dict()))
    assert s1.hash() == s2.hash()
    assert set(B.BuildSpec.from_dict(p["spec"].to_dict()).group.chars) == set(p["spec"].group.chars)


def test_plans_for_symmetric_and_spherical_tops_and_open_shell():
    for xyz, kw in ((NH3, {}), (CH4, {})):
        p = Molecule.from_xyz(xyz).plan(J=[0, 1, 2], nmax=1)
        bm = B.compile_build(p["spec"])
        assert all(c["passed"] for c in bm.certificates)
    # a doublet radical (CH3-like planar, multiplicity 2): half-integer J, double group, spin space
    ch3 = Molecule.from_xyz("C 0 0 0\nH 1.079 0 0\nH -0.5395 0.9344 0\nH -0.5395 -0.9344 0", multiplicity=2)
    assert ch3.schoenflies() == "D3h"
    p = ch3.plan(J=[0.5, 1.5], nmax=1)
    bm = B.compile_build(p["spec"])
    assert all(c["passed"] for c in bm.certificates) and any(s.kind == "spin" for s in p["spec"].spaces)


def test_symbolic_lm_features_and_generation_prefers_resonances():
    f = term_features("k: a1+ a2 a2 + h.c.", frequencies={"1": 3000.0, "2": 1495.0})
    assert f["resonance"] == 1 and f["detuning"] < 0.01
    g = term_features("k: a1+ a2 + h.c.", frequencies={"1": 3000.0, "2": 1495.0})
    lm = SymbolicLM()
    assert lm.logit(f) > lm.logit(g)                       # near-resonant beats detuned
    modes = [B.Mode("1", "A", 2), B.Mode("2", "A", 1)]
    spec = B.BuildSpec([B.VibSpace(modes, polyads=[0, 1, 2, 3])], "C1", ["w1: n1", "w2: n2"])
    from riss_drucker import algebra as A
    top = lm.generate(spec, A.multidegrees(spec, vib=(3, 4)), top=3, frequencies={"1": 3000.0, "2": 1495.0})
    assert any("a1+ a2 a2" in t or "a2+ a2+ a1" in t for t in top)


def test_run_any_molecule_with_data_grows_the_model():
    w = Molecule.from_xyz(H2O)
    p = w.plan(J=[0, 1, 2, 3, 4], with_rotation=True, nmax=0)
    truth_terms = p["spec"].terms + ["DK: sym(Jz Jz Jz Jz) + h.c."]
    truth = B.compile_build(p["spec"].with_terms(truth_terms))
    th = {**p["theta"], "DK": -2e-3}
    th = {k: v for k, v in th.items() if k in truth.param_names}
    rng = np.random.default_rng(2)
    data = [AB.LevelDatum(b.two_j / 2, b.irrep, i, e + 1e-4 * rng.standard_normal(), 1e-4, b.polyad)
            for b, (ws, _) in zip(truth.blocks, truth.eig(th)) for i, e in enumerate(ws)]
    out = run(w, data, J=[0, 1, 2, 3, 4], nmax=0, max_steps=3)
    res = out["result"]
    assert any(s.accepted for s in res.steps)
    assert res.fit.reduced_chi2 < 3.0
