"""Dynamic Hamiltonian builder: groups, product-space compiler, algebraic counts,
redundancy, autobuild, spectra, ML proposers. Synthetic and exact limits only."""
import collections
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import algebra as A, autobuild as AB, build as B, symmetry as S, vib
from riss_drucker.grammar import GrammarError
from riss_drucker.ml import amortized, dataset, neural, structure


# ---- symmetry ---------------------------------------------------------------------------------

def test_character_tables_counts_and_spin_statistics():
    td = S.get_group("Td")
    assert td.order == 24 and sorted(len(c) for c in td.classes) == [1, 3, 6, 6, 8]
    assert [td.level_counts(2 * J) for J in range(4)] == [
        {"A1": 1}, {"F1": 1}, {"E": 1, "F2": 1}, {"A2": 1, "F1": 1, "F2": 1}]
    assert td.spin_statistics([0.5] * 4) == {"A1": 5, "A2": 5, "E": 2, "F1": 3, "F2": 3}   # CH4
    assert td.jahn_teller_active("F2") == {"E": 1, "F2": 1}                              # CH4+: T x (e + t2)
    assert all(td.frobenius_schur(n) == 1 for n in td.chars)
    assert S.get_group("C3v").intertwiners("E", "E", "E").shape == (1, 4, 2)
    g8 = S.get_group("G8(H2O2)")
    assert g8.order == 8 and sum(g8.physical(n) for n in g8.chars) == 4
    w = g8.spin_statistics([0, 0, 0.5, 0.5])
    assert {k: v for k, v in w.items() if g8.physical(k)} == {"A1": 1, "A2": 1, "B1": 3, "B2": 3}
    tdd = S.get_group("Td", double=True)
    assert tdd.order == 48 and tdd.level_counts(1) == {"E1/2": 1} and tdd.level_counts(3) == {"G3/2": 1}
    d2d = S.get_group("D2", double=True)
    assert d2d.frobenius_schur("E1/2") == -1          # quaternionic: Kramers class for odd electrons


# ---- builder: exact limits and cross-checks ---------------------------------------------------

def test_built_asymmetric_rotor_equals_cpp_rotor():
    spec = B.BuildSpec([B.RotSpace()], "D2", ["A: Jz Jz", "Bx: Jx Jx", "Cy: Jy Jy"], J=list(range(6)))
    bm = B.compile_build(spec)
    e1 = np.sort(np.concatenate([w for w, _ in bm.eig({"A": 3.0, "Bx": 1.5, "Cy": 1.0})]))
    m = rd.compile_asym_rotor(j_max=5, order=2)
    e2 = np.sort(rd.solve(m, {"A": 3.0, "B": 1.5, "C": 1.0}, window=(-1e9, 1e9), method="dense").energies())
    np.testing.assert_allclose(e1, e2, atol=1e-10)
    assert all(c["passed"] for c in bm.certificates)


def test_td_coriolis_is_exact_and_counts_match_characters():
    spec = B.BuildSpec([B.RotSpace(), B.VibSpace([B.Mode("3", "F2")], polyads=[0, 1])], "Td",
                       ["w3: n3", "B: J2", "z: sym(i*Jz a3x+ a3y) + h.c."], J=list(range(6)))
    bm = B.compile_build(spec)
    th = {"w3": 3019.0, "B": 5.24, "z": 1.0}
    for (w, _), b in zip(bm.eig(th), bm.blocks):
        if b.polyad != 1:
            continue
        J = b.two_j // 2
        # sym averages the three components: z-term = -(1/3) J.l with J.l in {-J, 1, J+1}
        allowed = {round(3019.0 + 5.24 * J * (J + 1) - x / 3, 8) for x in (-J, 1, J + 1) if J + 1 > 0}
        assert {round(e, 8) for e in w} <= allowed
    td = bm.group
    for c in bm.counts:
        chi = td.rotation_character(c["two_j"])
        if c["polyad"] == 1:
            chi = chi * td.chars["F2"]
        assert td.decompose(chi).get(c["irrep"]) == c["count"]


def test_builder_matches_vib_compiler_on_methane_polyads():
    METH = [B.Mode("1", "A1", 2), B.Mode("2", "E", 1), B.Mode("3", "F2", 2), B.Mode("4", "F2", 1)]
    VM = [vib.Mode(m.name, m.irrep, m.weight) for m in METH]
    terms = ["w1: n1", "w2: n2", "w3: n3", "w4: n4", "k: sym(a1+ a4x a4x) + h.c.", "x44: n4*n4",
             "g: sym(a3x+ a4y a4z) + h.c."]
    bb = B.compile_build(B.BuildSpec([B.VibSpace(METH, polyads=[0, 1, 2, 3])], "Td", terms))
    vm = vib.compile_vib(vib.VibSpec(VM, "Td", [0, 1, 2, 3], terms))
    th = dict(w1=2917, w2=1534, w3=3019, w4=1306, k=-30.0, x44=-5.0, g=12.0)
    u1 = np.unique(np.round(np.concatenate([x for x, _ in bb.eig(th)]), 6))
    u2 = np.unique(np.round(rd.solve(vm, th, window=(-1e9, 1e9), method="dense").energies(), 6))
    np.testing.assert_allclose(u1, u2, atol=1e-6)


def test_jahn_teller_hidden_symmetry_and_its_breaking():
    spec = B.BuildSpec([B.ElecSpace([("1", "E")]), B.VibSpace([B.Mode("2", "E")], nmax=14)], "C3v",
                       ["w: n2", "k: sym(e[1x,1x] q2x)"])
    bm = B.compile_build(spec)
    lev = {b.irrep: w for b, (w, _) in zip(bm.blocks, bm.eig({"w": 1.0, "k": 1.5}))}
    # linear E x e: conserved j = l + 1/2 -> A1 and A2 levels pair up (point group misses it)
    np.testing.assert_allclose(lev["A1"][:6], lev["A2"][:6], atol=1e-6)
    bm2 = B.compile_build(spec.with_terms(spec.terms + ["g: sym(e[1x,1x] q2x q2x)"]))
    lev2 = {b.irrep: w for b, (w, _) in zip(bm2.blocks, bm2.eig({"w": 1.0, "k": 1.5, "g": 0.2}))}
    assert np.min(np.abs(lev2["A1"][:4] - lev2["A2"][:4])) > 1e-3


def test_open_shell_doublet_from_grammar():
    spec = B.BuildSpec([B.RotSpace(), B.SpinSpace(1)], "D2", ["B: N2", "g: Nx Sx + Ny Sy + Nz Sz"],
                       J=[0.5, 1.5, 2.5])
    bm = B.compile_build(spec)
    for b, (w, _) in zip(bm.blocks, bm.eig({"B": 1.0, "g": 0.1})):
        J = b.two_j / 2
        exact = {round(N * (N + 1) + 0.05 * (J * (J + 1) - N * (N + 1) - 0.75), 9) for N in (J - 0.5, J + 0.5)}
        assert {round(e, 9) for e in w} <= exact


def test_h2o2_half_angle_frame_gauge_selects_allowed_couplings():
    base = [B.RotSpace(), B.TorsionSpace("t", 24, period_pi=4)]
    ok = ["F: p[t] p[t]", "V1: cos1[t]", "A: Jz Jz", "BC: Jx Jx + Jy Jy",
          "Dx: Jx (p[t] cos0.5[t] + cos0.5[t] p[t])", "Ga: sin0.5[t] (Jy Jz + Jz Jy)"]
    bm = B.compile_build(B.BuildSpec(base, "G8(H2O2)", ok, J=[0, 1]))
    j0 = [c for c in bm.counts if c["two_j"] == 0]
    assert sum(c["count"] for c in j0) == 25          # K = 0 keeps the 2pi-periodic functions only
    with pytest.raises(GrammarError, match="total_symmetry"):  # K-odd coupling needs a half-integer order
        B.compile_build(B.BuildSpec(base, "G8(H2O2)", ok[:4] + ["Ga: sin1[t] (Jy Jz + Jz Jy)"], J=[0]))
    with pytest.raises(GrammarError, match="hermiticity"):
        B.compile_build(B.BuildSpec(base, "G8(H2O2)", ok[:4] + ["Dx: Jx p[t] cos0.5[t]"], J=[0]))
    # J = 0 equals the 1D torsion problem
    m = rd.compile_torsion_rotor(n_fold=1, v_orders=[1], m_max=12)
    th = {"F": 20.0, "V1": 400.0, "A": 10.0, "BC": 0.85, "Dx": 0.0, "Ga": 0.0}
    e_b = np.sort(np.concatenate([w for b, (w, _) in zip(bm.blocks, bm.eig(th)) if b.two_j == 0]))
    e_t = np.sort(rd.solve(m, {"F": 20.0, "V1": -800.0}, window=(-1e9, 1e9), method="dense").energies()) + 400.0
    np.testing.assert_allclose(e_b[:8], e_t[:8], atol=1e-8)


# ---- algebraic certificates ------------------------------------------------------------------

def test_operator_counts_equal_molien_counts():
    METH = [B.Mode("1", "A1", 2), B.Mode("2", "E", 1), B.Mode("3", "F2", 2), B.Mode("4", "F2", 1)]
    spec = B.BuildSpec([B.VibSpace(METH, polyads=[0, 1, 2, 3])], "Td", [])
    en = A.enumerate_terms(spec, A.multidegrees(spec, vib=(2, 4)))
    assert [d["count"] for d in en.by_degree] == [4, 4, 30] and all(c["passed"] for c in en.certificates)
    td = B.BuildSpec([B.RotSpace()], "Td", [], J=[0])
    assert [d["count"] for d in A.enumerate_terms(td, A.multidegrees(td, rot=(0, 8))).by_degree] == \
        [1, 0, 1, 0, 2, 0, 3, 0, 4]             # spherical top: J^2; J^4, Omega_4; ...
    d2 = B.BuildSpec([B.RotSpace()], "D2", [], J=[0])
    assert [d["count"] for d in A.enumerate_terms(d2, A.multidegrees(d2, rot=(2, 6))).by_degree] == \
        [3, 0, 6, 0, 10]                        # asymmetric top before reduction
    h = B.BuildSpec([B.RotSpace(), B.TorsionSpace("t", 8, 4)], "G8(H2O2)", [], J=[0])
    en = A.enumerate_terms(h, A.multidegrees(h, rot=(0, 2), torsion={"t": (2, 2)}, total=2))
    terms = " ".join(en.terms)
    assert "Jx cos0.5[t] p[t]" in terms and "Jy Jz sin0.5[t]" in terms


def test_contact_transformation_redundancy_reproduces_watson_reduction():
    rs = B.BuildSpec([B.RotSpace()], "D2", [], J=list(range(9)))
    rng = np.random.default_rng(1)
    q6 = A.enumerate_terms(rs, A.multidegrees(rs, rot=(2, 6))).terms
    bm = B.compile_build(rs.with_terms(q6))
    th = dict(zip(bm.param_names, [3.0, 1.5, 1.0] + list(1e-4 * rng.standard_normal(6))
                  + list(1e-7 * rng.standard_normal(10))))
    r = A.redundancy(bm, th, A.multidegrees(rs, rot=(3, 5)))
    assert (r["n_terms"], r["redundant"], r["identifiable_max"]) == (19, 4, 15)   # 3 + 5 + 7 (Watson)


# ---- autobuild --------------------------------------------------------------------------------

def _levels(bm, th, sigma, rng):
    out = []
    for b, (w, _) in zip(bm.blocks, bm.eig(th)):
        for i, e in enumerate(w):
            out.append(AB.LevelDatum(b.two_j / 2, b.irrep, i, e + sigma * rng.standard_normal(), sigma, b.polyad))
    return out


def test_autobuild_finds_quartic_distortion_and_rejects_redundant_term():
    rng = np.random.default_rng(3)
    rs = B.BuildSpec([B.RotSpace()], "D2", [], J=list(range(7)))
    q = A.enumerate_terms(rs, A.multidegrees(rs, rot=(2, 4))).terms
    # a reduced set of 5 quartics; another reduction differs only at sextic order ~ D^2/B J^6 < sigma
    truth = B.compile_build(rs.with_terms(q[:3] + q[3:8]))
    tt = dict(zip(truth.param_names, [1.5, 0.5, 1.5] + list(-3e-5 * (1 + np.abs(rng.standard_normal(5))))))
    data = _levels(truth, tt, 1e-4, rng)
    res = AB.autobuild(rs.with_terms(q[:3]), data, dict(zip(truth.param_names[:3], [1.5, 0.5, 1.5])),
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, rot=(4, 4)))],
                       redundancy_degrees=A.multidegrees(rs, rot=(3, 3)), max_steps=8,
                       new_term_sigma=1e-3)
    acc = [s for s in res.steps if s.accepted]
    # the truth's Jx^2 Jy^2 term is unitarily equivalent to a combination of the others:
    # four quartics reproduce the data, the fifth is rejected as removable
    assert len(acc) == 4 and any("contact transformation" in s.reason and "Jx Jx Jy Jy" in s.term
                                 for s in res.steps)
    assert res.fit.reduced_chi2 < 3.0
    assert res.certificates[-1]["name"] == "fisher_rank_vs_redundancy" and res.certificates[-1]["passed"]


def test_autobuild_discovers_fermi_resonance_from_dictionary():
    rng = np.random.default_rng(0)
    modes = [B.Mode("1", "A", 2), B.Mode("2", "A", 1)]
    sp = [B.VibSpace(modes, polyads=[0, 1, 2, 3, 4])]
    truth = B.compile_build(B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2", "x22: n2*n2", "k: a1+ a2 a2 + h.c."]))
    data = _levels(truth, {"w1": 3000.0, "w2": 1495.0, "x22": -4.0, "k": 30.0}, 0.01, rng)
    spec0 = B.BuildSpec(sp, "C1", ["w1: n1", "w2: n2", "x22: n2*n2"])
    res = AB.autobuild(spec0, data, {"w1": 3000.0, "w2": 1495.0, "x22": -4.0},
                       [AB.DictionaryProposer(lambda s: A.multidegrees(s, vib=(3, 3)))], max_steps=3,
                       new_term_sigma=100.0)
    first = res.steps[0]
    assert first.accepted and "a1+ a2 a2" in first.term.replace("a1 ", "a1").replace("  ", " ") or \
        "a2+ a2+ a1" in first.term
    assert res.fit.reduced_chi2 < 2.0


# ---- spectra ---------------------------------------------------------------------------------

def test_line_strengths_match_rotor_module_and_sum_rule():
    spec = B.BuildSpec([B.RotSpace()], "D2", ["A: Jz Jz", "Bx: Jx Jx", "Cy: Jy Jy"], J=list(range(7)))
    bm = B.compile_build(spec)
    th = {"A": 3.0, "Bx": 1.5, "Cy": 1.0}
    lines = B.line_strengths(bm, th, {"x": 0.6, "y": 0.8})
    # sum rule: sum over upper states of S(f <- i) = (2J+1) |mu|^2 for every lower state with J < Jmax
    tot = collections.defaultdict(float)
    for ln in lines:
        tot[(ln["lower_block"], ln["lower_index"])] += ln["strength"]
    vec = bm.eig(th)
    for b, (w, _) in enumerate(vec):
        J = bm.blocks[b].two_j / 2
        if J >= 5:
            continue
        for i, e in enumerate(w):
            down = sum(l["strength"] for l in lines if l["upper_block"] == b and l["upper_index"] == i)
            assert tot[(b, i)] + down == pytest.approx((2 * J + 1) * 1.0, rel=1e-8)
    m = rd.compile_asym_rotor(j_max=6, order=2)
    sol = rd.solve(m, {"A": 3.0, "B": 1.5, "C": 1.0}, window=(-1e9, 1e9), method="dense", keep_vectors=True)
    ref = sorted((round(l.wavenumber, 8), round(l.strength, 8)) for l in
                 rd.spectra.line_strengths(m, sol, (0.0, 0.6, 0.8))
                 if l.wavenumber > 1e-9 and l.strength > 1e-10)
    ours = sorted((round(l["wavenumber"], 8), round(l["strength"], 8)) for l in lines)
    assert ours == ref


# ---- ML ---------------------------------------------------------------------------------------

def test_neural_posterior_is_calibrated_and_verified(tmp_path):
    m = rd.compile_asym_rotor(j_max=4, order=2)
    prior = dataset.Prior({"A": ("uniform", 2.5, 4.0), "B": ("uniform", 1.6, 2.4), "C": ("uniform", 0.8, 1.4)})
    dataset.generate(m, prior, 600, tmp_path / "tr", seed=1, n_levels=4, n_moments=8, noise=1e-3)
    dataset.generate(m, prior, 150, tmp_path / "te", seed=2, n_levels=4, n_moments=8, noise=1e-3)
    tr, te = dataset.load(tmp_path / "tr"), dataset.load(tmp_path / "te")
    post = neural.MLPPosterior(hidden=(32, 32), ensemble=2, epochs=300, seed=0).fit(tr["levels"], tr["theta"])
    cal = amortized.calibration(post, te["levels"], te["theta"])
    assert max(cal["per_parameter_rmse"]) < 0.05
    assert abs(cal["empirical"][1] - 0.9545) < 0.1
    truth = {"A": 3.3, "B": 2.1, "C": 1.05}
    obs = rd.synth.synthesize_levels(m, truth, noise=1e-3, seed=4)
    mean, _ = post.predict(dataset.level_features(m, truth, 4)[None, :])
    v = amortized.verify(m, obs, dict(zip(["A", "B", "C"], mean[0])))
    assert v["certified"] and v["converged"]


def test_amortized_structure_proposer_ranks_the_needed_term_first():
    modes = [B.Mode("1", "A", 2), B.Mode("2", "A", 1)]
    spec = B.BuildSpec([B.VibSpace(modes, polyads=[0, 1, 2, 3, 4])], "C1", ["w1: n1", "w2: n2"])
    cands = ["k: a1+ a2 a2 + h.c.", "x11: n1 n1", "x22: n2 n2", "x12: n1 n2"]

    def base(rng):
        return {"w1": rng.uniform(2900, 3100), "w2": rng.uniform(1450, 1550)}
    ds = structure.simulate(spec, cands, base, {"k": 30.0, "x11": 10.0, "x22": 5.0, "x12": 8.0}, 1500,
                            n_levels=5, seed=0)
    prop = structure.AmortizedProposer(ds, top=4)
    rng = np.random.default_rng(5)
    truth = B.compile_build(spec.with_terms(spec.terms + [cands[0]]))
    data = _levels(truth, {"w1": 3010.0, "w2": 1500.0, "k": 25.0}, 0.01, rng)
    p = prop.probabilities(data)
    assert max(p, key=p.get) == cands[0] and p[cands[0]] > 0.8


def test_llm_proposer_prompt_and_parsing_offline():
    from riss_drucker.llm import LLMProposer
    spec = B.BuildSpec([B.RotSpace()], "D2", ["A: Jz Jz", "Bx: Jx Jx", "Cy: Jy Jy"], J=[0, 1, 2])
    seen = {}

    def fake(prompt):
        seen["p"] = prompt
        return "Here you go:\nDK: Jz Jz Jz Jz\n```\nbad line\ndJ: J2 J2\n```"
    prop = LLMProposer(fake, dictionary_degrees=lambda s: A.multidegrees(s, rot=(4, 4)))
    terms = prop.propose({"spec": spec, "diagnostics": {"note": "residual grows as J^4"}})
    assert terms == ["DK: Jz Jz Jz Jz", "dJ: J2 J2"]
    assert "character" not in seen["p"] or "characters" in seen["p"]
    assert "Jx Jx Jx Jx" in seen["p"] and "residual grows" in seen["p"]


def test_built_model_spec_hash_and_cache(tmp_path):
    from riss_drucker import cache
    bs = B.BuildSpec([B.RotSpace(), B.VibSpace([B.Mode("3", "F2")], polyads=[0, 1])], "Td",
                     ["w3: n3", "B: J2", "z: sym(i*Jz a3x+ a3y) + h.c."], J=[0, 1, 2])
    s1, s2 = rd.ModelSpec.built(bs), rd.ModelSpec.built(B.BuildSpec.from_dict(bs.to_dict()))
    assert s1.hash() == s2.hash()
    assert rd.ModelSpec.built(bs.with_terms(bs.terms[:2])).hash() != s1.hash()
    m1, i1 = cache.compile_cached(s1, tmp_path)
    m2, i2 = cache.compile_cached(s1, tmp_path)
    assert not i1["hit"] and i2["hit"]
    th = {"w3": 3019.0, "B": 5.2, "z": 0.3}
    np.testing.assert_allclose(np.sort(rd.solve(m1, th, window=(-1e9, 1e9), method="dense").energies()),
                               np.sort(rd.solve(m2, th, window=(-1e9, 1e9), method="dense").energies()))


def test_ch4_cation_jahn_teller_couplings_from_the_dictionary():
    """T2 electronic state x (nu2 e + nu4 t2): the dictionary yields exactly one linear coupling per
    JT-active mode ([F2^2] = A1 + E + F2), certified against the character count."""
    sp = [B.ElecSpace([("1", "F2")]), B.VibSpace([B.Mode("2", "E"), B.Mode("4", "F2")], nmax=4)]
    spec = B.BuildSpec(sp, "Td", [])
    assert spec and S.get_group("Td").jahn_teller_active("F2") == {"E": 1, "F2": 1}
    en = A.enumerate_terms(spec, A.multidegrees(spec, vib=(0, 2)))
    assert [d["count"] for d in en.by_degree] == [1, 2, 15]
    lin = en.by_degree[1]["terms"]
    bm = B.compile_build(spec.with_terms(["w2: n2", "w4: n4"] + lin))
    assert all(c["passed"] for c in bm.certificates)


def test_methane_polyad_sublevels_from_characters_include_tetradecad_60():
    td = S.get_group("Td")
    METH = [B.Mode("1", "A1", 2), B.Mode("2", "E", 1), B.Mode("3", "F2", 2), B.Mode("4", "F2", 1)]
    v = B._Vib(td, B.VibSpace(METH, polyads=[0, 1, 2, 3, 4]))
    counts = [sum(td.decompose([v.char(g, 0, P) for g in range(td.order)]).values()) for P in range(5)]
    assert counts == [1, 2, 9, 24, 60]


def test_reaction_path_hamiltonian_rigid_limit():
    """Rigid HOOH torsion. At trans the torsion decouples from rotation and G_tt approaches
    2/(m_H r_perp^2) (symmetric internal rotation; COM motion is a small correction). Away from
    trans the H atoms above/below the COM couple torsion to rotation about the C2 axis
    (X_x ~ m rho h cos(gamma/2)), which raises G_tt. The spec compiles with certificates."""
    from riss_drucker.abinitio import rph
    roo, roh, ang = 1.4556, 0.9672, math.radians(102.32)
    geoms, deg = [], np.arange(0, 181, 15.0)
    for g in deg:
        p = math.radians(g)
        h1 = (roh * math.sin(ang), 0.0, roh * math.cos(ang))
        h2 = (roh * math.sin(ang) * math.cos(p), roh * math.sin(ang) * math.sin(p), roo - roh * math.cos(ang))
        geoms.append([["O", 0, 0, 0], ["O", 0, 0, roo], ["H", *h1], ["H", *h2]])
    gam = rph.actual_dihedrals(geoms)
    np.testing.assert_allclose(np.degrees(gam), deg, atol=1e-6)
    V = 400 * (1 - np.cos(gam)) / 2 + 1000 * (1 - np.cos(2 * gam)) / 2
    spec, th, info = rph.build_h2o2_spec(deg, geoms, V, m_max=30, J=[0, 1])
    rperp = roh * math.sin(ang)          # rho
    ref = rph.HBAR2 / 2 * 2 / (rph.MASS["H"] * rperp**2)
    M = 2 * rph.MASS["O"] + 2 * rph.MASS["H"]
    # trans: X = 0 and I_tt = (m rho^2 / 2)(1 - 2 m_H / M) after removing the COM motion (exact)
    assert info["G_tt_cm"]["trans"] == pytest.approx(ref / (1 - 2 * rph.MASS["H"] / M), rel=1e-4)
    assert info["G_tt_cm"]["cis"] > 1.05 * ref
    assert max(info["symmetry_violation_G"].values()) < 1e-10
    bm = B.compile_build(spec)
    assert all(c["passed"] for c in bm.certificates)


def test_certified_rank_and_spin_orbit_kramers_blocks():
    assert A.certified_rank(np.diag([1.0, 1.0, 1e-14]))[0] == 2
    with pytest.raises(GrammarError, match="not certifiable"):
        A.certified_rank(np.diag([1.0, 1e-7, 1e-10]), tol=1e-9)
    # 2T2 + spin-orbit (no rotation: Theta^2 = -1): G3/2 (4) and E5/2 (2) split by 3/2 zeta
    sp = [B.ElecSpace([("1", "F2")]), B.SpinSpace(1)]
    so = "so: i*(e[1y,1z] - e[1z,1y]) Sx + i*(e[1z,1x] - e[1x,1z]) Sy + i*(e[1x,1y] - e[1y,1x]) Sz"
    bm = B.compile_build(B.BuildSpec(sp, "Td", ["E0: Pe[1]", so]))
    lev = {b.irrep: w for b, (w, _) in zip(bm.blocks, bm.eig({"E0": 0.0, "so": 2.0}))}
    assert all(b.kramers for b in bm.blocks)
    assert lev["E5/2"][0] - lev["G3/2"][0] == pytest.approx(3.0, abs=1e-10)
    # CH4+ type vibronic model: T2 x (e + t2) Jahn-Teller with spin-orbit compiles and certifies
    sp2 = [B.ElecSpace([("1", "F2")]), B.SpinSpace(1), B.VibSpace([B.Mode("2", "E"), B.Mode("4", "F2")], nmax=2)]
    lin = A.enumerate_terms(B.BuildSpec(sp2, "Td", []), [(("spin",), ("elec",), ("vib", 1))]).terms
    bm2 = B.compile_build(B.BuildSpec(sp2, "Td", ["w2: n2", "w4: n4", so] + lin))
    # 7 linear couplings, certified by the character count: 2 spin-free JT + 5 spin-orbit-vibronic
    assert all(c["passed"] for c in bm2.certificates) and len(lin) == 7
    assert sum("S" not in t.split(":", 1)[1] for t in lin) == 2


def test_methane_cation_six_well_tunneling_from_symmetry():
    """Gruetter, Woerner & Merkt, JCP 131, 024309 (2009): CH4+ wells give T1 + T2 (geometric phase),
    delta = 4|sigma|; CH2D2+ four-well isomer J=0: A1 (Delta+2sigma), B1+B2 (Delta), A2 (Delta-2sigma)."""
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "case_methane_cation.py"
    spec = importlib.util.spec_from_file_location("case_methane_cation", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.analyse()
    c = out["CH4+"]
    assert c["well_representation"] == {"F1": 1, "F2": 1} and c["dictionary_operators"] == 2
    assert c["certificates_passed"] and c["splitting_over_sigma"] == pytest.approx(4.0, abs=1e-9)
    d = out["CH2D2+"]
    assert d["well_orbits"] == [(1,), (2,), (3, 4, 5, 6)] and d["pattern_reproduced"]


@pytest.mark.skipif(not __import__("os").environ.get("RISS_CH4PLUS_PDF"),
                    reason="needs the PDF of Woerner, Qian & Merkt, JCP 126, 144305 (2007) in RISS_CH4PLUS_PDF")
def test_methane_cation_rovibronic_fit_matches_published_quality():
    import importlib.util
    import os
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "case_methane_cation_rovibronic.py"
    spec = importlib.util.spec_from_file_location("cmcr", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tables = mod.read_tables(pathlib.Path(os.environ["RISS_CH4PLUS_PDF"]))
    model = mod.Model(5)
    f = min((mod.fit_species("CD4+", tables["CD4+"], model, s) for s in (1.0, -1.0)), key=lambda x: x["rms_cm"])
    assert f["rms_cm"] < 0.35 and abs(f["constants"]["A"] - 3.35) < 0.1 and abs(f["delta"] - 1.4) < 0.6
    g = min((mod.fit_species("CH4+", tables["CH4+"], model, s) for s in (1.0, -1.0)), key=lambda x: x["rms_cm"])
    assert g["rms_cm"] < 1.05 and abs(g["delta"] - 16.4) < 2.0
