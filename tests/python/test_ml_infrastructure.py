"""Infrastructure for ML-driven general polyatomic modelling: grammar, polyad
compiler with Td symmetry, structure search, KPM moments, datasets, amortized
posterior, ab initio multi-fidelity. All offline and synthetic."""
import collections
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import grammar, selection, vib
from riss_drucker.abinitio import gp as gpm, multifidelity, torsion
from riss_drucker.ml import amortized, dataset, moments

METHANE = [vib.Mode("1", "A1", 2), vib.Mode("2", "E", 1), vib.Mode("3", "F2", 2), vib.Mode("4", "F2", 1)]


def sublevels(sol):
    out = {}
    for b, r in zip(sol.blocks, sol.results):
        e = np.sort(r["eigenvalues"])
        groups, i = [], 0
        while i < len(e):
            j = i
            while j + 1 < len(e) and abs(e[j + 1] - e[i]) < 1e-7 * max(1, abs(e[i])):
                j += 1
            groups.append(j - i + 1)
            i = j + 1
        out[b["irrep"]] = groups
    return out


# ---- grammar --------------------------------------------------------------------

def test_grammar_canonical_forms_and_errors():
    modes = {"1": ("x",), "4": ("x", "y", "z")}
    sym = vib.Symmetrizer(vib.VibSpec(METHANE, "Td"))
    t1 = grammar.parse_term("k: sym(a1+ a4x a4x) + h.c.", {"1": ("x",), "2": ("x", "y"), "3": ("x", "y", "z"),
                                                              "4": ("x", "y", "z")}, sym)
    t2 = grammar.parse_term("k2: 2 * sym(a1+ a4y a4y) + h.c.", {"1": ("x",), "2": ("x", "y"), "3": ("x", "y", "z"),
                                                                 "4": ("x", "y", "z")}, sym)
    assert t1.hermitian and t1.hash == t2.hash  # same operator up to scale -> deduplicated
    # normal ordering: a a+ = a+ a + 1
    p = grammar.normal_order({(("a", "1", "x"), ("c", "1", "x")): 1.0})
    assert p == {(("c", "1", "x"), ("a", "1", "x")): 1.0, (): 1.0}
    for bad in ("x: a9x", "x: a4w", "noparam", "x: n1 +", "x: ( n1"):
        with pytest.raises(grammar.GrammarError):
            grammar.parse_term(bad, {"1": ("x",), "4": ("x", "y", "z")}, lambda q: q)
    # a non-symmetric seed of an F2 component vanishes under Td averaging (odd in a4)
    with pytest.raises(grammar.GrammarError, match="identically zero"):
        grammar.parse_term("z: sym(a4x)", {"4": ("x", "y", "z")}, sym)


def test_compile_certificates_reject_bad_terms():
    spec = vib.VibSpec(METHANE, "Td", [0, 1], ["w4: n4", "bad: a4x+ a4y"])
    with pytest.raises(grammar.GrammarError, match="hermiticity|total_symmetry"):
        vib.compile_vib(spec)
    spec = vib.VibSpec(METHANE, "Td", [0, 1], ["w4: n4", "g44: sym(a4x+ a4x+ a4x a4x)"])
    vib.compile_vib(spec)  # symmetric, Hermitian, polyad-conserving: fine
    with pytest.raises(grammar.GrammarError, match="same operator"):  # sym(a4x+ a4x) is n4/3: a duplicate
        vib.compile_vib(vib.VibSpec(METHANE, "Td", [0, 1], ["w4: n4", "dup: sym(a4x+ a4x) + h.c."]))
    spec = vib.VibSpec(METHANE, "Td", [0, 1], ["w4: n4", "bad: sym(a1+ a4x) + h.c."])
    with pytest.raises(grammar.GrammarError):  # vanishes (F2 x A1 not symmetric) or breaks the polyad
        vib.compile_vib(spec)
    with pytest.raises(grammar.GrammarError, match="same operator"):
        vib.compile_vib(vib.VibSpec(METHANE, "Td", [0, 1], ["a: n4", "b: 3*n4"]))


# ---- Td polyads ------------------------------------------------------------------------

def test_methane_polyad_sublevel_counts_from_symmetry_alone():
    """Harmonic + all symmetry-allowed cubic/quartic terms with random constants:
    dyad 2, pentad 9, octad 24 sublevels (methane's known counts), degeneracies 1/2/3."""
    base = vib.VibSpec(METHANE, "Td", [0, 1, 2, 3], ["w1: n1", "w2: n2", "w3: n3", "w4: n4"])
    cands = selection.EnumerativeProposer(orders=(3, 4)).propose({"spec": base})
    assert len(cands) == len(set(cands)) and len(cands) > 30
    m = vib.compile_vib(vib.VibSpec(METHANE, "Td", [0, 1, 2, 3], base.terms + cands))
    rng = np.random.default_rng(1)
    th = dict(zip(m.param_names, [2917, 1534, 3019, 1306] + list(rng.normal(0, 5, len(m.param_names) - 4))))
    s = sublevels(rd.solve(m, th, window=(-1e9, 1e9), method="dense"))
    assert [len(s[f"P={p}"]) for p in range(4)] == [1, 2, 9, 24]
    assert set(sum(s.values(), [])) <= {1, 2, 3}
    assert collections.Counter(s["P=2"]) == {1: 3, 2: 2, 3: 4}


def test_fermi_resonance_two_level_exact():
    """C1, two 1D modes, nu1 ~ 2 nu2: the (1,0)/(0,2) pair is an exact 2x2 problem."""
    modes = [vib.Mode("1", "A", 2), vib.Mode("2", "A", 1)]
    m = vib.compile_vib(vib.VibSpec(modes, "C1", [2], ["w1: n1", "w2: n2", "k: a1+ a2 a2 + h.c."]))
    w1, w2, k = 3000.0, 1490.0, 25.0
    e = np.sort(rd.solve(m, {"w1": w1, "w2": w2, "k": k}, window=(-1e9, 1e9), method="dense").energies())
    off = k * math.sqrt(2.0)  # <0,2| a2+ a2+ a1 |1,0> = sqrt(2)
    ref = np.sort(np.linalg.eigvalsh([[w1, off], [off, 2 * w2]]))
    np.testing.assert_allclose(e, ref, atol=1e-10)


def test_structure_search_finds_the_resonance():
    """Truth has a Fermi resonance; the search ranks it first among the dictionary
    and accepts it on evidence; a spurious term is not accepted."""
    modes = [vib.Mode("1", "A1", 2), vib.Mode("2", "A1", 1)]
    truth = vib.VibSpec(modes, "C1", [0, 1, 2, 3, 4], ["w1: n1", "w2: n2", "x22: n2*n2", "k: a1+ a2 a2 + h.c."])
    tm = vib.compile_vib(truth)
    th = {"w1": 3000.0, "w2": 1495.0, "x22": -4.0, "k": 30.0}
    sol = rd.solve(tm, th, window=(-1e9, 1e9), method="dense")
    rng = np.random.default_rng(0)
    data = [(b["irrep"], i, e + 0.01 * rng.standard_normal())
            for b, r in zip(sol.blocks, sol.results) for i, e in enumerate(r["eigenvalues"])]

    def obs_for(model):
        keys = {b["irrep"]: b["key"] for b in model.blocks}
        return [rd.Observation(keys[p], i, e, 0.01) for p, i, e in data]

    base = vib.VibSpec(modes, "C1", [0, 1, 2, 3, 4], ["w1: n1", "w2: n2", "x22: n2*n2"])
    cands = ["k: a1+ a2 a2 + h.c.", "x11: n1*n1", "x12: n1*n2"]
    res = selection.search(base, obs_for, {"w1": 3000.0, "w2": 1495.0, "x22": -4.0}, cands, max_add=3)
    assert res.steps[0].term.startswith("k:") and res.steps[0].accepted
    assert res.fit.theta["k"] == pytest.approx(30.0, rel=0.01)
    assert all(not s.accepted for s in res.steps[1:])  # nothing else is needed


def test_callable_proposer_goes_through_the_grammar():
    def fake_llm(prompt_json):
        return "# proposals\nk: sym(a1+ a4x a4x) + h.c.\nnot a term\nbad: a4x+ a4y\n"
    terms = selection.CallableProposer(fake_llm).propose({"modes": ["1", "4"], "diagnostics": {}})
    assert terms == ["k: sym(a1+ a4x a4x) + h.c.", "bad: a4x+ a4y"]
    base = ["w1: n1", "w2: n2", "w3: n3", "w4: n4"]
    vib.compile_vib(vib.VibSpec(METHANE, "Td", [0, 1, 2], base + terms[:1]))
    with pytest.raises(grammar.GrammarError):  # the malformed proposal is rejected, never fitted
        vib.compile_vib(vib.VibSpec(METHANE, "Td", [0, 1, 2], base + terms[1:]))


# ---- ML: moments, datasets, amortized ---------------------------------------------------

def test_chebyshev_moments_match_exact_trace():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((60, 60)) * (rng.random((60, 60)) < 0.2)
    a = a + a.T + np.diag(rng.standard_normal(60) * 3)
    ip = np.concatenate([[0], np.cumsum((a != 0).sum(1))]).astype(np.int64)
    ix = np.concatenate([np.nonzero(r)[0] for r in a]).astype(np.int32)
    dt = a[a != 0]
    mu_e, c, h = moments.chebyshev_moments(ip, ix, dt, 20, exact=True)
    w = np.linalg.eigvalsh(a)
    x = (w - c) / h
    ref = [np.mean(np.cos(k * np.arccos(np.clip(x, -1, 1)))) for k in range(20)]
    np.testing.assert_allclose(mu_e, ref, atol=1e-12)
    mu_s, _, _ = moments.chebyshev_moments(ip, ix, dt, 20, n_vectors=400, seed=3)
    assert np.max(np.abs(mu_s - ref)) < 0.05  # stochastic trace: O(1/sqrt(R n))
    dos = moments.jackson_dos(mu_e, c, h, np.linspace(w.min(), w.max(), 200))
    assert np.all(dos > -1e-6) and dos.sum() > 0


def test_dataset_round_trip_and_amortized_posterior_is_calibrated(tmp_path):
    m = rd.compile_asym_rotor(j_max=4, order=2)
    prior = dataset.Prior({"A": ("uniform", 2.5, 4.0), "B": ("uniform", 1.6, 2.4), "C": ("uniform", 0.8, 1.4)})
    man = dataset.generate(m, prior, 300, tmp_path / "train", seed=1, shard_size=100, n_levels=4, n_moments=8,
                           noise=1e-3)
    assert len(man["shards"]) == 3 and man["param_names"] == ["A", "B", "C"]
    tr = dataset.load(tmp_path / "train")
    te = dataset.load(dataset.generate(m, prior, 100, tmp_path / "test", seed=2, n_levels=4, n_moments=8,
                                       noise=1e-3) and tmp_path / "test")
    post = amortized.BayesLinearPosterior().fit(tr["levels"], tr["theta"])
    cal = amortized.calibration(post, te["levels"], te["theta"])
    assert max(cal["per_parameter_rmse"]) < 0.05
    assert abs(cal["empirical"][1] - 0.9545) < 0.08
    # physics verifies the amortized guess
    truth = {"A": 3.3, "B": 2.1, "C": 1.05}
    obs = rd.synth.synthesize_levels(m, truth, noise=1e-3, seed=4)
    lv = dataset.level_features(m, truth, 4)
    mean, _ = post.predict(lv[None, :])
    v = amortized.verify(m, obs, dict(zip(["A", "B", "C"], mean[0])))
    assert v["certified"] and v["converged"]
    assert all(abs(v["fit"].theta[k] - truth[k]) < 5 * v["fit"].sigma[k] for k in truth)


# ---- ab initio: GP, multi-fidelity, propagation -------------------------------------------

def test_gp_and_delta_model_with_honest_variance():
    f_high = lambda x: 400 * (1 - np.cos(x)) / 2 + 900 * (1 - np.cos(2 * x)) / 2 + 30 * np.sin(x) ** 2
    f_low = lambda x: 380 * (1 - np.cos(x)) / 2 + 950 * (1 - np.cos(2 * x)) / 2
    lx = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    hx = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    dm = multifidelity.DeltaModel.fit(lx, f_low(lx), hx, f_high(hx), kind="periodic")
    grid = np.linspace(0, 2 * np.pi, 181)
    mu, var = dm.predict(grid)
    z = (f_high(grid) - mu) / np.sqrt(var + 1e-12)
    assert np.max(np.abs(f_high(grid) - mu)) < 2.0          # far better than the low level (~60)
    assert np.mean(np.abs(z) < 3) > 0.95                    # variance is honest on the truth
    g = gpm.GP("rbf").fit(np.linspace(0, 5, 12), np.sin(np.linspace(0, 5, 12)))
    m2, v2 = g.predict(np.array([2.5]))
    assert abs(m2[0] - np.sin(2.5)) < 0.05 and v2[0] < 0.05


def test_potential_uncertainty_propagates_to_levels():
    model = rd.compile_torsion_rotor(n_fold=1, v_orders=[1, 2, 3], m_max=40)
    phi = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    V = 300 * (1 - np.cos(phi)) / 2 + 800 * (1 - np.cos(2 * phi)) / 2 + 50 * (1 - np.cos(3 * phi)) / 2
    c = torsion.fit_cosine_series(phi, V + 123.0, [1, 2, 3])  # constant offset is irrelevant
    assert c == pytest.approx({"V1": 300.0, "V2": 800.0, "V3": 50.0}, abs=1e-8)
    rng = np.random.default_rng(5)
    samples = [V * (1 + 0.02 * rng.standard_normal()) for _ in range(40)]
    p = torsion.propagate(model, 20.0, phi, samples, [1, 2, 3])
    assert np.all(p["sd"][1:] > 0) and p["samples"].shape == (40, 8)


def test_reaction_path_tunneling_solver():
    """Sinc-DVR double well: high-barrier harmonic limit (hbar w/2, 3 hbar w/2, tiny splitting) and
    WKB-like decay of the splitting, log(splitting) linear in sqrt(V0)."""
    from riss_drucker.abinitio.tunneling import HBAR2_HALF, double_well_levels, splitting
    k = 2.0e4                                       # cm^-1 / (amu A^2)
    s = np.linspace(0, 1.0, 300)
    lev, _, _ = double_well_levels(s, 0.5 * k * (1.0 - s) ** 2, n_grid=700)
    hw = math.sqrt(2 * HBAR2_HALF * k)              # hbar w = sqrt(hbar^2 k / m), m = 1 amu
    assert lev[0] == pytest.approx(hw / 2, rel=2e-3) and lev[1] - lev[0] < 1e-3
    assert lev[2] == pytest.approx(1.5 * hw, rel=2e-3)
    x = np.linspace(0, 0.8, 160)
    V0s = np.array([400.0, 700.0, 1000.0, 1300.0])
    spl = np.array([splitting(x, V * (1 - (x / 0.8) ** 2) ** 2)["pair_splitting"] for V in V0s])
    assert np.all(np.diff(spl) < 0)
    r = np.corrcoef(np.sqrt(V0s), np.log(spl))[0, 1]
    assert r < -0.999


def test_case_studies_verify():
    """Re-analyse the committed CH4+ ab initio surface(s) and check every recorded number."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, str(root / "scripts" / "verify_case_studies.py"), "--figures",
                          str(Path(__import__("tempfile").mkdtemp()))], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "FAIL" not in out.stdout
