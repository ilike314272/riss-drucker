"""M1 exact reduced baseline from Python: torsion against exact references,
solver paths, certificates, warm start and parallel determinism."""
import json

import numpy as np
import pytest

import riss_drucker as rd


def torsion(n_fold=3, m_max=40, v_orders=(1,), **kw):
    return rd.compile_torsion_rotor(n_fold=n_fold, v_orders=list(v_orders), m_max=m_max, **kw)


def block_levels(sol, irrep):
    (r,) = [r for b, r in zip(sol.blocks, sol.results) if b["irrep"] == irrep]
    return np.asarray(r["eigenvalues"])


@pytest.mark.parametrize("V", [0.0, 15.0, 120.0, 900.0])
def test_mathieu_exact_A_species(V):
    """V3-only torsion maps onto Mathieu's equation with x = 3 phi / 2:
    E = V/2 + (9F/4) a(q), q = V/(9F). A1 (cos) <-> a_{2k}; A2 (sin) <-> b_{2k}."""
    special = pytest.importorskip("scipy.special")
    F = 5.0
    q = V / (9 * F)
    sol = rd.solve(torsion(m_max=90), {"F": F, "V3": V}, nlowest=6, method="dense")
    assert sol.certified, sol.failed_certificates()
    a1 = block_levels(sol, "sigma=0,+")
    a2 = block_levels(sol, "sigma=0,-")
    ref_a1 = [V / 2 + 9 * F / 4 * special.mathieu_a(2 * k, q) for k in range(6)]
    ref_a2 = [V / 2 + 9 * F / 4 * special.mathieu_b(2 * k + 2, q) for k in range(6)]
    np.testing.assert_allclose(a1, ref_a1, rtol=0, atol=1e-7 * max(1.0, V))
    np.testing.assert_allclose(a2, ref_a2, rtol=0, atol=1e-7 * max(1.0, V))


def periodic_dvr_levels(F, potential, npts=401):
    """Independent reference: periodic sinc DVR on [0, 2 pi) (Colbert-Miller)."""
    phi = 2 * np.pi * np.arange(npts) / npts
    d = np.arange(npts)[:, None] - np.arange(npts)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = F * (-1.0) ** d * np.cos(np.pi * d / npts) / (2 * np.sin(np.pi * d / npts) ** 2)
    np.fill_diagonal(t, F * (npts**2 - 1) / 12)
    return np.linalg.eigvalsh(t + np.diag(potential(phi)))


@pytest.mark.parametrize("V3,V6", [(350.0, 0.0), (350.0, -40.0), (40.0, 25.0)])
def test_all_species_against_independent_dvr(V3, V6):
    F = 5.3
    sol = rd.solve(torsion(m_max=80, v_orders=(1, 2)), {"F": F, "V3": V3, "V6": V6}, nlowest=8)
    assert sol.certified
    ours = np.sort(sol.energies())[:20]
    pot = lambda p: V3 / 2 * (1 - np.cos(3 * p)) + V6 / 2 * (1 - np.cos(6 * p))
    ref = periodic_dvr_levels(F, pot)[:20]
    np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_free_rotor_limit_and_assignments():
    sol = rd.solve(torsion(m_max=12), {"F": 2.0, "V3": 0.0}, nlowest=3)
    assert sol.certified
    for row in sol.levels():
        m = int(row["assignment"].split("=")[1].split(",")[0].replace("+-", ""))
        assert row["energy"] == pytest.approx(2.0 * m * m, abs=1e-12)
        assert row["weight"] == pytest.approx(1.0)


def test_tunnelling_splitting_shrinks_with_barrier():
    """A1/E splitting of the ground torsional level decreases as the barrier grows."""
    model = torsion(m_max=60)
    splits = []
    for V in (100.0, 300.0, 900.0):
        sol = rd.solve(model, {"F": 5.0, "V3": V}, nlowest=1)
        splits.append(block_levels(sol, "sigma=1")[0] - block_levels(sol, "sigma=0,+")[0])
    assert all(s > 0 for s in splits) and splits[0] > splits[1] > splits[2]


def test_dense_and_lanczos_paths_agree():
    model = torsion(m_max=900)  # blocks of ~300-600: both paths feasible
    th = {"F": 5.0, "V3": 400.0}
    dense = rd.solve(model, th, nlowest=10, method="dense")
    lanc = rd.solve(model, th, nlowest=10, method="lanczos")
    assert dense.certified and lanc.certified
    np.testing.assert_allclose(dense.energies(), lanc.energies(), atol=1e-8)
    assert {r["method"] for r in lanc.results} == {"lanczos"}
    # Single-V_n blocks are tridiagonal in the m basis: certified by Sturm counts.
    assert all(r["cost"]["inertia_method"] == "sturm" for r in lanc.results)


def test_window_mode_counts_match():
    model = torsion(m_max=60, v_orders=(1, 2))
    th = {"F": 5.0, "V3": 300.0, "V6": 20.0}
    sol = rd.solve(model, th, window=(200.0, 700.0))
    assert sol.certified
    for r in sol.results:
        cert = next(c for c in r["certificates"] if c["name"] == "inertia_count")
        assert cert["value"] == cert["bound"] == len(r["eigenvalues"])
        assert all(200.0 <= e < 700.0 for e in r["eigenvalues"])
    full = rd.solve(model, th, window=(-1e9, 1e9), method="dense")
    e = full.energies()
    assert len(sol.energies()) == int(((e >= 200.0) & (e < 700.0)).sum())


def test_warm_start_equals_cold_start():
    model = torsion(m_max=300)
    th0 = {"F": 5.0, "V3": 400.0}
    th1 = {"F": 5.0, "V3": 404.0}
    prev = rd.solve(model, th0, nlowest=6, method="lanczos", keep_vectors=True)
    cold = rd.solve(model, th1, nlowest=6, method="lanczos")
    warm = rd.solve(model, th1, nlowest=6, method="lanczos", warm_start=prev)
    assert cold.certified and warm.certified
    np.testing.assert_allclose(warm.energies(), cold.energies(), atol=1e-9)
    with pytest.raises(ValueError):
        rd.solve(model, th1, nlowest=6, warm_start=cold)  # no vectors kept


def test_threads_do_not_change_results():
    model = torsion(m_max=50, j_max=3, kp_coupling=True)
    th = {"F": 5.0, "V3": 200.0, "A": 4.0, "B": 0.8, "D_Kp": -0.9}
    s1 = rd.solve(model, th, nlowest=4, method="lanczos", threads=1, seed=5)
    s4 = rd.solve(model, th, nlowest=4, method="lanczos", threads=4, seed=5)
    assert s1.certified and s4.certified
    np.testing.assert_array_equal(s1.energies(), s4.energies())


def test_theta_validation_and_options():
    model = torsion()
    assert model.param_names == ["F", "V3"]
    with pytest.raises(ValueError, match="missing"):
        rd.solve(model, {"F": 1.0}, nlowest=2)
    with pytest.raises(ValueError):
        rd.solve(model, [1.0, 2.0, 3.0], nlowest=2)
    with pytest.raises(ValueError, match="exactly one"):
        rd.solve(model, [1.0, 2.0])
    with pytest.raises(TypeError):
        rd.solve(model, [1.0, 2.0], nlowest=2, not_an_option=1)
    with pytest.raises(ValueError):
        rd.compile_torsion_rotor(n_fold=0)


def test_failed_certificate_is_reported_not_raised():
    model = torsion(m_max=600)
    sol = rd.solve(model, {"F": 5.0, "V3": 400.0}, nlowest=40, method="lanczos",
                   lanczos_max_basis=3, lanczos_max_runs=1)
    assert not sol.certified
    names = {c["name"] for c in sol.failed_certificates()}
    assert "lanczos_convergence" in names


def test_memory_budget_fails_early():
    model = torsion(m_max=3000)
    with pytest.raises(MemoryError):
        rd.solve(model, {"F": 5.0, "V3": 400.0}, nlowest=2, method="dense", memory_budget=1e6)


def test_operators_are_exposed_and_assemble_linearly():
    model = torsion(m_max=10, v_orders=(1, 2))
    th = [5.0, 300.0, -20.0]
    for b in range(model.num_blocks):
        indptr, indices, data, shape = model.assemble(b, th)
        h = np.zeros(shape)
        for r in range(shape[0]):
            h[r, indices[indptr[r]:indptr[r + 1]]] = data[indptr[r]:indptr[r + 1]]
        ref = np.zeros(shape)
        for k, t in enumerate(th):
            ip, ix, dt, _ = model.operator(b, k)
            for r in range(shape[0]):
                ref[r, ix[ip[r]:ip[r + 1]]] += t * dt[ip[r]:ip[r + 1]]
        np.testing.assert_allclose(h, ref, atol=1e-13)
        np.testing.assert_allclose(h, h.T, atol=0)
    assert model.compile_certificates[0]["passed"]
    assert json.loads(model.spec_json)["v_orders"] == [1, 2]


def test_inertia_and_sturm_bindings():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((40, 40))
    a = a + a.T
    w = np.linalg.eigvalsh(a)
    for b in rd.linalg_backends():
        assert rd.inertia(a, 0.3, backend=b)["n_neg"] == int((w < 0.3).sum())
    d, e = rng.standard_normal(30), rng.standard_normal(29)
    t = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    assert rd.sturm_count(d, e, 0.1)["n_neg"] == int((np.linalg.eigvalsh(t) < 0.1).sum())


def test_solve_csr_on_scipy_style_arrays():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((50, 50)) * (rng.random((50, 50)) < 0.1)
    a = a + a.T + np.diag(rng.standard_normal(50))
    indptr = np.concatenate([[0], np.cumsum((a != 0).sum(1))])
    indices = np.concatenate([np.nonzero(row)[0] for row in a]).astype(np.int32)
    data = a[a != 0]
    res = rd.solve_csr(indptr, indices, data, nlowest=5, method="lanczos")
    assert res["certified"]
    np.testing.assert_allclose(res["eigenvalues"], np.linalg.eigvalsh(a)[:5], atol=1e-9)


def test_summary_is_json_serializable():
    sol = rd.solve(torsion(m_max=20), {"F": 5.0, "V3": 100.0}, nlowest=2)
    text = json.dumps(sol.summary())
    assert "inertia_count" in text and sol.summary()["certified"]
