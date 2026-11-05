"""Open-shell rotor and large-amplitude-motion models (M8 pulled forward):
exact limits, compile certificates, half-integer J, assignment, specs/cache."""
import math

import numpy as np
import pytest

import riss_drucker as rd

A, B, C = 3.0, 2.0, 1.0


def rotor_levels(j_max):
    a = rd.compile_asym_rotor(j_max=j_max, order=2)
    sol = rd.solve(a, {"A": A, "B": B, "C": C}, window=(-1e9, 1e9), method="dense")
    out = {}
    for b, r in zip(sol.blocks, sol.results):
        out.setdefault(b["two_j"] // 2, []).extend(r["eigenvalues"])
    return out


@pytest.mark.parametrize("two_s", [1, 2, 3])
def test_isotropic_spin_rotation_is_exact(two_s):
    """gamma N.S = gamma/2 [J(J+1) - N(N+1) - S(S+1)] commutes with the rotor."""
    m = rd.compile_open_rotor(two_s=two_s, n_max=5, order=2)
    assert all(c["passed"] for c in m.compile_certificates)
    names = {c["name"] for c in m.compile_certificates}
    assert {"hermiticity", "reality", "casimir_N2", "sixj_spin_rotation"} <= names
    g = 0.07
    sol = rd.solve(m, {"A": A, "B": B, "C": C, "eps_aa": g, "eps_bb": g, "eps_cc": g},
                   window=(-1e9, 1e9), method="dense")
    rot = rotor_levels(12)
    S = two_s / 2
    for b, r in zip(sol.blocks, sol.results):
        J = b["two_j"] / 2
        assert b["two_j"] % 2 == two_s % 2  # half-integer J for odd electron counts
        exp = [e + g / 2 * (J * (J + 1) - N * (N + 1) - S * (S + 1))
               for N in range(int(round(abs(J - S))), int(round(J + S)) + 1) for e in rot[N]]
        np.testing.assert_allclose(np.sort(r["eigenvalues"]), np.sort(exp), atol=1e-11)


def test_doublet_first_order_anisotropic_spin_rotation():
    """Prolate symmetric top, S = 1/2: to first order the F1-F2 splitting of (N, K) is
    (N + 1/2) [eps_bb + (eps_aa - eps_bb) K^2 / (N(N+1))] (eps_bb = eps_cc)."""
    m = rd.compile_open_rotor(two_s=1, n_max=6, order=2, axis="Ir")

    def worst_deviation(scale):
        ea, eb = 0.004 * scale, 0.0015 * scale
        sol = rd.solve(m, {"A": 5.0, "B": 1.0, "C": 1.0, "eps_aa": ea, "eps_bb": eb, "eps_cc": eb},
                       window=(-1e9, 1e9), method="dense", keep_vectors=True)
        lab = rd.open_shell.assign(m, sol)
        lv = {}
        for b, r in zip(sol.blocks, sol.results):
            for e, l in zip(r["eigenvalues"], lab[b["key"]]):
                lv.setdefault((l["N"], l["Ka"], l["F"]), []).append(e)
        dev, n = 0.0, 0
        for N in range(1, 6):
            for K in range(0, N + 1):
                if (N, K, 1) in lv and (N, K, 2) in lv:
                    split = np.mean(lv[(N, K, 1)]) - np.mean(lv[(N, K, 2)])
                    first = (N + 0.5) * (eb + (ea - eb) * K * K / (N * (N + 1)))
                    dev = max(dev, abs(split - first) / first)
                    n += 1
        assert n > 15
        return dev

    d1, d01 = worst_deviation(1.0), worst_deviation(0.1)
    assert d1 < 2e-3                      # first order is accurate ...
    assert d1 / d01 == pytest.approx(10, rel=0.3)  # ... and the relative error is linear in eps (second order)


def test_assignment_and_triplet_spin_spin():
    m = rd.compile_open_rotor(two_s=2, n_max=4, order=2, spin_spin=True)
    assert m.param_names[-2:] == ["D_ss", "E_ss"]
    sol = rd.solve(m, {n: 0.0 for n in m.param_names} | {"A": A, "B": B, "C": C, "D_ss": 0.3},
                   window=(-1e9, 1e9), method="dense", keep_vectors=True)
    assert sol.certified
    lab = rd.open_shell.assign(m, sol)
    for b in sol.blocks:
        for l in lab[b["key"]]:
            assert abs(l["J"] - l["N"]) <= 1 and 1 <= l["F"] <= 3
    with pytest.raises(ValueError):
        rd.compile_open_rotor(two_s=1, spin_spin=True)  # spin-spin needs S >= 1
    with pytest.raises(ValueError):
        rd.compile_open_rotor(two_s=0)


def test_inversion_exact_limits_and_tunnelling():
    m = rd.compile_inversion(n_basis=60, potential_order=4)
    assert all(c["passed"] for c in m.compile_certificates)
    h = np.sort(rd.solve(m, {"G": 1.0, "V2": 1.0, "V4": 0.0}, nlowest=5, method="dense").energies())
    np.testing.assert_allclose(h[:8], 2 * np.arange(8) + 1, atol=1e-10)
    q = np.sort(rd.solve(m, {"G": 1.0, "V2": 0.0, "V4": 1.0}, nlowest=2, method="dense").energies())
    assert q[0] == pytest.approx(1.0603620904841829, abs=1e-11)
    # Double well -V2 x^2 + V4 x^4: tunnelling doublets, '+' below '-', splitting shrinks with the barrier.
    splits = []
    for depth in (4.0, 8.0, 12.0):
        mm = rd.compile_inversion(n_basis=80, basis_scale=0.6, potential_order=4)
        sol = rd.solve(mm, {"G": 1.0, "V2": -depth, "V4": 1.0}, nlowest=1, method="dense")
        e = {b["irrep"]: r["eigenvalues"][0] for b, r in zip(sol.blocks, sol.results)}
        assert e["+"] < e["-"]
        splits.append(e["-"] - e["+"])
    assert splits[0] > splits[1] > splits[2] > 0
    assert math.log(splits[0] / splits[1]) > 1.0  # roughly exponential in the barrier


def test_inversion_rotation_coupling_shifts_splitting_with_K():
    m = rd.compile_inversion(n_basis=60, basis_scale=0.6, potential_order=4, j_max=3, rotation_coupling=True)
    th = {"G": 1.0, "V2": -6.0, "V4": 1.0, "B": 0.3, "C": 0.2, "Bx2": 0.0, "Cx2": 0.01}
    sol = rd.solve(m, th, nlowest=1, method="dense")
    split = {}
    for b, r in zip(sol.blocks, sol.results):
        key = (b["two_j"] // 2, b["sub"])
        split.setdefault(key, {})[b["irrep"]] = r["eigenvalues"][0]
    s = {k: v["-"] - v["+"] for k, v in split.items()}
    assert s[(3, "K=3")] != pytest.approx(s[(3, "K=0")], rel=1e-6)  # K-dependent inversion splitting
    th0 = dict(th, Cx2=0.0)
    sol0 = rd.solve(m, th0, nlowest=1, method="dense")
    s0 = {(b["two_j"] // 2, b["sub"]): r["eigenvalues"][0] for b, r in zip(sol0.blocks, sol0.results) if b["irrep"] == "+"}
    assert s0[(3, "K=3")] - s0[(3, "K=0")] == pytest.approx(0.2 * 9 - 0.3 * 9, abs=1e-10)  # uncoupled: rigid top


def test_ram_reduces_to_torsion_rotor_and_to_rotated_rotor():
    F, rho, V3, Aa, Bb = 5.0, 0.08, 300.0, 3.0, 1.0
    ram = rd.compile_ram_torsion(n_fold=3, m_max=24, j_max=3)
    assert all(c["passed"] for c in ram.compile_certificates)
    r1 = rd.solve(ram, {"A": Aa + F * rho**2, "B": Bb, "C": Bb, "D_ab": 0.0, "F": F, "D_pa": -2 * F * rho, "V3": V3},
                  nlowest=40, method="dense")
    tr = rd.compile_torsion_rotor(n_fold=3, m_max=24, j_max=3, kp_coupling=True)
    r2 = rd.solve(tr, {"F": F, "V3": V3, "A": Aa + F * rho**2, "B": Bb, "D_Kp": -2 * F * rho}, nlowest=40, method="dense")
    for J in range(4):
        e1 = np.sort([e for b, r in zip(r1.blocks, r1.results) if b["two_j"] == 2 * J for e in r["eigenvalues"]])[:30]
        e2 = np.sort([e for b, r in zip(r2.blocks, r2.results) if b["two_j"] == 2 * J for e in r["eigenvalues"]])[:30]
        np.testing.assert_allclose(e1, e2, atol=1e-10)
    # rho = 0, V = 0: rotor with an off-diagonal D_ab equals the principal-axis rotor + F m^2.
    Dab = 0.4
    w = np.linalg.eigvalsh([[3.0, Dab], [Dab, 2.0]])  # (a, b) inertia block -> principal constants
    small = rd.compile_ram_torsion(n_fold=1, m_max=0, j_max=4)
    s1 = rd.solve(small, {"A": 3.0, "B": 2.0, "C": 1.0, "D_ab": Dab, "F": 5.0, "D_pa": 0.0, "V1": 0.0},
                  window=(-1e9, 1e9), method="dense")
    rot = rd.compile_asym_rotor(j_max=4, order=2)
    s2 = rd.solve(rot, {"A": w[1], "B": w[0], "C": 1.0}, window=(-1e9, 1e9), method="dense")
    np.testing.assert_allclose(np.sort(s1.energies()), np.sort(s2.energies()), atol=1e-10)


def test_specs_hash_compile_and_cache(tmp_path):
    for spec in (rd.ModelSpec.open_rotor(two_s=1, n_max=3), rd.ModelSpec.inversion(n_basis=20),
                 rd.ModelSpec.ram_torsion(m_max=6, j_max=2)):
        m, info = rd.compile_cached(spec, tmp_path)
        m2, info2 = rd.compile_cached(spec, tmp_path)
        assert not info["hit"] and info2["hit"] and m2.param_names == m.param_names
        assert len(spec.hash()) == 64
    assert rd.ModelSpec.open_rotor(two_s=1, n_max=3).group_chain[-1] == "D2^(2)"  # double group (Kramers)
