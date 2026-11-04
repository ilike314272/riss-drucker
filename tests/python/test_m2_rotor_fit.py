"""M2: asymmetric rotor labels, parameter recovery, gauge and prior behaviour,
result schema (README 12: fit tests)."""
import json

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker.rotor import asym_top_labels, ka_kc_from_tau, label_index

# Illustrative near-prolate constants (cm^-1), H2CO-like magnitudes; not fitted values.
TRUE = {"A": 9.405, "B": 1.2953, "C": 1.1342, "DeltaJ": 2.4e-6, "DeltaJK": 9.7e-5, "DeltaK": 4.4e-3,
        "deltaJ": 3.4e-7, "deltaK": 2.7e-5}


@pytest.mark.parametrize("axis", ["Ir", "IIr", "IIIr"])
def test_block_labels_match_global_tau_ordering(axis):
    """Labels from the C2 parity rule equal labels from sorting all levels of J by
    energy (tau = -J..J) for a well-separated rigid rotor."""
    m = rd.compile_asym_rotor(j_max=10, order=2, axis=axis)
    sol = rd.solve(m, {"A": 3.0, "B": 2.0, "C": 1.0}, window=(-1.0, 1e6), method="dense")
    labels = asym_top_labels(m)
    by_j = {}
    for b, r in zip(sol.blocks, sol.results):
        for i, e in enumerate(r["eigenvalues"]):
            by_j.setdefault(b["two_j"] // 2, []).append((e, labels[b["key"]][i]))
    for J, rows in by_j.items():
        rows.sort()
        for tau, (_, lab) in zip(range(-J, J + 1), rows):
            ka, kc = ka_kc_from_tau(J, tau)
            assert lab == f"{J}({ka},{kc})"


def test_representation_invariance_of_levels():
    th = {"A": 3.1, "B": 2.2, "C": 1.05}
    out = {}
    for axis in ("Ir", "IIr", "IIIr"):
        m = rd.compile_asym_rotor(j_max=8, order=2, axis=axis)
        sol = rd.solve(m, th, window=(-1.0, 1e6), method="dense")
        lab = asym_top_labels(m)
        out[axis] = {lab[b["key"]][i]: e for b, r in zip(sol.blocks, sol.results) for i, e in enumerate(r["eigenvalues"])}
    for axis in ("IIr", "IIIr"):
        assert out[axis].keys() == out["Ir"].keys()
        for k in out["Ir"]:
            assert out[axis][k] == pytest.approx(out["Ir"][k], abs=1e-10)


def _data(m, noise, seed=0, j_filter=None):
    labs = asym_top_labels(m)
    return rd.synth.synthesize_levels(m, TRUE, noise=noise, seed=seed, labels=labs,
                                      block_filter=j_filter)


def test_parameter_recovery_watson_A():
    m = rd.compile_asym_rotor(j_max=20, order=4)
    obs = _data(m, noise=1e-4)
    start = rd.synth.perturb_theta(TRUE, rel=0.02, seed=3)
    res = rd.fit(m, obs, start)
    assert res.converged and res.certified and res.rank == 8 and not res.null_directions
    assert res.reduced_chi2 == pytest.approx(1.0, abs=0.15)
    pulls = np.array([(res.theta[n] - TRUE[n]) / res.sigma[n] for n in TRUE])
    assert np.all(np.abs(pulls) < 4.5), dict(zip(TRUE, pulls))
    assert res.rms == pytest.approx(1e-4, rel=0.2)


def test_pull_distribution_is_calibrated():
    """Across noise realizations, (theta_hat - theta_true)/sigma ~ N(0, 1)."""
    m = rd.compile_asym_rotor(j_max=12, order=4)
    pulls = []
    for seed in range(12):
        res = rd.fit(m, _data(m, noise=2e-4, seed=seed), TRUE)
        pulls += [(res.theta[n] - TRUE[n]) / res.sigma[n] for n in TRUE]
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.35 and 0.7 < pulls.std() < 1.3


def test_gauge_null_direction_detected_and_levels_invariant():
    m = rd.compile_asym_rotor(j_max=10, order=4, extra_operators=["J2"])
    th = {**TRUE, "J2": 0.0}
    obs = rd.synth.synthesize_levels(m, th, noise=1e-4, seed=1)
    start = rd.synth.perturb_theta(th, 0.01, seed=2) | {"J2": 0.05}
    res = rd.fit(m, obs, start)
    assert res.converged and res.rank == 8 and len(res.null_directions) == 1
    nd = res.null_directions[0]
    v = np.array([nd.get(k, 0.0) for k in ("A", "B", "C", "J2")])
    np.testing.assert_allclose(np.abs(v), 0.5, atol=1e-6)  # (1, 1, 1, -1)/2 up to sign
    assert np.sign(v[0]) == np.sign(v[1]) == np.sign(v[2]) == -np.sign(v[3])
    assert all(np.isnan(res.sigma[k]) for k in ("A", "B", "C", "J2"))
    # Gauge fixed exactly at the start: w . theta is unchanged by the fit.
    (w,) = res.gauge_constraints
    moved = sum(c * (res.theta[k] - start[k]) for k, c in w.items())
    scale = sum(abs(c * start[k]) for k, c in w.items())
    assert abs(moved) <= 1e-9 * scale
    # Moving along the null direction leaves every level unchanged (gauge test, README 12).
    th2 = dict(res.theta)
    for k, c in nd.items():
        th2[k] += 0.3 * c
    s1 = rd.solve(m, res.theta, window=(-1.0, 1e6), method="dense")
    s2 = rd.solve(m, th2, window=(-1.0, 1e6), method="dense")
    np.testing.assert_allclose(s1.energies(), s2.energies(), atol=1e-10)
    # Invariant combinations are still determined: A + B + C + J2 shifted together.
    assert res.theta["A"] + res.theta["J2"] == pytest.approx(TRUE["A"], abs=1e-3)


def test_prior_constrains_poorly_determined_parameter():
    m = rd.compile_asym_rotor(j_max=4, order=4)
    obs = _data(m, noise=1e-3)
    flat = rd.fit(m, obs, TRUE)
    prior = rd.Prior(mean={"DeltaK": 4.0e-3}, sigma={"DeltaK": 1e-4})
    pri = rd.fit(m, obs, TRUE, prior=prior)
    assert pri.sigma["DeltaK"] < flat.sigma["DeltaK"]
    assert pri.sigma["DeltaK"] <= 1e-4 * 1.0001
    assert pri.dof == flat.dof + 1


def test_fixed_parameters_and_label_lookup():
    m = rd.compile_asym_rotor(j_max=8, order=4)
    idx = label_index(m)
    blk, i = idx["5(1,4)"]
    assert blk.startswith("J2=10|irrep=O")
    res = rd.fit(m, _data(m, noise=1e-4), TRUE, fixed=["deltaK", "deltaJ"])
    assert res.fixed == ["deltaK", "deltaJ"] and np.isnan(res.sigma["deltaK"])
    assert res.theta["deltaK"] == TRUE["deltaK"]


def test_result_record_validates_against_schema(tmp_path):
    pytest.importorskip("jsonschema")
    spec = rd.ModelSpec.asym_rotor(j_max=8, order=4)
    m, _ = rd.compile_cached(spec, tmp_path)
    res = rd.fit(m, _data(m, noise=1e-4), TRUE)
    final = rd.solve(m, res.theta, nlowest=17)
    rec = rd.fit_result_record(res, spec, model=m, final_solution=final, run_record=rd.run_record(spec_hash=spec.hash()))
    rd.validate_result(rec)
    json.dumps(rec)  # serializable
    assert rec["certified"] and rec["model"]["spec_hash"] == spec.hash()
    assert all(lv["sigma_pred"] is not None for lv in rec["levels"])


def test_fit_input_validation():
    m = rd.compile_asym_rotor(j_max=2, order=2)
    with pytest.raises(ValueError):
        rd.fit(m, [], {"A": 3, "B": 2, "C": 1})
    with pytest.raises(ValueError):
        rd.fit(m, [rd.Observation("nope", 0, 1.0, 1e-3)], {"A": 3, "B": 2, "C": 1})
    with pytest.raises(ValueError):
        rd.fit(m, [rd.Observation(m.blocks[0]["key"], 0, 0.0, 0.0)], {"A": 3, "B": 2, "C": 1})
