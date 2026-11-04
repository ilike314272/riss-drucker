"""Windows, near-crossing resampling, truncation estimate, coverage by band."""
from dataclasses import replace

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import posterior
from riss_drucker.rotor import asym_top_labels

TRUE = {"A": 9.405, "B": 1.2953, "C": 1.1342, "DeltaJ": 2.4e-6, "DeltaJK": 9.7e-5, "DeltaK": 4.4e-3,
        "deltaJ": 3.4e-7, "deltaK": 2.7e-5}


@pytest.fixture(scope="module")
def fitted():
    m = rd.compile_asym_rotor(j_max=14, order=4)
    obs = rd.synth.synthesize_levels(m, TRUE, noise=1e-4, seed=0, labels=asym_top_labels(m),
                                     block_filter=lambda b: b["two_j"] <= 20)
    return m, obs, rd.fit(m, obs, TRUE)


def test_windows_match_fit_and_flags(fitted):
    m, obs, res = fitted
    w = posterior.level_windows(m, res, observations=obs, labels=asym_top_labels(m))
    by = {(x["block"], x["index"]): x for x in w}
    for lv in res.levels:
        assert by[(lv["block"], lv["index"])]["sigma"] == pytest.approx(lv["sigma_pred"], rel=1e-6, abs=1e-12)
    assert not any(x["near_crossing"] for x in w)
    ext = [x for x in w if x["extrapolated"]]
    assert ext and all(x["J"] > 10 for x in ext)
    assert all(x["J"] <= 10 or x["extrapolated"] for x in w)
    assert all(x["window"][0] < x["energy"] < x["window"][1] or x["sigma"] == 0 for x in w)


def test_near_crossings_flagged_and_resampled_only(fitted):
    m, obs, res = fitted
    wide = replace(res, covariance=np.asarray(res.covariance) * 1e12)
    w = posterior.level_windows(m, wide)
    flagged = [x for x in w if x["near_crossing"]]
    assert flagged and len(flagged) < len(w)
    assert any(x["beyond_weyl_fence"] for x in w)
    rs = posterior.resample_near_crossings(m, wide, w, n_samples=60, seed=1)
    assert len(rs) == len(flagged)
    for r in rs:
        assert r["n_samples"] == 60 and r["sample_std"] > 0
        q = r["quantiles"]
        assert q[0.025] <= q[0.5] <= q[0.975]


def test_torsion_truncation_estimate_tracks_actual_shift():
    th = {"F": 5.3, "V3": 400.0}
    ref = rd.solve(rd.compile_torsion_rotor(n_fold=3, m_max=60), th, nlowest=4, method="dense")
    ref_e = {b["irrep"]: np.asarray(r["eigenvalues"]) for b, r in zip(ref.blocks, ref.results)}
    prev = None
    for m_max in (9, 12, 15):
        small = rd.compile_torsion_rotor(n_fold=3, m_max=m_max)
        est = posterior.torsion_truncation(small, th, nlowest=4)
        sol = rd.solve(small, th, nlowest=4, method="dense")
        worst = 0.0
        for b, r in zip(sol.blocks, sol.results):
            actual = np.asarray(r["eigenvalues"]) - ref_e[b["irrep"]][: len(r["eigenvalues"])]
            assert np.all(actual >= -1e-9)  # truncation is variational: levels move down as the basis grows
            e = np.array([x["shift_estimate"] for x in est if x["block"] == b["key"]])
            big = actual > 1e-9
            assert np.all(actual[big] <= 2.0 * e[big] + 1e-12)
            assert np.all(e[big] <= 5.0 * actual[big] + 1e-12)
            worst = max(worst, e.max())
        assert prev is None or worst < prev
        prev = worst


def test_coverage_calibrated_for_correct_model(fitted):
    m, obs, _ = fitted
    rep = posterior.coverage_by_band(m, obs, TRUE, band_of=lambda o: f"J%3={(int(o.label.split('(')[0])) % 3}")
    for nominal, emp, se in zip(rep["nominal"], rep["overall"]["empirical"], rep["overall"]["binomial_se"]):
        assert abs(emp - nominal) < 4 * se + 0.01, rep


def test_coverage_detects_misspecified_extrapolation():
    """Data from a sextic Hamiltonian, quartic model, held-out high-J band: under-coverage."""
    m6 = rd.compile_asym_rotor(j_max=14, order=6)
    th6 = {**TRUE, "PhiJ": 1e-10, "PhiJK": 2e-8, "PhiKJ": -4e-7, "PhiK": 6e-6, "phiJ": 3e-11,
           "phiJK": 1e-8, "phiK": 2e-6}
    obs = rd.synth.synthesize_levels(m6, th6, noise=1e-4, seed=2, labels=asym_top_labels(m6))
    m4 = rd.compile_asym_rotor(j_max=14, order=4)
    band = lambda o: "low" if int(o.label.split("(")[0]) <= 9 else "high"
    rep = posterior.coverage_by_band(m4, obs, TRUE, band_of=band)
    assert rep["bands"]["high"]["empirical"][1] < 0.8  # far below nominal 0.9545


def test_result_record_with_m3_parts_validates(fitted):
    pytest.importorskip("jsonschema")
    m, obs, res = fitted
    spec = rd.ModelSpec.asym_rotor(j_max=14, order=4)
    w = posterior.level_windows(m, res, observations=obs, labels=asym_top_labels(m))
    rep = posterior.coverage_by_band(m, obs, TRUE, band_of=lambda o: str(int(o.label.split("(")[0]) % 2))
    rec = rd.fit_result_record(res, spec, model=m, windows=w, coverage=rep)
    rd.validate_result(rec)
    assert all("near_crossing" in lv for lv in rec["levels"])
