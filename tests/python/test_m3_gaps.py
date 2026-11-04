"""M3 gap fills: peak finding + Voigt fit, blends in the line fit, EM-calibrated
assignment posteriors, model-error hyperparameter."""
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import assign, peaks, posterior
from riss_drucker.rotor import asym_top_labels

pytest.importorskip("scipy")

TRUE = {"A": 9.405, "B": 1.2953, "C": 1.1342, "DeltaJ": 2.4e-6, "DeltaJK": 9.7e-5, "DeltaK": 4.4e-3,
        "deltaJ": 3.4e-7, "deltaK": 2.7e-5}
SG, GM = 0.004, 0.002


def test_peak_finding_is_complete_and_calibrated():
    rng = np.random.default_rng(0)
    x = np.arange(0, 50, 0.002)
    pos = np.sort(rng.uniform(1, 49, 80))
    area = rng.uniform(0.02, 1, 80)
    iso = np.array([min(abs(pos[i] - np.delete(pos, i))) > 0.05 for i in range(80)])
    zp, za = [], []
    for seed in range(10):
        y = peaks.simulate_spectrum(pos, area, x, sigma_g=SG, gamma=GM, noise=0.05, seed=seed)
        found = peaks.find_peaks(x, y, sigma_g=SG, gamma=GM, min_snr=5)
        near = [np.min(np.abs(pos - f.position)) for f in found]
        assert max(near) < 0.01                      # no spurious peaks
        assert len(found) >= 78                      # only the unresolved pair may merge
        for f in found:
            k = int(np.argmin(np.abs(pos - f.position)))
            if iso[k]:
                zp.append((f.position - pos[k]) / f.position_sigma)
                za.append((f.area - area[k]) / f.area_sigma)
    zp, za = np.array(zp), np.array(za)
    assert abs(zp.mean()) < 0.15 and 0.8 < zp.std() < 1.2
    assert abs(za.mean()) < 0.3 and 0.8 < za.std() < 1.3


def test_doublet_resolution_and_noise_estimate():
    x = np.arange(0, 2, 0.001)
    fw = peaks.voigt_fwhm(SG, GM)
    for sep, expect in [(1.5 * fw, 2), (0.3 * fw, 1)]:
        y = peaks.simulate_spectrum([1.0, 1.0 + sep], [0.5, 0.4], x, sigma_g=SG, gamma=GM, noise=0.01, seed=1)
        assert len(peaks.find_peaks(x, y, sigma_g=SG, gamma=GM)) == expect
    y = peaks.simulate_spectrum([], [], x, sigma_g=SG, gamma=GM, noise=0.02, seed=2)
    assert peaks.estimate_noise(y) == pytest.approx(0.02, rel=0.1)
    assert peaks.find_peaks(x, y, sigma_g=SG, gamma=GM) == []


@pytest.fixture(scope="module")
def rotor():
    m = rd.compile_asym_rotor(j_max=15, order=4)
    obs = rd.synth.synthesize_levels(m, TRUE, noise=1e-4, seed=0, block_filter=lambda b: b["two_j"] <= 16)
    return m, rd.fit(m, obs, TRUE)


def test_blends_enter_the_line_fit(rotor):
    m, res = rotor
    mu = (1.0, 0.0, 0.0)  # a-type: many unresolved K-doublets
    pk, _ = rd.synth.synthesize_spectrum(m, TRUE, mu, 150.0, fwhm=2e-3, snr=3000, seed=7)
    # Start from a draw consistent with the covariance used for sigma_pred.
    start = dict(zip(res.param_names, posterior.sample_theta(res, 1, seed=8)[0]))
    pred, _ = assign.predict_lines(m, start, mu, 150.0, covariance=res.covariance, min_relative_intensity=1e-4)
    merged = assign.merge_blends(pred, 1e-3)
    a = assign.assign(merged, pk)
    with_b = assign.line_observations(a, merged, pk)
    without = assign.line_observations(a, merged, pk, include_blends=False)
    nb = sum(1 for o in with_b if o.blend)
    assert nb > 20 and len(with_b) == len(without) + nb
    f = rd.fit(m, with_b, start)
    assert f.converged and f.certified and f.reduced_chi2 < 2
    assert max(abs(f.theta[n] - TRUE[n]) / f.sigma[n] for n in TRUE) < 5
    # Blend weights must sum to one.
    bad = rd.Observation(**{**with_b[0].__dict__, "blend": [(*with_b[0].components()[0][:4], 0.5)]})
    with pytest.raises(ValueError, match="sum to 1"):
        rd.fit(m, [bad], start)


def test_em_calibrated_posteriors(rotor):
    m, res = rotor
    mem = lambda s: set(s.split("+")) if s else set()
    wrong = exp = 0.0
    hi_conf_wrong = 0
    for seed in range(6):
        mu, spur = [((1.0, 0, 0), 0.05), ((0.2, 1.0, 0.3), 0.3)][seed % 2]
        pk, _ = rd.synth.synthesize_spectrum(m, TRUE, mu, 150.0, fwhm=2e-3, snr=3000, spurious_density=spur, seed=seed)
        th = dict(zip(res.param_names, posterior.sample_theta(res, 1, seed=50 + seed)[0]))
        pred, _ = assign.predict_lines(m, th, mu, 150.0, covariance=res.covariance, min_relative_intensity=1e-4)
        a = assign.assign(assign.merge_blends(pred, 1e-3), pk)
        assert a.settings["p_detect_mode"] == "auto" and 0 < a.settings["background_fraction"] < 0.5
        for r in a.lines:
            if r["peak"] is None:
                continue
            w = not (mem(r["line"]) & mem(pk[r["peak"]].truth))
            wrong += w
            hi_conf_wrong += w and r["posterior"] > 0.99
        exp += a.expected_misassignments
    assert hi_conf_wrong == 0
    assert wrong <= exp + 3 * math.sqrt(exp) + 2   # not over-confident
    assert exp <= 4 * wrong + 8                     # and not wildly conservative


def test_model_error_restores_coverage_under_misspecification():
    m6 = rd.compile_asym_rotor(j_max=14, order=6)
    th6 = {**TRUE, "PhiJ": 1e-12, "PhiJK": 1e-10, "PhiKJ": -2e-9, "PhiK": 4e-8, "phiJ": 5e-13,
           "phiJK": 5e-11, "phiK": 2e-9}
    obs = rd.synth.synthesize_levels(m6, th6, noise=1e-4, seed=2, labels=asym_top_labels(m6))
    m4 = rd.compile_asym_rotor(j_max=14, order=4)
    jpar = lambda o: str(int(o.label.split("(")[0]) % 2)
    kpar = lambda o: str(int(o.label.split("(")[1].split(",")[0]) % 3)
    cal = posterior.calibrate_model_error(m4, obs, TRUE, band_of=jpar)
    assert cal["model_error"] > 1e-3 and cal["z_rms_after"] == pytest.approx(1.0, abs=0.05)
    before = posterior.coverage_by_band(m4, obs, TRUE, band_of=kpar)
    after = posterior.coverage_by_band(m4, obs, TRUE, band_of=kpar, model_error=cal["model_error"])
    assert before["overall"]["empirical"][1] < 0.3
    assert after["overall"]["empirical"][1] > 0.9  # calibrated on a different split
    ok = rd.synth.synthesize_levels(m4, TRUE, noise=1e-4, seed=3, labels=asym_top_labels(m4))
    assert posterior.calibrate_model_error(m4, ok, TRUE, band_of=jpar)["model_error"] < 5e-5
    with pytest.raises(ValueError):
        rd.fit(m4, ok, TRUE, model_error=-1.0)
