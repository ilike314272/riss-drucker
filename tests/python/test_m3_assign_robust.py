"""Assignment (gating, DP, posteriors, Poisson null, BH) and the outlier mixture."""
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import assign
from riss_drucker.assign import Peak, PredictedLine
from riss_drucker.rotor import asym_top_labels

TRUE = {"A": 9.405, "B": 1.2953, "C": 1.1342, "DeltaJ": 2.4e-6, "DeltaJK": 9.7e-5, "DeltaK": 4.4e-3,
        "deltaJ": 3.4e-7, "deltaK": 2.7e-5}


def test_chance_probability_and_bh():
    assert assign.chance_probability(0.0, 5.0) == 0.0
    assert assign.chance_probability(0.1, 5.0) == pytest.approx(1 - math.exp(-1.0))
    # Benjamini-Hochberg reference example.
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    assert assign.benjamini_hochberg(p, 0.05) == [True, True, True, True, False]
    assert assign.benjamini_hochberg([0.5, 0.9], 0.05) == [False, False]


def test_monotone_alignment_resolves_order():
    # Two lines whose nearest peaks are crossed; order-preserving DP must not cross them.
    pred = [PredictedLine("a", 10.000, 0.004, series="R"), PredictedLine("b", 10.010, 0.004, series="R")]
    peaks = [Peak(10.0065, 0.001), Peak(10.0120, 0.001)]
    res = assign.assign(pred, peaks, use_intensity=False, density_half_width=1.0)
    got = {r["line"]: r["peak"] for r in res.lines}
    assert got == {"a": 0, "b": 1}


def test_merge_blends():
    ls = [PredictedLine("x", 1.0, 0.0, intensity=1.0), PredictedLine("y", 1.0004, 0.0, intensity=3.0),
          PredictedLine("z", 2.0, 0.0, intensity=1.0)]
    out = assign.merge_blends(ls, 1e-3)
    assert [o.id for o in out] == ["x+y", "z"]
    assert out[0].position == pytest.approx(1.0003) and out[0].intensity == 4.0


@pytest.fixture(scope="module")
def setup():
    m = rd.compile_asym_rotor(j_max=15, order=4)
    obs = rd.synth.synthesize_levels(m, TRUE, noise=1e-4, seed=0, block_filter=lambda b: b["two_j"] <= 16)
    return m, rd.fit(m, obs, TRUE)


def members(i):
    return set(i.split("+")) if i else set()


@pytest.mark.parametrize("seed,mu,spur", [(0, (1.0, 0, 0), 0.02), (1, (0.2, 1.0, 0.3), 0.2), (2, (0.2, 1.0, 0.3), 0.2)])
def test_end_to_end_assignment_is_accurate_and_calibrated(setup, seed, mu, spur):
    m, res = setup
    peaks, _ = rd.synth.synthesize_spectrum(m, TRUE, mu, 150.0, fwhm=2e-3, snr=3000, spurious_density=spur, seed=seed)
    th = dict(zip(res.param_names, rd.posterior.sample_theta(res, 1, seed=5 + seed)[0]))
    pred, _ = assign.predict_lines(m, th, mu, 150.0, covariance=res.covariance, min_relative_intensity=1e-4)
    a = assign.assign(assign.merge_blends(pred, 1e-3), peaks)
    rows = [r for r in a.lines if r["peak"] is not None]
    wrong = sum(1 for r in rows if not (members(r["line"]) & members(peaks[r["peak"]].truth)))
    assert len(rows) > 0.85 * sum(p.truth is not None for p in peaks)
    assert wrong <= 0.01 * len(rows) + 2
    assert wrong <= a.expected_misassignments + 3 * math.sqrt(a.expected_misassignments) + 3
    to_spurious = sum(1 for r in rows if peaks[r["peak"]].truth is None)
    assert to_spurious <= max(3, 0.1 * sum(p.truth is None for p in peaks))


def test_robust_fit_down_weights_outliers():
    m = rd.compile_asym_rotor(j_max=12, order=4)
    obs = rd.synth.synthesize_levels(m, TRUE, noise=1e-4, seed=3, labels=asym_top_labels(m))
    rng = np.random.default_rng(4)
    bad = rng.choice(len(obs), size=len(obs) // 20, replace=False)
    for i in bad:
        obs[i].energy += rng.choice([-1, 1]) * rng.uniform(30, 100) * 1e-4
    plain = rd.fit(m, obs, TRUE)
    rob = rd.robust.fit_robust(m, obs, TRUE)
    assert plain.reduced_chi2 > 10
    pulls = [abs(rob.fit.theta[n] - TRUE[n]) / rob.fit.sigma[n] for n in TRUE]
    assert max(pulls) < 4.5
    p = np.asarray(rob.outlier_probability)
    assert np.all(p[bad] > 0.9)
    clean = np.setdiff1d(np.arange(len(obs)), bad)
    assert np.mean(p[clean] < 0.1) > 0.97
    assert rob.outlier_fraction == pytest.approx(len(bad) / len(obs), abs=0.02)


def test_fit_to_assigned_lines_recovers_constants(setup):
    """Close the loop: synthetic spectrum -> assignment -> fit to line positions."""
    m, res = setup
    mu = (0.2, 1.0, 0.3)
    peaks, _ = rd.synth.synthesize_spectrum(m, TRUE, mu, 150.0, fwhm=2e-3, snr=3000, spurious_density=0.2, seed=4)
    start = dict(zip(res.param_names, rd.posterior.sample_theta(res, 1, seed=6)[0]))
    pred, _ = assign.predict_lines(m, start, mu, 150.0, covariance=res.covariance, min_relative_intensity=1e-4)
    merged = assign.merge_blends(pred, 1e-3)
    a = assign.assign(merged, peaks)
    obs = assign.line_observations(a, merged, peaks)
    assert len(obs) > 300
    fit = rd.fit(m, obs, start)
    assert fit.converged and fit.certified
    pulls = [abs(fit.theta[n] - TRUE[n]) / fit.sigma[n] for n in TRUE]
    assert max(pulls) < 5, dict(zip(TRUE, pulls))
