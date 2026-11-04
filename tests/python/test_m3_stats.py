"""Level statistics (README 12): beta-Hermite <r>, Brody, superposition P(0)."""
import math

import numpy as np
import pytest

from riss_drucker import stats

pytest.importorskip("scipy")


@pytest.mark.parametrize("beta,name", [(0, "poisson"), (1, "goe"), (2, "gue"), (4, "gse")])
def test_beta_hermite_gap_ratio(beta, name):
    rng = np.random.default_rng(beta)
    r = np.concatenate([stats.gap_ratios(stats.central(stats.beta_hermite_levels(600, beta, rng)))
                        for _ in range(25)])
    assert r.mean() == pytest.approx(stats.GAP_RATIO_REFERENCE[name], abs=0.012)


@pytest.mark.parametrize("beta,name", [(1, "goe"), (2, "gue")])
def test_circular_ensembles_gap_ratio(beta, name):
    rng = np.random.default_rng(10 + beta)
    r = np.concatenate([stats.gap_ratios(stats.circular_phases(300, beta, rng)) for _ in range(20)])
    assert r.mean() == pytest.approx(stats.GAP_RATIO_REFERENCE[name], abs=0.012)


@pytest.mark.parametrize("q", [0.0, 0.3, 0.7, 1.0])
def test_brody_sampler_and_fit(q):
    rng = np.random.default_rng(int(10 * q))
    s = stats.brody_spacings(40000, q, rng)
    assert s.mean() == pytest.approx(1.0, abs=0.02)
    assert stats.fit_brody(s) == pytest.approx(q, abs=0.03)


def test_poisson_brody_gap_ratio():
    rng = np.random.default_rng(5)
    levels = np.cumsum(stats.brody_spacings(60000, 0.0, rng))
    assert stats.mean_gap_ratio(levels) == pytest.approx(stats.GAP_RATIO_REFERENCE["poisson"], abs=0.01)


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.7, 0.3), (0.4, 0.3, 0.3)])
def test_superposition_p0(fractions):
    """Independent COE sequences superposed: P(0) = 1 - sum f_i^2."""
    rng = np.random.default_rng(7)
    total = 300
    spacings = []
    for _ in range(120):
        phases = np.concatenate([stats.circular_phases(int(round(f * total)), 1, rng) for f in fractions])
        phases.sort()
        s = np.diff(np.concatenate([phases, [phases[0] + 2 * math.pi]])) * len(phases) / (2 * math.pi)
        spacings.append(s)
    p0 = stats.spacing_density_at_zero(np.concatenate(spacings))
    assert p0 == pytest.approx(stats.superposition_p0(fractions), abs=0.06)


def test_validation():
    with pytest.raises(ValueError):
        stats.superposition_p0([0.5, 0.4])
    with pytest.raises(ValueError):
        stats.brody_spacings(10, 1.5, np.random.default_rng())
    with pytest.raises(ValueError):
        stats.beta_hermite_levels(10, -1, np.random.default_rng())
