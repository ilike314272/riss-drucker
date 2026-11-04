"""Line strengths: sum rule, C2 selection rules, closed forms; intensities."""
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker import spectra


def full_solution(axis="Ir", j_max=6, th=None):
    m = rd.compile_asym_rotor(j_max=j_max, order=2, axis=axis)
    th = th or {"A": 3.0, "B": 2.0, "C": 1.0}
    return m, rd.solve(m, th, nlowest=2 * j_max + 1, keep_vectors=True, method="dense")


@pytest.mark.parametrize("axis", ["Ir", "IIr", "IIIr"])
def test_sum_rule_and_selection_rules(axis):
    m, sol = full_solution(axis)
    for mu, allowed in [((1, 0, 0), {"a"}), ((0, 1, 0), {"b"}), ((0, 0, 1), {"c"}), ((0.3, 0.5, 0.7), {"a", "b", "c"})]:
        assert spectra.strength_sum_rule_defect(m, sol, mu) < 1e-12
        lines = spectra.line_strengths(m, sol, mu)
        assert {spectra.band_type(l.lower_label, l.upper_label) for l in lines} == allowed
        j0 = {l.upper_label: l.strength for l in lines if l.j_lower == 0}
        expect = {"1(0,1)": mu[0] ** 2, "1(1,1)": mu[1] ** 2, "1(1,0)": mu[2] ** 2}
        for lab, s in expect.items():
            if s > 0:
                assert j0[lab] == pytest.approx(s, rel=1e-12)


def test_symmetric_top_honl_london():
    """Prolate top (B = C), a-type R branch: sum over the K doublet = ((J+1)^2 - K^2)/(J+1)."""
    m, sol = full_solution(th={"A": 5.0, "B": 1.2, "C": 1.2}, j_max=7)
    lines = spectra.line_strengths(m, sol, (1.0, 0, 0))
    tot = {}
    for l in lines:
        if l.j_upper == l.j_lower + 1:
            ka = int(l.lower_label.split("(")[1].split(",")[0])
            tot[(l.j_lower, ka)] = tot.get((l.j_lower, ka), 0.0) + l.strength
    for (J, K), s in tot.items():
        n_doublet = 1 if K == 0 else 2
        assert s == pytest.approx(n_doublet * ((J + 1) ** 2 - K * K) / (J + 1), rel=1e-10)


def test_intensities_formula_and_partition_function():
    m, sol = full_solution(j_max=5)
    lines = spectra.line_strengths(m, sol, (0.5, 1.0, 0.0))
    T = 20.0
    Q = spectra.add_intensities(lines, sol, T)
    e0 = min(min(r["eigenvalues"]) for r in sol.results)
    q_ref = sum((b["two_j"] + 1) * math.exp(-spectra.C2 * (e - e0) / T)
                for b, r in zip(sol.blocks, sol.results) for e in r["eigenvalues"])
    assert Q == pytest.approx(q_ref, rel=1e-14)
    l = lines[len(lines) // 2]
    ref = l.strength * l.wavenumber * math.exp(-spectra.C2 * (l.lower_energy - e0) / T) * (
        1 - math.exp(-spectra.C2 * l.wavenumber / T)) / Q
    assert l.intensity == pytest.approx(ref, rel=1e-14)
    with pytest.raises(ValueError):
        spectra.line_strengths(m, rd.solve(m, {"A": 3, "B": 2, "C": 1}, nlowest=2), (1, 0, 0))
