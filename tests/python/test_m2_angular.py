"""Angular algebra vs SymPy exact values (README 5.2: SymPy cross-checks)."""
import random
from fractions import Fraction

import pytest

import riss_drucker as rd
from riss_drucker import angular

sympy = pytest.importorskip("sympy")
from sympy import Rational  # noqa: E402
from sympy.physics import wigner  # noqa: E402

H = Fraction(1, 2)


def as_frac_sq(expr):
    """SymPy exact value -> (sign, value^2 as Fraction)."""
    if expr == 0:
        return 0, Fraction(0)
    sq = sympy.expand(expr**2)
    assert sq.is_Rational, sq
    return (1 if expr > 0 else -1), Fraction(int(sq.p), int(sq.q))


def test_3j_exact_matches_sympy():
    rng = random.Random(1)
    cases = 0
    while cases < 150:
        j1, j2 = rng.randint(0, 16) * H, rng.randint(0, 16) * H
        j3 = rng.choice([abs(j1 - j2) + k for k in range(int(j1 + j2 - abs(j1 - j2)) + 1)])
        m1 = rng.choice([-j1 + k for k in range(int(2 * j1) + 1)])
        m2 = rng.choice([-j2 + k for k in range(int(2 * j2) + 1)])
        m3 = -m1 - m2
        if abs(m3) > j3:
            continue
        ref = wigner.wigner_3j(*(Rational(x.numerator, x.denominator) for x in (j1, j2, j3, m1, m2, m3)))
        assert angular.wigner3j_exact(j1, j2, j3, m1, m2, m3) == as_frac_sq(ref)
        assert angular.wigner3j(j1, j2, j3, m1, m2, m3) == pytest.approx(float(ref), rel=4e-16, abs=1e-300)
        cases += 1


def test_6j_and_9j_match_sympy():
    rng = random.Random(2)
    n6 = n9 = 0
    while n6 < 60:
        a = [rng.randint(0, 8) for _ in range(6)]
        ref = wigner.wigner_6j(*a)
        if ref == 0:
            continue
        assert angular.wigner6j_exact(*a) == as_frac_sq(ref)
        n6 += 1
    while n9 < 25:
        a = [rng.randint(0, 3) for _ in range(9)]
        ref = wigner.wigner_9j(*a, prec=40)
        if ref == 0:
            continue
        assert angular.wigner9j(*a) == pytest.approx(float(ref), rel=1e-13, abs=1e-15)
        n9 += 1


def test_clebsch_gordan_and_sum_rules():
    # sympy order: (j1, j2, j3, m1, m2, m3)
    ref = wigner.clebsch_gordan(Rational(3, 2), Rational(3, 2), 1, Rational(1, 2), -Rational(1, 2), 0)
    assert angular.clebsch_gordan(1.5, 0.5, 1.5, -0.5, 1, 0) == pytest.approx(float(ref), rel=1e-15)
    d = angular.sum_rule_defects(3)
    assert all(v < 1e-13 for v in d.values()), d
    with pytest.raises(ValueError):
        angular.wigner3j(0.3, 1, 1, 0, 0, 0)
