"""Wigner 3j/6j/9j and Clebsch-Gordan coefficients (C++ core).

Arguments are angular momenta as numbers (integers or half-integers, e.g. 0.5
or Fraction(1, 2)). 3j and 6j are exact up to one final rounding; the
``*_exact`` variants return (sign, Fraction) with value = sign * sqrt(Fraction).
"""
from __future__ import annotations

from fractions import Fraction

from . import _core


def _two(x) -> int:
    t = Fraction(x) * 2
    if t.denominator != 1:
        raise ValueError(f"{x} is not an integer or half-integer")
    return int(t)


def wigner3j(j1, j2, j3, m1, m2, m3) -> float:
    return _core.wigner3j_2(*map(_two, (j1, j2, j3, m1, m2, m3)))


def wigner6j(j1, j2, j3, j4, j5, j6) -> float:
    return _core.wigner6j_2(*map(_two, (j1, j2, j3, j4, j5, j6)))


def wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, j9) -> float:
    return _core.wigner9j_2(*map(_two, (j1, j2, j3, j4, j5, j6, j7, j8, j9)))


def clebsch_gordan(j1, m1, j2, m2, J, M) -> float:
    return _core.clebsch_gordan_2(*map(_two, (j1, m1, j2, m2, J, M)))


def _exact(t) -> tuple[int, Fraction]:
    sign, num, den, _ = t
    return sign, Fraction(int(num), int(den))


def wigner3j_exact(j1, j2, j3, m1, m2, m3) -> tuple[int, Fraction]:
    """(sign, r) with 3j = sign * sqrt(r), r a reduced Fraction."""
    return _exact(_core.wigner3j_exact_2(*map(_two, (j1, j2, j3, m1, m2, m3))))


def wigner6j_exact(j1, j2, j3, j4, j5, j6) -> tuple[int, Fraction]:
    return _exact(_core.wigner6j_exact_2(*map(_two, (j1, j2, j3, j4, j5, j6))))


def sum_rule_defects(jmax) -> dict:
    """Sum-rule certificates up to j <= jmax: maximum deviations."""
    t = _two(jmax)
    return {"cg_orthogonality": _core.cg_orthogonality_defect(t),
            "cg_completeness": _core.cg_completeness_defect(t),
            "sixj_orthogonality": _core.sixj_orthogonality_defect(min(t, 5))}
