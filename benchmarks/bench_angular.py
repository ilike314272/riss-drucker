#!/usr/bin/env python3
"""Exact angular-algebra decision benchmark (README 3.1: "decide in M2 by benchmark").

Candidates for 3j symbols:
  cpp_exact     riss_drucker C++: prime-factorized Racah sum + in-house big integer
  python_exact  pure-Python Racah sum with exact integers (stdlib only)
  sympy_exact   sympy.physics.wigner.wigner_3j (exact, then evaluated)
  naive_double  floating-point Racah sum (what you get without exact arithmetic)

Accuracy reference: SymPy exact values evaluated to 40 significant digits.
Prints JSON lines; pass --out to save them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from fractions import Fraction

import riss_drucker as rd
from riss_drucker import _core


def racah_terms(tj1, tj2, tj3, tm1, tm2, tm3):
    a1, a2 = (tj3 - tj2 + tm1) // 2, (tj3 - tj1 - tm2) // 2
    b1, b2, b3 = (tj1 + tj2 - tj3) // 2, (tj1 - tm1) // 2, (tj2 + tm2) // 2
    ks = range(max(0, -a1, -a2), min(b1, b2, b3) + 1)
    return ks, (a1, a2, b1, b2, b3)


def valid(tj1, tj2, tj3, tm1, tm2, tm3):
    if tm1 + tm2 + tm3 or (tj1 + tj2 + tj3) % 2:
        return False
    if not abs(tj1 - tj2) <= tj3 <= tj1 + tj2:
        return False
    return all(abs(m) <= j and (j + m) % 2 == 0 for j, m in ((tj1, tm1), (tj2, tm2), (tj3, tm3)))


def python_exact(tj1, tj2, tj3, tm1, tm2, tm3) -> float:
    if not valid(tj1, tj2, tj3, tm1, tm2, tm3):
        return 0.0
    f = math.factorial
    pref = Fraction(f((tj1 + tj2 - tj3) // 2) * f((tj1 - tj2 + tj3) // 2) * f((-tj1 + tj2 + tj3) // 2),
                    f((tj1 + tj2 + tj3) // 2 + 1))
    for tj, tm in ((tj1, tm1), (tj2, tm2), (tj3, tm3)):
        pref *= f((tj + tm) // 2) * f((tj - tm) // 2)
    ks, (a1, a2, b1, b2, b3) = racah_terms(tj1, tj2, tj3, tm1, tm2, tm3)
    s = sum(Fraction((-1) ** k, f(k) * f(a1 + k) * f(a2 + k) * f(b1 - k) * f(b2 - k) * f(b3 - k)) for k in ks)
    sign = -1 if ((tj1 - tj2 - tm3) // 2) % 2 else 1
    sq = s * s * pref  # value^2, exact
    return sign * (1 if s > 0 else -1) * math.sqrt(sq.numerator / sq.denominator) if s else 0.0


def naive_double(tj1, tj2, tj3, tm1, tm2, tm3) -> float:
    if not valid(tj1, tj2, tj3, tm1, tm2, tm3):
        return 0.0
    lf = lambda n: math.lgamma(n + 1)
    lp = 0.5 * (lf((tj1 + tj2 - tj3) // 2) + lf((tj1 - tj2 + tj3) // 2) + lf((-tj1 + tj2 + tj3) // 2)
                - lf((tj1 + tj2 + tj3) // 2 + 1))
    for tj, tm in ((tj1, tm1), (tj2, tm2), (tj3, tm3)):
        lp += 0.5 * (lf((tj + tm) // 2) + lf((tj - tm) // 2))
    ks, (a1, a2, b1, b2, b3) = racah_terms(tj1, tj2, tj3, tm1, tm2, tm3)
    s = sum((-1) ** k * math.exp(lp - (lf(k) + lf(a1 + k) + lf(a2 + k) + lf(b1 - k) + lf(b2 - k) + lf(b3 - k)))
            for k in ks)
    return (-1 if ((tj1 - tj2 - tm3) // 2) % 2 else 1) * s


def workload(name, rng):
    if name == "small_j<=6":
        out = []
        for t1 in range(0, 13):
            for t2 in range(0, 13):
                for t3 in range(abs(t1 - t2), t1 + t2 + 1, 2):
                    for m1 in range(-t1, t1 + 1, 2):
                        for m2 in range(-t2, t2 + 1, 2):
                            if abs(m1 + m2) <= t3:
                                out.append((t1, t2, t3, m1, m2, -m1 - m2))
        return out
    if name == "large_j40-80":
        out = []
        while len(out) < 150:
            t1, t2 = 2 * rng.randint(40, 80), 2 * rng.randint(40, 80)
            t3 = rng.randrange(abs(t1 - t2), t1 + t2 + 1, 2)
            m1, m2 = 2 * rng.randint(-t1 // 2, t1 // 2), 2 * rng.randint(-t2 // 2, t2 // 2)
            if abs(m1 + m2) <= t3:
                out.append((t1, t2, t3, m1, m2, -m1 - m2))
        return out
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--accuracy-samples", type=int, default=60)
    args = ap.parse_args()
    rng = random.Random(7)
    try:
        from sympy import Rational, N
        from sympy.physics.wigner import wigner_3j
    except ImportError:
        wigner_3j = None
    cands = {"cpp_exact": lambda *a: _core.wigner3j_2(*a), "python_exact": python_exact,
             "naive_double": naive_double}
    if wigner_3j is not None:
        cands["sympy_exact"] = lambda *a: float(wigner_3j(*(Rational(x, 2) for x in a)))
    lines = []
    for wname in ("small_j<=6", "large_j40-80"):
        args_list = workload(wname, rng)
        sample = rng.sample(args_list, min(args.accuracy_samples, len(args_list)))
        ref = None
        if wigner_3j is not None:
            ref = [N(wigner_3j(*(Rational(x, 2) for x in a)), 40) for a in sample]
        for cname, fn in cands.items():
            subset = args_list if cname != "sympy_exact" else args_list[: max(1, len(args_list) // 10)]
            t0 = time.perf_counter()
            for a in subset:
                fn(*a)
            per_call = (time.perf_counter() - t0) / len(subset)
            err = None
            if ref is not None:
                err = 0.0
                for a, r in zip(sample, ref):
                    v = fn(*a)
                    scale = max(abs(float(r)), 1e-300)
                    err = max(err, float(abs(v - r)) / scale if r != 0 else abs(v))
            line = {"workload": wname, "candidate": cname, "n_calls": len(subset), "seconds_per_call": per_call,
                    "max_rel_error_vs_sympy_40digits": err}
            lines.append(line)
            print(json.dumps(line))
    lines.append({"version": rd.__version__, "git_commit": rd.build_info()["git_commit"]})
    if args.out:
        with open(args.out, "w") as fh:
            for line in lines:
                fh.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
