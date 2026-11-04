#pragma once
// Angular-momentum algebra (README 5.2): Wigner 3j/6j/9j and Clebsch-Gordan
// coefficients. All arguments are doubled (2j, 2m) so half-integers are exact.
//
// 3j and 6j are computed exactly: the Racah sums use prime-factorized
// factorials and an in-house big integer, giving the value as
// sign * sqrt(num/den) with integers num, den in lowest terms. The double
// result is rounded once from that exact form. 9j is a sum of products of
// exactly computed 6j symbols, so it is accurate to a few ulp. This choice was
// made by benchmark (docs/conventions.md, benchmarks/bench_angular.py).
#include <cstdint>
#include <string>

namespace riss_drucker::angular {

struct ExactValue {
  int sign = 0;          // -1, 0, +1
  std::string num = "0"; // decimal integer
  std::string den = "1"; // decimal integer, value = sign * sqrt(num / den)
  double value = 0.0;
};

ExactValue wigner3j_exact(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3);
double wigner3j(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3);

ExactValue wigner6j_exact(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6);
double wigner6j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6);

// {j1 j2 j3; j4 j5 j6; j7 j8 j9}
double wigner9j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6, int tj7, int tj8, int tj9);

// <j1 m1 j2 m2 | J M>, Condon-Shortley phase.
double clebsch_gordan(int tj1, int tm1, int tj2, int tm2, int tJ, int tM);

// Sum-rule certificates (README 5.2): maximum deviation over all arguments
// with 2j <= two_jmax.
double cg_orthogonality_defect(int two_jmax);   // sum_m1m2 <..|JM><..|J'M'> = delta
double cg_completeness_defect(int two_jmax);    // sum_JM <m1m2|JM><m1'm2'|JM> = delta
double sixj_orthogonality_defect(int two_jmax); // sum_x (2x+1)(2f+1){abx;cdf}{abx;cdf'} = delta

}  // namespace riss_drucker::angular
