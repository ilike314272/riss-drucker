// Angular-momentum algebra: closed forms, exact values, symmetries, sum rules,
// and large-j behaviour (README 12, unit tests).
#include <cmath>
#include <string>

#include "riss_drucker/angular.hpp"
#include "test_harness.hpp"

using namespace riss_drucker::angular;

TEST_CASE("3j closed forms and exact representation") {
  // (1 1 1; 1 -1 0) = 1/sqrt(6)
  const auto v = wigner3j_exact(2, 2, 2, 2, -2, 0);
  CHECK(v.sign == 1 && v.num == "1" && v.den == "6");
  CHECK_CLOSE(v.value, 1.0 / std::sqrt(6.0), 1e-16);
  // (j j 0; m -m 0) = (-1)^{j-m} / sqrt(2j+1), integer and half-integer j
  for (int tj = 0; tj <= 40; ++tj)
    for (int tm = -tj; tm <= tj; tm += 2) {
      const double ph = (((tj - tm) / 2) % 2 == 0) ? 1.0 : -1.0;
      CHECK_CLOSE(wigner3j(tj, tj, 0, tm, -tm, 0), ph / std::sqrt(tj + 1.0), 2e-16);
    }
  // (1/2 1/2 1; 1/2 -1/2 0) = 1/sqrt(6)
  CHECK_CLOSE(wigner3j(1, 1, 2, 1, -1, 0), 1.0 / std::sqrt(6.0), 1e-16);
  // Selection rules give exact zeros.
  CHECK(wigner3j(2, 2, 2, 2, 2, -2) == 0.0);   // m sum != 0
  CHECK(wigner3j(2, 2, 6, 0, 0, 0) == 0.0);    // triangle violated
  CHECK(wigner3j(2, 2, 2, 0, 0, 0) == 0.0);    // odd J sum with all m = 0
  CHECK(wigner3j_exact(2, 2, 2, 2, 0, 0).sign == 0);  // m sum != 0
}

TEST_CASE("3j symmetries: column permutations and m sign flip") {
  for (int a = 0; a <= 8; ++a)
    for (int b = 0; b <= 8; ++b)
      for (int c = std::abs(a - b); c <= a + b; c += 2)
        for (int ma = -a; ma <= a; ma += 2)
          for (int mb = -b; mb <= b; mb += 2) {
            const int mc = -ma - mb;
            if (std::abs(mc) > c) continue;
            const double v = wigner3j(a, b, c, ma, mb, mc);
            const double odd = (((a + b + c) / 2) % 2 == 0) ? 1.0 : -1.0;
            CHECK_CLOSE(wigner3j(b, c, a, mb, mc, ma), v, 1e-15);        // cyclic
            CHECK_CLOSE(wigner3j(b, a, c, mb, ma, mc), odd * v, 1e-15);  // odd permutation
            CHECK_CLOSE(wigner3j(a, b, c, -ma, -mb, -mc), odd * v, 1e-15);
          }
}

TEST_CASE("6j closed forms, symmetry and exactness") {
  // {a b c; 0 c b} = (-1)^{a+b+c} / sqrt((2b+1)(2c+1))
  for (int a = 0; a <= 10; ++a)
    for (int b = 0; b <= 10; ++b)
      for (int c = std::abs(a - b); c <= a + b; c += 2) {
        const double ph = (((a + b + c) / 2) % 2 == 0) ? 1.0 : -1.0;
        CHECK_CLOSE(wigner6j(a, b, c, 0, c, b), ph / std::sqrt((b + 1.0) * (c + 1.0)), 1e-15);
      }
  // {1 1 1; 1 1 1} = 1/6
  const auto e = wigner6j_exact(2, 2, 2, 2, 2, 2);
  CHECK(e.sign == 1 && e.num == "1" && e.den == "36");
  // column permutation and upper/lower swap of two columns
  CHECK_CLOSE(wigner6j(2, 4, 4, 6, 2, 4), wigner6j(4, 2, 4, 2, 6, 4), 1e-15);
  CHECK_CLOSE(wigner6j(2, 4, 4, 6, 2, 4), wigner6j(6, 2, 4, 2, 4, 4), 1e-15);
}

TEST_CASE("9j reduces to 6j when one entry is zero") {
  // {a b e; c d e; f f 0} = (-1)^{b+c+e+f} {a b e; d c f} / sqrt((2e+1)(2f+1))
  for (int a = 0; a <= 4; ++a)
    for (int b = 0; b <= 4; ++b)
      for (int c = 0; c <= 4; ++c)
        for (int d = 0; d <= 4; ++d)
          for (int e = std::abs(a - b); e <= a + b; e += 2)
            for (int f = std::abs(a - c); f <= a + c; f += 2) {
              const double s6 = wigner6j(a, b, e, d, c, f);
              const int ph2 = b + c + e + f;
              if (ph2 % 2) continue;
              const double ph = ((ph2 / 2) % 2 == 0) ? 1.0 : -1.0;
              CHECK_CLOSE(wigner9j(a, b, e, c, d, e, f, f, 0), ph * s6 / std::sqrt((e + 1.0) * (f + 1.0)), 1e-14);
            }
}

TEST_CASE("sum rules: CG orthogonality and completeness, 6j orthogonality") {
  CHECK(cg_orthogonality_defect(8) < 1e-14);
  CHECK(cg_completeness_defect(8) < 1e-14);
  CHECK(sixj_orthogonality_defect(5) < 1e-14);
}

TEST_CASE("large j: normalization holds where naive double Racah sums fail") {
  // sum_{m1} (2J+1) (j1 j2 J; m1 M-m1 -M)^2 = 1 for j1 = j2 = 60, J = 60.
  const int t = 120;
  double s = 0.0;
  for (int tm1 = -t; tm1 <= t; tm1 += 2) s += (t + 1.0) * std::pow(wigner3j(t, t, t, tm1, -tm1, 0), 2);
  CHECK_CLOSE(s, 1.0, 1e-13);
  // Half-integer large j.
  s = 0.0;
  for (int tm1 = -101; tm1 <= 101; tm1 += 2) s += 201.0 * std::pow(wigner3j(101, 101, 200, tm1, -tm1, 0), 2);
  CHECK_CLOSE(s, 1.0, 1e-13);
}

TEST_CASE("invalid arguments") { CHECK_THROWS(wigner3j(-2, 2, 2, 0, 0, 0)); }

RD_TEST_MAIN
