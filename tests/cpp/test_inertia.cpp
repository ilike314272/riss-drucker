// Inertia certificates vs the eigenvalue count, including matrices that force
// 2x2 Bunch-Kaufman pivots, singular shifts, and every built backend.
#include <algorithm>
#include <random>

#include "riss_drucker/inertia.hpp"
#include "test_harness.hpp"
#include "test_util.hpp"

using namespace riss_drucker;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
std::int64_t count_less(const VectorXd& w, double s) {
  return std::count_if(w.data(), w.data() + w.size(), [&](double x) { return x < s; });
}
}  // namespace

TEST_CASE("BK inertia matches eigenvalue counts on random indefinite matrices") {
  std::mt19937_64 rng(10);
  std::normal_distribution<double> d;
  for (int n : {1, 2, 3, 7, 30, 120}) {
    MatrixXd a(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j <= i; ++j) a(i, j) = a(j, i) = d(rng);
    const VectorXd w = Eigen::SelfAdjointEigenSolver<MatrixXd>(a).eigenvalues();
    for (double s : {-10.0, -1.3, -0.2, 0.0, 0.37, 2.0, 50.0}) {
      for (auto b : linalg::available_backends()) {
        const auto in = inertia_dense(a, s, b);
        CHECK(in.n_neg == count_less(w, s));
        CHECK(in.n_neg + in.n_zero + in.n_pos == n);
      }
    }
  }
}

TEST_CASE("BK handles zero diagonals (2x2 pivots) and exact singular shifts") {
  // [[0, 1], [1, 0]] needs a 2x2 pivot: eigenvalues -1, +1.
  MatrixXd a(2, 2);
  a << 0, 1, 1, 0;
  auto in = inertia_bunch_kaufman(a, 0.0);
  CHECK(in.n_neg == 1 && in.n_pos == 1 && in.n_zero == 0);
  // Block-diagonal mix of 2x2-pivot blocks and a zero row/column.
  MatrixXd b = MatrixXd::Zero(5, 5);
  b(1, 0) = b(0, 1) = 2.0;
  b(3, 2) = b(2, 3) = -1.0;
  b(4, 4) = 3.0;
  in = inertia_bunch_kaufman(b, 0.0);  // eigenvalues -2, 2, -1, 1, 3 ... and none zero
  CHECK(in.n_neg == 2 && in.n_pos == 3);
  MatrixXd z = MatrixXd::Zero(3, 3);
  z(2, 2) = 1.0;
  in = inertia_bunch_kaufman(z, 0.0);
  CHECK(in.n_zero == 2 && in.n_pos == 1 && in.min_abs_pivot == 0.0);
  for (auto bk : linalg::available_backends()) {
    const auto iz = inertia_dense(z, 0.0, bk);
    CHECK(iz.n_neg == 0 && iz.n_zero + iz.n_pos == 3);
  }
}

TEST_CASE("BK inertia on a rotated known spectrum, shifts between eigenvalues") {
  std::mt19937_64 rng(11);
  const int n = 60;
  VectorXd lam(n);
  for (int i = 0; i < n; ++i) lam[i] = -15.0 + 0.5 * i;
  const MatrixXd q = rd_test::random_orthogonal(n, rng);
  const MatrixXd a = q * lam.asDiagonal() * q.transpose();
  for (int i = 0; i < n; ++i)
    for (auto b : linalg::available_backends())
      CHECK(inertia_dense(a, lam[i] + 0.25, b).n_neg == i + 1);
}

TEST_CASE("Sturm count matches eigenvalue counts and BK on tridiagonal matrices") {
  std::mt19937_64 rng(12);
  std::normal_distribution<double> d;
  for (int n : {1, 2, 10, 200}) {
    VectorXd diag(n), off(std::max(n - 1, 0));
    for (int i = 0; i < n; ++i) diag[i] = d(rng);
    for (int i = 0; i + 1 < n; ++i) off[i] = d(rng);
    off.head(std::min<int>(n - 1, 3)).setZero();  // zero couplings are allowed
    MatrixXd t = diag.asDiagonal();
    for (int i = 0; i + 1 < n; ++i) t(i + 1, i) = t(i, i + 1) = off[i];
    const VectorXd w = Eigen::SelfAdjointEigenSolver<MatrixXd>(t).eigenvalues();
    for (double s : {-3.0, -0.5, 0.0, 0.1, 1.0, 3.0}) {
      CHECK(sturm_count(diag, off, s).n_neg == count_less(w, s));
      CHECK(inertia_bunch_kaufman(t, s).n_neg == count_less(w, s));
    }
  }
  CHECK_THROWS(sturm_count(VectorXd::Zero(3), VectorXd::Zero(3), 0.0));
}

RD_TEST_MAIN
