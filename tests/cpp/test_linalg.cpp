// Every built linalg backend must pass the same eigh tests and agree.
#include <random>

#include "riss_drucker/linalg_backend.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
MatrixXd random_orthogonal(int n, std::mt19937_64& rng) {
  std::normal_distribution<double> d;
  MatrixXd g(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) g(i, j) = d(rng);
  Eigen::HouseholderQR<MatrixXd> qr(g);
  return qr.householderQ();
}
}  // namespace

TEST_CASE("eigh recovers a known spectrum on every backend") {
  std::mt19937_64 rng(42);
  for (int n : {1, 2, 5, 40, 200}) {
    VectorXd lam(n);
    for (int i = 0; i < n; ++i) lam[i] = -3.0 + 0.5 * i + 1e-3 * i * i;  // sorted, distinct
    const MatrixXd q = random_orthogonal(n, rng);
    const MatrixXd a = q * lam.asDiagonal() * q.transpose();
    const double scale = lam.cwiseAbs().maxCoeff();
    for (auto b : linalg::available_backends()) {
      const auto r = linalg::eigh(a, b, true);
      CHECK(r.eigenvalues.size() == n);
      CHECK_CLOSE((r.eigenvalues - lam).cwiseAbs().maxCoeff(), 0.0, 1e-12 * n * scale);
      // Residual ||A V - V diag(w)|| and orthonormality.
      const MatrixXd res = a * r.eigenvectors - r.eigenvectors * r.eigenvalues.asDiagonal();
      CHECK_CLOSE(res.norm(), 0.0, 1e-12 * n * scale);
      const MatrixXd ortho = r.eigenvectors.transpose() * r.eigenvectors - MatrixXd::Identity(n, n);
      CHECK_CLOSE(ortho.norm(), 0.0, 1e-12 * n);
      // Eigenvalues-only path gives the same values.
      const auto w = linalg::eigh(a, b, false);
      CHECK(w.eigenvectors.size() == 0);
      CHECK_CLOSE((w.eigenvalues - r.eigenvalues).cwiseAbs().maxCoeff(), 0.0, 1e-12 * n * scale);
    }
  }
}

TEST_CASE("backends agree with each other") {
  std::mt19937_64 rng(5);
  std::normal_distribution<double> d;
  MatrixXd a(80, 80);
  for (int i = 0; i < 80; ++i)
    for (int j = 0; j <= i; ++j) a(i, j) = a(j, i) = d(rng);
  const auto ref = linalg::eigh(a, linalg::Backend::Eigen, false);
  for (auto b : linalg::available_backends()) {
    const auto r = linalg::eigh(a, b, false);
    CHECK_CLOSE((r.eigenvalues - ref.eigenvalues).cwiseAbs().maxCoeff(), 0.0, 1e-11);
  }
}

TEST_CASE("only the lower triangle is read") {
  MatrixXd a(3, 3);
  a << 2, 999, 999, 1, 3, 999, 0, 1, 4;  // garbage above the diagonal
  MatrixXd s = a.triangularView<Eigen::Lower>();
  s = s.selfadjointView<Eigen::Lower>();
  const auto ref = linalg::eigh(s, linalg::Backend::Eigen, false);
  for (auto b : linalg::available_backends())
    CHECK_CLOSE((linalg::eigh(a, b, false).eigenvalues - ref.eigenvalues).cwiseAbs().maxCoeff(), 0.0, 1e-13);
}

TEST_CASE("errors: non-square input, unbuilt backend, empty matrix") {
  CHECK_THROWS(linalg::eigh(MatrixXd(2, 3), linalg::Backend::Eigen));
  for (auto b : {linalg::Backend::Lapack, linalg::Backend::Mkl})
    if (!linalg::backend_available(b)) CHECK_THROWS(linalg::eigh(MatrixXd::Identity(2, 2), b));
  CHECK(linalg::eigh(MatrixXd(0, 0), linalg::Backend::Eigen).eigenvalues.size() == 0);
  CHECK_THROWS(linalg::parse_backend("cusolver"));
}

TEST_CASE("dense memory estimate scales as N^2") {
  // README 3.1: N = 20000 dense is ~3.2 GB for one matrix; eigh needs several.
  CHECK(linalg::dense_eigh_bytes(20000) >= 3.2e9);
  CHECK_CLOSE(linalg::dense_eigh_bytes(2000) * 100.0, linalg::dense_eigh_bytes(20000), 1.0);
}

RD_TEST_MAIN
