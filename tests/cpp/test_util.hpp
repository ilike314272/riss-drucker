#pragma once
#include <random>

#include <Eigen/Dense>

#include "riss_drucker/sparse.hpp"

namespace rd_test {

inline Eigen::MatrixXd random_orthogonal(int n, std::mt19937_64& rng) {
  std::normal_distribution<double> d;
  Eigen::MatrixXd g(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) g(i, j) = d(rng);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(g);
  return qr.householderQ();
}

// Q diag(lam) Q^T as CSR (dense pattern).
inline riss_drucker::CsrMatrix csr_from_dense(const Eigen::MatrixXd& a) {
  std::vector<riss_drucker::Triplet> t;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      if (a(i, j) != 0.0) t.push_back({i, j, a(i, j)});
  return riss_drucker::CsrMatrix::from_triplets(a.rows(), a.cols(), std::move(t));
}

// Random sparse symmetric banded matrix.
inline riss_drucker::CsrMatrix random_banded(int n, int bw, std::mt19937_64& rng) {
  std::normal_distribution<double> d;
  std::vector<riss_drucker::Triplet> t;
  for (int i = 0; i < n; ++i) {
    t.push_back({i, i, 4.0 * d(rng)});
    for (int k = 1; k <= bw && i + k < n; ++k) {
      const double v = d(rng);
      t.push_back({i + k, i, v});
      t.push_back({i, i + k, v});
    }
  }
  return riss_drucker::CsrMatrix::from_triplets(n, n, std::move(t));
}

}  // namespace rd_test
