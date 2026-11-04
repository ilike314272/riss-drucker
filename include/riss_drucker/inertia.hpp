#pragma once
// Inertia certificates (README 5.3): the number of eigenvalues below a shift
// sigma, computed without diagonalizing.
//  * Dense symmetric blocks: Bunch-Kaufman LDL^T of (A - sigma I), Sylvester's
//    law of inertia on the block-diagonal D (1x1 and 2x2 pivots). In-house
//    implementation for the Eigen backend; LAPACK dsytrf for lapack/mkl.
//  * Tridiagonal blocks: Sturm sequence count.
// Counts are deterministic code, independent of the eigensolver they certify.
#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include "riss_drucker/linalg_backend.hpp"

namespace riss_drucker {

struct Inertia {
  std::int64_t n_neg = 0;   // eigenvalues < sigma
  std::int64_t n_zero = 0;  // exact zero pivots (sigma is an eigenvalue to working precision)
  std::int64_t n_pos = 0;
  // Smallest |eigenvalue of a pivot block|; small values mean sigma is close
  // to an eigenvalue and the count near sigma is sensitive to rounding.
  double min_abs_pivot = std::numeric_limits<double>::infinity();
  std::string method;
};

// In-house Bunch-Kaufman (partial pivoting, alpha = (1 + sqrt 17) / 8).
// Reads the lower triangle only.
Inertia inertia_bunch_kaufman(const Eigen::MatrixXd& a, double sigma);

// Dispatch on backend: Eigen -> in-house; Lapack/Mkl -> dsytrf.
Inertia inertia_dense(const Eigen::MatrixXd& a, double sigma, linalg::Backend backend);

// Sturm count for the symmetric tridiagonal matrix (diag, offdiag).
// Zero pivots are replaced by -pivmin as in LAPACK dstebz.
Inertia sturm_count(const Eigen::VectorXd& diag, const Eigen::VectorXd& offdiag, double sigma);

}  // namespace riss_drucker
