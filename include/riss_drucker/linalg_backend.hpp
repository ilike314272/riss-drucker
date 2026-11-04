#pragma once
// Single abstraction over dense linear-algebra backends (README section 3.1).
// Eigen is always built and is the default; LAPACK/OpenBLAS and oneMKL are
// optional CMake options. All backends must pass the same tests.
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace riss_drucker::linalg {

enum class Backend : int { Eigen = 0, Lapack = 1, Mkl = 2 };

const char* to_string(Backend b);
Backend parse_backend(const std::string& name);  // throws std::invalid_argument
std::vector<Backend> available_backends();
bool backend_available(Backend b);
// RISS_DRUCKER_LINALG if set and available (else throws), otherwise Eigen.
Backend default_backend();

struct EighResult {
  Eigen::VectorXd eigenvalues;   // ascending
  Eigen::MatrixXd eigenvectors;  // columns; empty if not requested
};

// Symmetric eigendecomposition. Only the lower triangle of `a` is read.
// Throws std::invalid_argument for non-square input, std::runtime_error if the
// backend reports failure.
EighResult eigh(const Eigen::MatrixXd& a, Backend backend, bool compute_vectors = true);
inline EighResult eigh(const Eigen::MatrixXd& a, bool compute_vectors = true) {
  return eigh(a, default_backend(), compute_vectors);
}

// Bytes needed for a dense eigendecomposition of dimension n (matrix copy,
// eigenvectors, workspace estimate). Used to fail early against a memory budget.
double dense_eigh_bytes(long n);

}  // namespace riss_drucker::linalg
