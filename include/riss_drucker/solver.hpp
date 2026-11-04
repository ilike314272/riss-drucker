#pragma once
// Block solver with certificates (README 5.3). A result is "certified" only if
// every certificate passed; failures are reported in the result, never
// absorbed (README 10).
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "riss_drucker/inertia.hpp"
#include "riss_drucker/lanczos.hpp"
#include "riss_drucker/linalg_backend.hpp"
#include "riss_drucker/sparse.hpp"

namespace riss_drucker {

struct Certificate {
  std::string name;     // "inertia_count" | "residual" | "orthogonality" | ...
  bool passed = false;
  double value = 0.0;   // measured quantity
  double bound = 0.0;   // acceptance bound
  std::string detail;
};

enum class SolveMethod : int { Auto = 0, Dense = 1, Lanczos = 2 };
const char* to_string(SolveMethod m);
SolveMethod parse_solve_method(const std::string& s);

// Raised when a solve would exceed the memory budget (README 3.1: fail early).
struct BudgetError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SolveOptions {
  SolveMethod method = SolveMethod::Auto;
  // Lowest-k mode if nlowest >= 0; otherwise window mode [window_lo, window_hi).
  std::int64_t nlowest = -1;
  double window_lo = -std::numeric_limits<double>::infinity();
  double window_hi = std::numeric_limits<double>::infinity();
  std::int64_t dense_max_dim = 1500;      // Auto: dense at or below this dimension
  std::int64_t inertia_dense_max_dim = 6000;  // largest block for a dense BK certificate
  double memory_budget_bytes = 0.0;       // 0 = unlimited
  linalg::Backend backend = linalg::Backend::Eigen;
  LanczosOptions lanczos;
  double residual_tol = 1e-8;             // residual / ||H||_inf
  double orthogonality_tol = 1e-8;
  int max_widen = 6;
  // Lowest-k mode: levels closer than degeneracy_tol * ||H||_inf to level k
  // are returned with it, so a degenerate multiplet is never split.
  double degeneracy_tol = 1e-10;
  // Lanczos on -(H - sigma I)^{-1}, sigma below the spectrum (sparse Cholesky).
  // Falls back to plain Lanczos if the factorization fails.
  bool shift_invert = true;
  bool keep_vectors = true;
};

struct BlockCost {
  std::int64_t dim = 0, nnz = 0, bandwidth = 0;
  double dense_bytes = 0.0;    // dense eigh footprint
  double lanczos_bytes = 0.0;  // Krylov basis + locked vectors
  std::string recommended;     // "dense" | "lanczos"
  std::string inertia_method;  // "sturm" | "bunch_kaufman" | "unavailable"
};

struct BlockResult {
  std::string method;
  Eigen::VectorXd eigenvalues;   // ascending
  Eigen::MatrixXd eigenvectors;  // dim x nev (empty if !keep_vectors)
  Eigen::VectorXd residuals;     // ||Hx - lambda x|| / ||x||, scalar reference path
  std::vector<Certificate> certificates;
  bool certified = false;
  double window_lo = std::numeric_limits<double>::quiet_NaN();  // accepted window
  double window_hi = std::numeric_limits<double>::quiet_NaN();
  std::int64_t nev_requested = -1;
  int widen_steps = 0;
  std::int64_t boundary_extended = 0;  // extra levels added to keep a multiplet whole
  int lanczos_runs = 0;
  std::string lanczos_mode;  // "plain" | "shift_invert" (Lanczos path only)
  std::int64_t matvecs = 0;
  BlockCost cost;
  double gap_ratio_mean = std::numeric_limits<double>::quiet_NaN();
};

BlockCost estimate_cost(const CsrMatrix& h, const SolveOptions& opts);

// Mean consecutive-spacing ratio <r> = <min(s_i, s_i+1) / max(s_i, s_i+1)>
// over the given ascending levels (NaN if fewer than 3 or all spacings zero).
double gap_ratio_mean(const Eigen::VectorXd& levels);

// Count of eigenvalues < sigma with the method chosen by estimate_cost.
// Throws BudgetError if no certificate method fits.
Inertia count_below(const CsrMatrix& h, double sigma, const SolveOptions& opts);

BlockResult solve_block(const CsrMatrix& h, const SolveOptions& opts,
                        const Eigen::MatrixXd* warm = nullptr);

}  // namespace riss_drucker
