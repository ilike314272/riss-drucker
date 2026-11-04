#pragma once
// Lanczos with full reorthogonalization and explicit locking/deflation
// (README 5.3). Each run starts from a vector orthogonal to the locked pairs,
// so a second run finds the second copy of a degenerate eigenvalue that a
// single Krylov space cannot see. The caller (solve_block) decides how many
// pairs are needed from an inertia count.
#include <cstdint>
#include <functional>
#include <string>

#include <Eigen/Dense>

#include "riss_drucker/sparse.hpp"

namespace riss_drucker {

struct LanczosOptions {
  int max_basis = 200;          // Krylov steps per run (capped by the free dimension)
  int max_runs = 40;            // runs (restarts) before giving up
  double tol = 1e-10;           // Ritz residual tolerance relative to ||H||_inf
  std::uint64_t seed = 0;       // start-vector RNG seed (deterministic per block)
};

// Locked (converged) eigenpairs; persists across calls so a solve can be extended.
struct LanczosState {
  Eigen::VectorXd values;    // ascending
  Eigen::MatrixXd vectors;   // n x nlocked, orthonormal
  int runs = 0;
  std::int64_t matvecs = 0;
  std::int64_t locked() const { return values.size(); }
};

// Symmetric linear operator y = A x. `norm_hint` (> 0) scales the convergence
// tolerance; with 0 the scale is estimated from the Lanczos coefficients
// (used for shift-invert, whose norm is not known in advance).
struct LinearOperator {
  std::int64_t n = 0;
  double norm_hint = 0.0;
  std::function<void(const double*, double*)> apply;
};
LinearOperator as_operator(const CsrMatrix& h);

// Extends `state` until it holds at least `target` locked pairs taken from the
// low end of the spectrum, or the run budget is exhausted. `warm` columns (if
// any) seed the first run of a fresh state. Returns true if target was reached.
bool lanczos_extend_lowest(const LinearOperator& op, std::int64_t target, const LanczosOptions& opts,
                           LanczosState& state, const Eigen::MatrixXd* warm = nullptr);
inline bool lanczos_extend_lowest(const CsrMatrix& h, std::int64_t target, const LanczosOptions& opts,
                                  LanczosState& state, const Eigen::MatrixXd* warm = nullptr) {
  return lanczos_extend_lowest(as_operator(h), target, opts, state, warm);
}

}  // namespace riss_drucker
