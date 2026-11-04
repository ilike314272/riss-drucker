#pragma once
// A compiled model: per-block operator lists aligned with the parameter names.
// Operators are compiled once per model structure; a fit only changes theta.
#include <cstdint>
#include <string>
#include <vector>

#include "riss_drucker/block_id.hpp"
#include "riss_drucker/solver.hpp"
#include "riss_drucker/sparse.hpp"

namespace riss_drucker {

struct CompiledBlock {
  BlockId id;
  std::vector<CsrMatrix> ops;              // aligned with CompiledModel::param_names
  std::vector<std::string> basis_labels;   // one per basis function
  std::int64_t dim() const { return ops.empty() ? 0 : ops.front().rows; }
};

struct CompiledModel {
  std::string name;
  std::string spec_json;                   // canonical structure description
  std::vector<std::string> param_names;
  std::vector<CompiledBlock> blocks;
  std::vector<Certificate> compile_certificates;
};

CsrMatrix assemble(const CompiledBlock& block, const std::vector<double>& theta);

// Hellmann-Feynman Jacobian dE_i/dt_k = <psi_i|O_k|psi_i> for the columns of
// `vectors` (normalized eigenvectors of H(theta) in this block). Never
// differentiates through the eigensolver (README 5.4). For exactly degenerate
// levels individual rows depend on the basis chosen in the multiplet; sums over
// the multiplet do not.
Eigen::MatrixXd hf_jacobian(const CompiledBlock& block, const Eigen::MatrixXd& vectors);

// Solves every block, parallel across blocks with `threads` workers (blocks
// are independent; no threading inside a block). Each block gets a seed
// derived from `seed` and its BlockId key, so results do not depend on the
// thread count. `warm` (optional) holds one eigenvector matrix per block.
std::vector<BlockResult> solve_model(const CompiledModel& model, const std::vector<double>& theta,
                                     const SolveOptions& opts, int threads, std::uint64_t seed,
                                     const std::vector<Eigen::MatrixXd>* warm = nullptr);

}  // namespace riss_drucker
