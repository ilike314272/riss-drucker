#include "riss_drucker/model.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <thread>

namespace riss_drucker {

CsrMatrix assemble(const CompiledBlock& block, const std::vector<double>& theta) {
  // The union pattern is recomputed per call (O(nnz log nnz)), negligible next
  // to the eigensolve. Cache it per block if fitting loops show otherwise.
  return OperatorSum(block.ops).assemble(theta);
}

Eigen::MatrixXd hf_jacobian(const CompiledBlock& block, const Eigen::MatrixXd& vectors) {
  const auto n = block.dim();
  if (vectors.rows() != n)
    throw std::invalid_argument("hf_jacobian: eigenvectors have " + std::to_string(vectors.rows()) +
                                " rows, block dimension is " + std::to_string(n));
  Eigen::MatrixXd jac(vectors.cols(), static_cast<Eigen::Index>(block.ops.size()));
  Eigen::VectorXd ox(n);
  for (Eigen::Index i = 0; i < vectors.cols(); ++i) {
    const Eigen::VectorXd x = vectors.col(i).normalized();
    for (std::size_t k = 0; k < block.ops.size(); ++k) {
      block.ops[k].matvec(SimdLevel::Scalar, x.data(), ox.data());  // reference path
      jac(i, static_cast<Eigen::Index>(k)) = x.dot(ox);
    }
  }
  return jac;
}

std::vector<BlockResult> solve_model(const CompiledModel& model, const std::vector<double>& theta,
                                     const SolveOptions& opts, int threads, std::uint64_t seed,
                                     const std::vector<Eigen::MatrixXd>* warm) {
  if (theta.size() != model.param_names.size())
    throw std::invalid_argument("solve_model: expected " + std::to_string(model.param_names.size()) +
                                " parameters, got " + std::to_string(theta.size()));
  if (warm && warm->size() != model.blocks.size())
    throw std::invalid_argument("solve_model: warm start must have one entry per block");
  const std::size_t nb = model.blocks.size();
  std::vector<BlockResult> out(nb);
  std::vector<std::exception_ptr> errors(nb);
  std::atomic<std::size_t> next{0};

  auto worker = [&]() {
    for (std::size_t i = next++; i < nb; i = next++) {
      try {
        const auto& b = model.blocks[i];
        SolveOptions o = opts;
        o.lanczos.seed = seed ^ stable_hash64(b.id.key());  // independent of thread count
        const CsrMatrix h = assemble(b, theta);
        const Eigen::MatrixXd* w = (warm && (*warm)[i].size()) ? &(*warm)[i] : nullptr;
        out[i] = solve_block(h, o, w);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    }
  };
  const int nt = std::max(1, std::min<int>(threads, static_cast<int>(nb)));
  if (nt == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
  }
  for (const auto& e : errors)
    if (e) std::rethrow_exception(e);
  return out;
}

}  // namespace riss_drucker
