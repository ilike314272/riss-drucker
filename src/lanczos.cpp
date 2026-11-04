#include "riss_drucker/lanczos.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "riss_drucker/kernels.hpp"

namespace riss_drucker {
namespace {

// Two passes of modified Gram-Schmidt against the first `ncols` columns of M,
// using the dispatched dot/axpy kernels (README 4: reorthogonalization).
void orthogonalize(Eigen::VectorXd& w, const Eigen::MatrixXd& M, Eigen::Index ncols) {
  const auto n = static_cast<std::size_t>(w.size());
  for (int pass = 0; pass < 2; ++pass)
    for (Eigen::Index j = 0; j < ncols; ++j) {
      const double c = kernels::dot(n, M.col(j).data(), w.data());
      kernels::axpy(n, -c, M.col(j).data(), w.data());
    }
}

double norm2(const Eigen::VectorXd& w) {
  return std::sqrt(kernels::dot(static_cast<std::size_t>(w.size()), w.data(), w.data()));
}

Eigen::VectorXd random_vector(Eigen::Index n, std::mt19937_64& rng) {
  std::normal_distribution<double> d;
  Eigen::VectorXd v(n);
  for (Eigen::Index i = 0; i < n; ++i) v[i] = d(rng);
  return v;
}

}  // namespace

LinearOperator as_operator(const CsrMatrix& h) {
  if (h.rows != h.cols) throw std::invalid_argument("lanczos: matrix must be square");
  LinearOperator op;
  op.n = h.rows;
  op.norm_hint = std::max(h.inf_norm(), 1e-300);
  op.apply = [&h](const double* x, double* y) { h.matvec(x, y); };
  return op;
}

bool lanczos_extend_lowest(const LinearOperator& op, std::int64_t target, const LanczosOptions& opts,
                           LanczosState& state, const Eigen::MatrixXd* warm) {
  const Eigen::Index n = op.n;
  target = std::min<std::int64_t>(target, n);
  if (state.vectors.rows() != n) {
    if (state.locked() != 0) throw std::invalid_argument("lanczos: state dimension mismatch");
    state.vectors.resize(n, 0);
  }
  double hnorm = op.norm_hint > 0 ? op.norm_hint : 0.0;  // grows with |alpha|, beta if unknown
  Eigen::Index kmax = std::max(opts.max_basis, 2);

  while (state.locked() < target && state.runs < opts.max_runs) {
    const Eigen::Index nlocked = state.locked();
    const Eigen::Index free = n - nlocked;
    if (free <= 0) break;
    const Eigen::Index m = std::min(kmax, free);
    std::mt19937_64 rng(opts.seed + 0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(state.runs + 1));

    // Start vector: warm start (sum of previous eigenvectors) on the first run.
    Eigen::VectorXd v;
    if (state.runs == 0 && warm && warm->rows() == n && warm->cols() > 0)
      v = warm->rowwise().sum();
    else
      v = random_vector(n, rng);
    orthogonalize(v, state.vectors, nlocked);
    double vn = norm2(v);
    if (vn < 1e-8 * std::sqrt(static_cast<double>(n))) {
      v = random_vector(n, rng);
      orthogonalize(v, state.vectors, nlocked);
      vn = norm2(v);
    }
    v /= vn;

    Eigen::MatrixXd Q(n, m);
    Eigen::VectorXd alpha(m), beta(m);
    Eigen::VectorXd w(n);
    Eigen::Index steps = 0;
    bool invariant = false;
    for (Eigen::Index j = 0; j < m; ++j) {
      Q.col(j) = v;
      op.apply(v.data(), w.data());
      ++state.matvecs;
      alpha[j] = kernels::dot(static_cast<std::size_t>(n), v.data(), w.data());
      if (op.norm_hint <= 0) hnorm = std::max(hnorm, std::abs(alpha[j]));
      kernels::axpy(static_cast<std::size_t>(n), -alpha[j], v.data(), w.data());
      if (j > 0) kernels::axpy(static_cast<std::size_t>(n), -beta[j - 1], Q.col(j - 1).data(), w.data());
      orthogonalize(w, state.vectors, nlocked);  // full reorthogonalization
      orthogonalize(w, Q, j + 1);
      steps = j + 1;
      const double b = norm2(w);
      beta[j] = b;
      if (b <= 1e-13 * std::max(hnorm, 1e-300)) { invariant = true; break; }
      v = w / b;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> tes;
    Eigen::VectorXd sub = steps > 1 ? Eigen::VectorXd(beta.head(steps - 1)) : Eigen::VectorXd(0);
    tes.computeFromTridiagonal(alpha.head(steps), sub, Eigen::ComputeEigenvectors);
    const double blast = invariant ? 0.0 : beta[steps - 1];
    if (op.norm_hint <= 0) hnorm = std::max(hnorm, tes.eigenvalues().cwiseAbs().maxCoeff());

    // Lock converged Ritz pairs from the bottom of the spectrum, stopping at
    // the first unconverged one, so locked pairs are the lowest found.
    Eigen::Index newly = 0;
    for (Eigen::Index i = 0; i < steps; ++i) {
      const double resid = blast * std::abs(tes.eigenvectors()(steps - 1, i));
      if (resid > opts.tol * hnorm) break;
      Eigen::VectorXd y = Q.leftCols(steps) * tes.eigenvectors().col(i);
      orthogonalize(y, state.vectors, state.locked());
      const double yn = norm2(y);
      if (yn < 0.5) continue;  // lost to deflation: already represented
      y /= yn;
      state.vectors.conservativeResize(n, state.locked() + 1);
      state.vectors.col(state.locked()) = y;
      state.values.conservativeResize(state.values.size() + 1);
      state.values[state.values.size() - 1] = tes.eigenvalues()[i];
      ++newly;
    }
    ++state.runs;
    if (newly == 0) kmax = std::min<Eigen::Index>(2 * kmax, n);
  }

  // Locked pairs from different runs can interleave: sort ascending.
  std::vector<Eigen::Index> perm(static_cast<std::size_t>(state.locked()));
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_sort(perm.begin(), perm.end(),
                   [&](Eigen::Index a, Eigen::Index b) { return state.values[a] < state.values[b]; });
  Eigen::VectorXd vals(state.locked());
  Eigen::MatrixXd vecs(n, state.locked());
  for (std::size_t i = 0; i < perm.size(); ++i) {
    vals[static_cast<Eigen::Index>(i)] = state.values[perm[i]];
    vecs.col(static_cast<Eigen::Index>(i)) = state.vectors.col(perm[i]);
  }
  state.values = std::move(vals);
  state.vectors = std::move(vecs);
  return state.locked() >= target;
}

}  // namespace riss_drucker
