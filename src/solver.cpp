#include "riss_drucker/solver.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace riss_drucker {

namespace {
constexpr double kEps = 2.220446049250313e-16;
constexpr double kInf = std::numeric_limits<double>::infinity();

std::string fmt(double x) {
  std::ostringstream s;
  s.precision(12);
  s << x;
  return s.str();
}
}  // namespace

const char* to_string(SolveMethod m) {
  switch (m) {
    case SolveMethod::Auto: return "auto";
    case SolveMethod::Dense: return "dense";
    case SolveMethod::Lanczos: return "lanczos";
  }
  return "unknown";
}

SolveMethod parse_solve_method(const std::string& s) {
  std::string n = s;
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
  if (n == "auto") return SolveMethod::Auto;
  if (n == "dense") return SolveMethod::Dense;
  if (n == "lanczos") return SolveMethod::Lanczos;
  throw std::invalid_argument("unknown solve method '" + s + "' (expected auto|dense|lanczos)");
}

BlockCost estimate_cost(const CsrMatrix& h, const SolveOptions& opts) {
  BlockCost c;
  c.dim = h.rows;
  c.nnz = h.nnz();
  c.bandwidth = h.bandwidth();
  c.dense_bytes = linalg::dense_eigh_bytes(static_cast<long>(c.dim));
  const double nev = opts.nlowest >= 0 ? static_cast<double>(opts.nlowest + 1) : 64.0;
  c.lanczos_bytes = 8.0 * static_cast<double>(c.dim) *
                    (std::min<double>(opts.lanczos.max_basis, static_cast<double>(c.dim)) + nev);
  const bool budget = opts.memory_budget_bytes > 0;
  const bool dense_fits = !budget || c.dense_bytes <= opts.memory_budget_bytes;
  switch (opts.method) {
    case SolveMethod::Dense: c.recommended = "dense"; break;
    case SolveMethod::Lanczos: c.recommended = "lanczos"; break;
    case SolveMethod::Auto:
      c.recommended = (c.dim <= opts.dense_max_dim && dense_fits) ? "dense" : "lanczos";
  }
  const double bk_bytes = 8.0 * static_cast<double>(c.dim) * static_cast<double>(c.dim);
  if (c.bandwidth <= 1)
    c.inertia_method = "sturm";
  else if (c.dim <= opts.inertia_dense_max_dim && (!budget || bk_bytes <= opts.memory_budget_bytes))
    c.inertia_method = "bunch_kaufman";
  else
    c.inertia_method = "unavailable";
  return c;
}

double gap_ratio_mean(const Eigen::VectorXd& levels) {
  if (levels.size() < 3) return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.0;
  int cnt = 0;
  for (Eigen::Index i = 0; i + 2 < levels.size(); ++i) {
    const double s1 = levels[i + 1] - levels[i], s2 = levels[i + 2] - levels[i + 1];
    const double mx = std::max(s1, s2);
    if (mx <= 0.0) continue;
    sum += std::min(s1, s2) / mx;
    ++cnt;
  }
  return cnt ? sum / cnt : std::numeric_limits<double>::quiet_NaN();
}

namespace {

// Counts eigenvalues below sigma; caches the dense matrix for repeated counts.
class Counter {
 public:
  Counter(const CsrMatrix& h, const SolveOptions& opts, const std::string& method)
      : h_(h), opts_(opts), method_(method) {}
  Inertia operator()(double sigma) {
    if (method_ == "sturm") {
      if (!tri_) tri_ = std::make_pair(h_.diagonal(), h_.subdiagonal());
      return sturm_count(tri_->first, tri_->second, sigma);
    }
    if (method_ == "bunch_kaufman") {
      if (!dense_) dense_ = h_.to_dense();
      return inertia_dense(*dense_, sigma, opts_.backend);
    }
    throw BudgetError("inertia count unavailable: block of dimension " + std::to_string(h_.rows) +
                      " is too large for a dense Bunch-Kaufman count within inertia_dense_max_dim / "
                      "the memory budget, and is not tridiagonal");
  }
  bool available() const { return method_ != "unavailable"; }

 private:
  const CsrMatrix& h_;
  const SolveOptions& opts_;
  std::string method_;
  std::optional<Eigen::MatrixXd> dense_;
  std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> tri_;
};

}  // namespace

Inertia count_below(const CsrMatrix& h, double sigma, const SolveOptions& opts) {
  Counter c(h, opts, estimate_cost(h, opts).inertia_method);
  return c(sigma);
}

BlockResult solve_block(const CsrMatrix& h, const SolveOptions& opts, const Eigen::MatrixXd* warm) {
  if (h.rows != h.cols) throw std::invalid_argument("solve_block: matrix must be square");
  if (!(opts.window_lo < opts.window_hi) && opts.nlowest < 0)
    throw std::invalid_argument("solve_block: empty window (need window_lo < window_hi)");

  BlockResult r;
  r.cost = estimate_cost(h, opts);
  r.method = r.cost.recommended;
  r.nev_requested = opts.nlowest;
  const bool budget = opts.memory_budget_bytes > 0;
  if (r.method == "dense" && budget && r.cost.dense_bytes > opts.memory_budget_bytes)
    throw BudgetError("dense solve of dimension " + std::to_string(r.cost.dim) + " needs ~" +
                      fmt(r.cost.dense_bytes / 1e9) + " GB, over the memory budget of " +
                      fmt(opts.memory_budget_bytes / 1e9) + " GB; use method='lanczos'");
  if (r.method == "lanczos" && budget && r.cost.lanczos_bytes > opts.memory_budget_bytes)
    throw BudgetError("Lanczos solve of dimension " + std::to_string(r.cost.dim) + " needs ~" +
                      fmt(r.cost.lanczos_bytes / 1e9) + " GB, over the memory budget");

  const Eigen::Index n = h.rows;
  const double hnorm = h.inf_norm();

  // Hermiticity of the assembled block (cheap; operators were checked at compile).
  {
    Certificate c{"hermiticity", false, h.symmetry_defect(), 1e-12 * std::max(hnorm, 1.0), ""};
    c.passed = c.value <= c.bound;
    r.certificates.push_back(c);
  }
  if (n == 0) {
    r.certified = r.certificates.front().passed;
    return r;
  }

  Counter count(h, opts, r.cost.inertia_method);
  if (opts.nlowest < 0 && std::isfinite(opts.window_hi) && r.method == "lanczos" && !count.available())
    count(0.0);  // throws BudgetError with the reason: a window Lanczos solve needs counts

  // Candidate eigenpairs (ascending).
  Eigen::VectorXd all_vals;
  Eigen::MatrixXd all_vecs;
  LanczosState st;
  LanczosOptions lopt = opts.lanczos;
  bool lanczos_ok = true;
  Eigen::VectorXd lz_vals;  // Lanczos candidates mapped to eigenvalues of H
  Eigen::MatrixXd lz_vecs;

  // Shift-invert (lowest levels of stiff spectra): Lanczos on
  // B = -(H - sigma I)^{-1} with sigma below a Gershgorin bound, so H - sigma I
  // is positive definite and a sparse Cholesky factor is stable without
  // pivoting. B preserves the ordering of eigenvalues; values are mapped back
  // by Rayleigh quotients with H, and every certificate is evaluated on H.
  LinearOperator lop;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
  double si_sigma = std::numeric_limits<double>::quiet_NaN();
  if (r.method == "lanczos") {
    lop = as_operator(h);
    r.lanczos_mode = "plain";
    if (opts.shift_invert) {
      double gersh = kInf;
      for (Eigen::Index i = 0; i < n; ++i) {
        double diag = 0.0, off = 0.0;
        for (auto k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k)
          if (h.col_idx[k] == i) diag += h.values[k];
          else off += std::abs(h.values[k]);
        gersh = std::min(gersh, diag - off);
      }
      std::vector<Eigen::Triplet<double>> t;
      t.reserve(static_cast<std::size_t>(h.nnz() + n));
      for (Eigen::Index i = 0; i < n; ++i)
        for (auto k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) t.emplace_back(i, h.col_idx[k], h.values[k]);
      for (double margin = 1e-6 * std::max(hnorm, 1.0); margin <= 1e-2 * std::max(hnorm, 1.0); margin *= 10) {
        Eigen::SparseMatrix<double> a(n, n);
        a.setFromTriplets(t.begin(), t.end());
        Eigen::SparseMatrix<double> id(n, n);
        id.setIdentity();
        a -= (gersh - margin) * id;
        llt.compute(a);
        if (llt.info() == Eigen::Success) {
          si_sigma = gersh - margin;
          break;
        }
      }
      if (std::isfinite(si_sigma)) {
        r.lanczos_mode = "shift_invert";
        lop.norm_hint = 0.0;
        lop.apply = [&llt, n](const double* x, double* y) {
          Eigen::Map<Eigen::VectorXd>(y, n) = -llt.solve(Eigen::Map<const Eigen::VectorXd>(x, n));
        };
      }
    }
  }
  auto lanczos_need = [&](std::int64_t target) {
    lanczos_ok = lanczos_extend_lowest(lop, target, lopt, st, warm) && lanczos_ok;
    lz_vecs = st.vectors;
    if (r.lanczos_mode != "shift_invert") {
      lz_vals = st.values;
      return;
    }
    lz_vals.resize(st.locked());
    Eigen::VectorXd hx(n);
    for (Eigen::Index i = 0; i < st.locked(); ++i) {
      h.matvec(lz_vecs.col(i).data(), hx.data());
      lz_vals[i] = lz_vecs.col(i).dot(hx);  // Rayleigh quotient (unit vectors)
    }
    std::vector<Eigen::Index> perm(static_cast<std::size_t>(lz_vals.size()));
    for (std::size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<Eigen::Index>(i);
    std::stable_sort(perm.begin(), perm.end(), [&](auto a, auto b) { return lz_vals[a] < lz_vals[b]; });
    Eigen::VectorXd v2(lz_vals.size());
    Eigen::MatrixXd m2(n, lz_vals.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      v2[static_cast<Eigen::Index>(i)] = lz_vals[perm[i]];
      m2.col(static_cast<Eigen::Index>(i)) = lz_vecs.col(perm[i]);
    }
    lz_vals = std::move(v2);
    lz_vecs = std::move(m2);
  };
  if (r.method == "dense") {
    auto e = linalg::eigh(h.to_dense(), opts.backend, true);
    all_vals = std::move(e.eigenvalues);
    all_vecs = std::move(e.eigenvectors);
  }
  auto candidates = [&]() -> const Eigen::VectorXd& {
    return r.method == "dense" ? all_vals : lz_vals;
  };

  // Lowest-k becomes a window [-inf, hi) with hi in the gap above level k.
  // Never split a (numerically) degenerate cluster at the boundary: extend k
  // until the gap above it exceeds the degeneracy tolerance.
  double lo = opts.window_lo, hi = opts.window_hi;
  const std::int64_t k_req = opts.nlowest >= 0 ? std::min<std::int64_t>(opts.nlowest, n) : -1;
  auto choose_hi = [&]() {
    lo = -kInf;
    r.boundary_extended = 0;
    if (k_req == 0) return -kInf;
    if (k_req >= n) return kInf;
    const double degen = opts.degeneracy_tol * std::max(hnorm, 1.0);
    std::int64_t kk = k_req;
    if (r.method == "lanczos") lanczos_need(kk + 1);
    while (kk < n) {
      const auto& v = candidates();
      if (v.size() < kk + 1) break;  // unconverged; the certificates decide
      if (v[kk] - v[kk - 1] > degen) break;
      ++kk;
      if (r.method == "lanczos" && kk < n) lanczos_need(kk + 1);
    }
    r.boundary_extended = kk - k_req;
    const auto& v = candidates();
    if (kk >= n) return kInf;
    if (v.size() >= kk + 1) return 0.5 * (v[kk - 1] + v[kk]);
    return v.size() ? v[v.size() - 1] + 1.0 : kInf;
  };

  const double widen_base = 1e3 * kEps * std::max(hnorm, 1.0);
  Inertia ihi, ilo;
  std::int64_t expected = 0, found = 0;
  Eigen::Index first = 0;  // index of the first candidate in the window
  bool counted = false;
  auto run_window = [&]() {
    for (int attempt = 0;; ++attempt) {
      std::int64_t c_hi = n, c_lo = 0;
      counted = count.available();
      if (counted) {
        if (std::isfinite(hi)) { ihi = count(hi); c_hi = ihi.n_neg; }
        else if (hi < 0) c_hi = 0;
        if (std::isfinite(lo)) { ilo = count(lo); c_lo = ilo.n_neg; }
      } else if (hi == -kInf) {
        c_hi = 0;
      }
      expected = c_hi - c_lo;

      if (r.method == "lanczos") {
        // Lanczos must hold every eigenvalue below hi. If the count says some
        // are missing (e.g. a second copy of a degenerate level), run more.
        std::int64_t target = counted ? c_hi : st.locked();
        if (!counted && k_req >= 0) target = std::min<std::int64_t>(k_req + 1, n);
        for (int guard = 0; guard < lopt.max_runs + 1; ++guard) {
          lanczos_need(target);
          const auto& v = lz_vals;
          const std::int64_t below = std::count_if(v.data(), v.data() + v.size(), [&](double x) { return x < hi; });
          if (!counted || below >= c_hi || st.runs >= lopt.max_runs) break;
          target = st.locked() + (c_hi - below);
        }
      }

      const auto& v = candidates();
      first = std::lower_bound(v.data(), v.data() + v.size(), lo) - v.data();
      const Eigen::Index last = std::lower_bound(v.data(), v.data() + v.size(), hi) - v.data();
      found = last - first;
      if (!counted || found == expected || attempt >= opts.max_widen) break;
      if (!std::isfinite(lo) && !std::isfinite(hi)) break;  // nothing to widen: a solver shortfall
      // Mismatch at rounding level near an edge: widen outward and recount.
      const double delta = widen_base * std::pow(4.0, attempt);
      if (std::isfinite(lo)) lo -= delta;
      if (std::isfinite(hi)) hi += delta;
      ++r.widen_steps;
    }
  };

  if (k_req >= 0) hi = choose_hi();
  run_window();
  // Lowest-k with Lanczos: the edge may have been placed from a candidate set
  // still missing degenerate partners. After a matching count the candidates
  // below hi are complete, so re-place the edge and recount once.
  if (k_req > 0 && counted && found == expected && found > k_req + r.boundary_extended) {
    hi = choose_hi();
    run_window();
  }
  r.window_lo = lo;
  r.window_hi = hi;
  r.lanczos_runs = st.runs;
  r.matvecs = st.matvecs;

  {
    Certificate c;
    c.name = "inertia_count";
    c.value = static_cast<double>(found);
    c.bound = static_cast<double>(expected);
    if (counted) {
      c.passed = found == expected;
      c.detail = "method=" + (std::isfinite(hi) ? ihi.method : std::isfinite(lo) ? ilo.method : std::string("trivial")) +
                 " window=[" + fmt(lo) + ", " + fmt(hi) + ") returned=" + std::to_string(found) +
                 " counted=" + std::to_string(expected) + " widen_steps=" + std::to_string(r.widen_steps) +
                 " boundary_extended=" + std::to_string(r.boundary_extended);
      if (std::isfinite(hi)) c.detail += " min_abs_pivot(hi)=" + fmt(ihi.min_abs_pivot);
    } else {
      c.passed = false;
      c.detail = "inertia count unavailable for this block (too large for dense Bunch-Kaufman, not "
                 "tridiagonal); result is not certified";
    }
    r.certificates.push_back(c);
  }
  if (r.method == "lanczos") {
    Certificate c{"lanczos_convergence", lanczos_ok, static_cast<double>(st.locked()),
                  static_cast<double>(expected), "mode=" + r.lanczos_mode +
                  (std::isfinite(si_sigma) ? " sigma=" + fmt(si_sigma) : std::string()) +
                  " runs=" + std::to_string(st.runs) + " matvecs=" + std::to_string(st.matvecs)};
    r.certificates.push_back(c);
  }

  // Selected pairs.
  const Eigen::MatrixXd& vecs = r.method == "dense" ? all_vecs : lz_vecs;
  const Eigen::VectorXd& vals = candidates();
  r.eigenvalues = vals.segment(first, found);
  Eigen::MatrixXd sel = vecs.middleCols(first, found);

  // Residual certificate on the scalar reference path (README 4).
  r.residuals.resize(found);
  Eigen::VectorXd hx(n);
  for (Eigen::Index i = 0; i < found; ++i) {
    const Eigen::VectorXd x = sel.col(i);
    h.matvec(SimdLevel::Scalar, x.data(), hx.data());
    r.residuals[i] = (hx - r.eigenvalues[i] * x).norm() / x.norm();
  }
  {
    Certificate c{"residual", false, found ? r.residuals.maxCoeff() : 0.0,
                  opts.residual_tol * std::max(hnorm, 1e-300), "max ||Hx - lambda x|| / ||x||"};
    c.passed = c.value <= c.bound;
    r.certificates.push_back(c);
  }
  {
    const double ortho = found ? (sel.transpose() * sel - Eigen::MatrixXd::Identity(found, found))
                                     .cwiseAbs().maxCoeff() : 0.0;
    Certificate c{"orthogonality", ortho <= opts.orthogonality_tol, ortho, opts.orthogonality_tol,
                  "max |V^T V - I|"};
    r.certificates.push_back(c);
  }
  r.certified = std::all_of(r.certificates.begin(), r.certificates.end(),
                            [](const Certificate& c) { return c.passed; });
  if (opts.keep_vectors) r.eigenvectors = std::move(sel);
  r.gap_ratio_mean = gap_ratio_mean(r.eigenvalues);
  return r;
}

}  // namespace riss_drucker
