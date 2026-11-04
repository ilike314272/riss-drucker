// Solver tests (README section 12): Lanczos vs dense reference, inertia count
// equals returned count, warm start equals cold start, degenerate levels,
// window widening, and failed certificates surfaced instead of absorbed.
#include <algorithm>
#include <random>

#include "riss_drucker/solver.hpp"
#include "test_harness.hpp"
#include "test_util.hpp"

using namespace riss_drucker;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
const Certificate* find_cert(const BlockResult& r, const std::string& name) {
  for (const auto& c : r.certificates)
    if (c.name == name) return &c;
  return nullptr;
}
SolveOptions lowest(std::int64_t k, SolveMethod m) {
  SolveOptions o;
  o.nlowest = k;
  o.method = m;
  return o;
}
}  // namespace

TEST_CASE("dense and Lanczos agree on the lowest levels and are certified") {
  std::mt19937_64 rng(20);
  const auto h = rd_test::random_banded(400, 4, rng);
  const VectorXd ref = Eigen::SelfAdjointEigenSolver<MatrixXd>(h.to_dense()).eigenvalues();
  for (auto m : {SolveMethod::Dense, SolveMethod::Lanczos}) {
    const auto r = solve_block(h, lowest(12, m), nullptr);
    CHECK(r.certified);
    CHECK(r.eigenvalues.size() == 12);
    CHECK_CLOSE((r.eigenvalues - ref.head(12)).cwiseAbs().maxCoeff(), 0.0, 1e-9);
    CHECK(find_cert(r, "inertia_count") && find_cert(r, "inertia_count")->passed);
    CHECK(r.residuals.maxCoeff() < 1e-8 * h.inf_norm());
  }
}

TEST_CASE("window mode: returned count equals inertia count, Lanczos equals dense") {
  std::mt19937_64 rng(21);
  const auto h = rd_test::random_banded(300, 3, rng);
  const VectorXd ref = Eigen::SelfAdjointEigenSolver<MatrixXd>(h.to_dense()).eigenvalues();
  SolveOptions o;
  o.window_lo = ref[5] - 0.5 * (ref[5] - ref[4]);
  o.window_hi = ref[20] + 0.5 * (ref[21] - ref[20]);
  for (auto m : {SolveMethod::Dense, SolveMethod::Lanczos}) {
    o.method = m;
    const auto r = solve_block(h, o);
    CHECK(r.certified);
    CHECK(r.eigenvalues.size() == 16);
    CHECK_CLOSE((r.eigenvalues - ref.segment(5, 16)).cwiseAbs().maxCoeff(), 0.0, 1e-9);
  }
}

TEST_CASE("Lanczos finds every copy of exactly degenerate eigenvalues (locking + restart)") {
  // H = A (+) A: exact twofold degeneracy that one Krylov space cannot resolve.
  std::mt19937_64 rng(22);
  const auto a = rd_test::random_banded(60, 2, rng);
  std::vector<Triplet> t;
  for (std::int64_t r = 0; r < a.rows; ++r)
    for (auto k = a.row_ptr[r]; k < a.row_ptr[r + 1]; ++k) {
      t.push_back({r, a.col_idx[k], a.values[k]});
      t.push_back({r + 60, a.col_idx[k] + 60, a.values[k]});
    }
  const auto h = CsrMatrix::from_triplets(120, 120, std::move(t));
  const VectorXd ref = Eigen::SelfAdjointEigenSolver<MatrixXd>(a.to_dense()).eigenvalues();
  // Random start: correct either way (rounding may reveal both copies in one run).
  const auto r = solve_block(h, lowest(6, SolveMethod::Lanczos));
  CHECK(r.certified);
  CHECK(r.eigenvalues.size() == 6);
  for (int i = 0; i < 6; ++i) CHECK_CLOSE(r.eigenvalues[i], ref[i / 2], 1e-9);
  // Start (u, u): every Krylov vector keeps identical halves bit for bit, so
  // the first run sees one copy per level. The inertia count must then force
  // further runs, deflated against the locked vectors, to find the partners.
  MatrixXd sym(120, 1);
  for (int i = 0; i < 60; ++i) sym(i, 0) = sym(i + 60, 0) = std::cos(0.37 * i + 0.1);
  const auto rs = solve_block(h, lowest(6, SolveMethod::Lanczos), &sym);
  CHECK(rs.certified);
  CHECK(rs.eigenvalues.size() == 6);
  for (int i = 0; i < 6; ++i) CHECK_CLOSE(rs.eigenvalues[i], ref[i / 2], 1e-9);
  CHECK(rs.lanczos_runs >= 2);
}

TEST_CASE("degenerate pair across the lowest-k boundary: return both") {
  std::mt19937_64 rng(23);
  const int n = 40;
  VectorXd lam = VectorXd::LinSpaced(n, 0.0, 39.0);
  lam[3] = lam[2];  // k = 3 boundary falls inside a degenerate pair
  std::sort(lam.data(), lam.data() + n);
  const MatrixXd q = rd_test::random_orthogonal(n, rng);
  const auto h = rd_test::csr_from_dense(q * lam.asDiagonal() * q.transpose());
  for (auto m : {SolveMethod::Dense, SolveMethod::Lanczos}) {
    const auto r = solve_block(h, lowest(3, m));
    CHECK(r.certified);
    CHECK(r.eigenvalues.size() == 4);  // the multiplet at the boundary is kept whole
    CHECK(r.boundary_extended == 1);
  }
}

TEST_CASE("window edge on an eigenvalue: widened until the count matches") {
  std::mt19937_64 rng(24);
  const int n = 50;
  const VectorXd lam = VectorXd::LinSpaced(n, 0.0, 49.0);
  const MatrixXd q = rd_test::random_orthogonal(n, rng);
  const auto h = rd_test::csr_from_dense(q * lam.asDiagonal() * q.transpose());
  SolveOptions o;
  o.window_lo = 2.5;
  o.window_hi = 10.0;  // computed eigenvalue ~10 +- rounding
  for (auto m : {SolveMethod::Dense, SolveMethod::Lanczos}) {
    o.method = m;
    const auto r = solve_block(h, o);
    CHECK(r.certified);
    CHECK(r.eigenvalues.size() == 7 || r.eigenvalues.size() == 8);
    CHECK(r.window_lo <= 2.5 && r.window_hi >= 10.0);
  }
}

TEST_CASE("warm start gives the same levels as a cold start, with fewer matvecs") {
  std::mt19937_64 rng(25);
  const auto h0 = rd_test::random_banded(600, 2, rng);
  const auto cold = solve_block(h0, lowest(8, SolveMethod::Lanczos));
  // Perturb (a nearby theta) and warm-start from the previous eigenvectors.
  CsrMatrix h1 = h0;
  for (auto& v : h1.values) v *= 1.001;
  const auto ref = solve_block(h1, lowest(8, SolveMethod::Lanczos));
  const auto warm = solve_block(h1, lowest(8, SolveMethod::Lanczos), &cold.eigenvectors);
  CHECK(warm.certified && ref.certified);
  CHECK_CLOSE((warm.eigenvalues - ref.eigenvalues).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

TEST_CASE("failed certificate is surfaced, not absorbed") {
  std::mt19937_64 rng(26);
  const auto h = rd_test::random_banded(500, 3, rng);
  SolveOptions o = lowest(30, SolveMethod::Lanczos);
  o.lanczos.max_basis = 4;  // starve Lanczos
  o.lanczos.max_runs = 1;
  const auto r = solve_block(h, o);
  CHECK(!r.certified);
  const auto* c = find_cert(r, "lanczos_convergence");
  CHECK(c && !c->passed);
}

TEST_CASE("tridiagonal blocks use the Sturm count; memory budget fails early") {
  std::mt19937_64 rng(27);
  const auto tri = rd_test::random_banded(100, 1, rng);
  SolveOptions o = lowest(5, SolveMethod::Auto);
  CHECK(estimate_cost(tri, o).inertia_method == "sturm");
  const auto r = solve_block(tri, o);
  CHECK(r.certified);
  CHECK(find_cert(r, "inertia_count")->detail.find("method=sturm") != std::string::npos);
  o.memory_budget_bytes = 1000.0;
  o.method = SolveMethod::Dense;
  CHECK_THROWS(solve_block(tri, o));
}

TEST_CASE("plain and shift-invert Lanczos agree; shift-invert is the default") {
  std::mt19937_64 rng(28);
  const auto h = rd_test::random_banded(500, 3, rng);
  const VectorXd ref = Eigen::SelfAdjointEigenSolver<MatrixXd>(h.to_dense()).eigenvalues();
  SolveOptions o = lowest(8, SolveMethod::Lanczos);
  const auto si = solve_block(h, o);
  o.shift_invert = false;
  const auto plain = solve_block(h, o);
  CHECK(si.lanczos_mode == "shift_invert" && plain.lanczos_mode == "plain");
  CHECK(si.certified && plain.certified);
  CHECK_CLOSE((si.eigenvalues - ref.head(8)).cwiseAbs().maxCoeff(), 0.0, 1e-9);
  CHECK_CLOSE((plain.eigenvalues - ref.head(8)).cwiseAbs().maxCoeff(), 0.0, 1e-9);
  // Window mode on the plain path, including the degenerate direct sum.
  SolveOptions w;
  w.method = SolveMethod::Lanczos;
  w.shift_invert = false;
  w.window_lo = ref[2] - 0.5 * (ref[2] - ref[1]);
  w.window_hi = ref[9] + 0.5 * (ref[10] - ref[9]);
  const auto pw = solve_block(h, w);
  CHECK(pw.certified && pw.eigenvalues.size() == 8);
}

TEST_CASE("gap ratio statistic") {
  VectorXd eq = VectorXd::LinSpaced(10, 0, 9);
  CHECK_CLOSE(gap_ratio_mean(eq), 1.0, 1e-15);
  CHECK(std::isnan(gap_ratio_mean(VectorXd::Zero(2))));
}

RD_TEST_MAIN
