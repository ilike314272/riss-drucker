// M1 physics checks: free rotor, harmonic limit, rigid rotor, block structure.
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "riss_drucker/models/torsion_rotor.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;
using models::TorsionRotorSpec;

namespace {
std::vector<double> all_levels(const CompiledModel& m, const std::vector<double>& theta) {
  SolveOptions o;  // full spectrum per block
  o.method = SolveMethod::Dense;
  std::vector<double> out;
  for (const auto& r : solve_model(m, theta, o, 2, 7))
    for (Eigen::Index i = 0; i < r.eigenvalues.size(); ++i) out.push_back(r.eigenvalues[i]);
  std::sort(out.begin(), out.end());
  return out;
}
}  // namespace

TEST_CASE("free rotor: levels F m^2, all blocks certified, basis complete") {
  TorsionRotorSpec s;
  s.n_fold = 3;
  s.m_max = 12;
  const auto m = models::compile_torsion_rotor(s);
  CHECK(m.param_names == (std::vector<std::string>{"F", "V3"}));
  std::int64_t total = 0;
  for (const auto& b : m.blocks) total += b.dim();
  CHECK(total == 2 * 12 + 1);
  const auto lv = all_levels(m, {2.0, 0.0});
  std::vector<double> ref;
  for (int k = -12; k <= 12; ++k) ref.push_back(2.0 * k * k);
  std::sort(ref.begin(), ref.end());
  CHECK(lv.size() == ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(lv[i], ref[i], 1e-12);
}

TEST_CASE("block structure: sigma blocks, parity split only when self-conjugate") {
  TorsionRotorSpec s;
  s.n_fold = 3;
  s.m_max = 9;
  const auto m = models::compile_torsion_rotor(s);
  std::map<std::string, std::int64_t> dims;
  for (const auto& b : m.blocks) dims[b.id.irrep] = b.dim();
  // sigma=0: m in {0, +-3, +-6, +-9} -> cos 4, sin 3; sigma=1: {-8,-5,-2,1,4,7}; sigma=2 likewise.
  CHECK(dims.size() == 4);
  CHECK(dims["sigma=0,+"] == 4 && dims["sigma=0,-"] == 3);
  CHECK(dims["sigma=1"] == 6 && dims["sigma=2"] == 6);
  CHECK(m.compile_certificates.front().passed);
}

// Mathieu large-q expansion in torsion units, s = 2v + 1, q = V / (n^2 F):
// (Abramowitz & Stegun 20.2.30, a = 4(E - V/2)/(n^2 F)):
// E_v = n sqrt(FV)(v + 1/2) - n^2 F/4 [(s^2+1)/8 + (s^3+3s)/(2^7 sqrt q) + (5s^4+34s^2+9)/(2^12 q)] + O(q^-3/2)
static double harmonic_level(int v, double F, double V, double n) {
  const double s = 2.0 * v + 1.0, q = V / (n * n * F);
  return n * std::sqrt(F * V) * (v + 0.5) -
         0.25 * n * n * F *
             ((s * s + 1.0) / 8.0 + (s * s * s + 3.0 * s) / (128.0 * std::sqrt(q)) +
              (5.0 * s * s * s * s + 34.0 * s * s + 9.0) / (4096.0 * q));
}

TEST_CASE("harmonic limit: low levels follow the large-barrier expansion") {
  TorsionRotorSpec s;
  s.n_fold = 3;
  s.m_max = 120;
  const auto m = models::compile_torsion_rotor(s);
  const double F = 1.0, V = 2.0e4, n = 3.0;
  SolveOptions o;
  o.nlowest = 4;
  o.method = SolveMethod::Lanczos;
  const auto res = solve_model(m, {F, V}, o, 1, 3);
  for (std::size_t i = 0; i < res.size(); ++i) {
    const auto& r = res[i];
    const std::string& irrep = m.blocks[i].id.irrep;
    CHECK(r.certified);
    // E species hold every v; A1 (cos) even v; A2 (sin) odd v. Tunnelling
    // splittings between species are far below the tolerance here.
    for (int j = 0; j < 3; ++j) {
      const int v = irrep == "sigma=0,+" ? 2 * j : irrep == "sigma=0,-" ? 2 * j + 1 : j;
      CHECK_CLOSE(r.eigenvalues[j], harmonic_level(v, F, V, n), 1e-3 * (v + 1));
    }
  }
}

TEST_CASE("rigid rotor: V = 0, no coupling -> B J(J+1) + (A - B) K^2 + F m^2") {
  TorsionRotorSpec s;
  s.n_fold = 2;
  s.m_max = 4;
  s.j_max = 3;
  const auto m = models::compile_torsion_rotor(s);
  CHECK(m.param_names == (std::vector<std::string>{"F", "V2", "A", "B"}));
  const double F = 5.0, A = 9.0, B = 1.5;
  SolveOptions o;
  o.method = SolveMethod::Dense;
  const auto res = solve_model(m, {F, 0.0, A, B}, o, 3, 1);
  for (std::size_t i = 0; i < m.blocks.size(); ++i) {
    const auto& b = m.blocks[i];
    const int J = b.id.two_j / 2;
    const int K = std::stoi(b.id.sub.substr(2));
    CHECK(res[i].certified);
    const double rot = B * J * (J + 1) + (A - B) * K * K;
    std::vector<double> expect;
    for (const auto& lab : b.basis_labels) {  // "m=3" or "m=+-3,cos"
      const auto pos = lab.find_first_of("0123456789");
      const int mm = std::stoi(lab.substr(pos));
      expect.push_back(rot + F * mm * mm);
    }
    std::sort(expect.begin(), expect.end());
    CHECK(static_cast<Eigen::Index>(expect.size()) == res[i].eigenvalues.size());
    for (std::size_t j = 0; j < expect.size(); ++j) CHECK_CLOSE(res[i].eigenvalues[j], expect[j], 1e-11);
  }
}

TEST_CASE("K p coupling: F (p - rho K)^2 gauge, (K, sigma) and (-K, -sigma) are degenerate") {
  TorsionRotorSpec s;
  s.n_fold = 3;
  s.m_max = 30;
  s.j_max = 2;
  s.kp_coupling = true;
  const auto m = models::compile_torsion_rotor(s);
  const double F = 5.0, rho = 0.1, A = 3.0, B = 1.0;
  // D_Kp = -2 F rho; A absorbs F rho^2.
  const std::vector<double> theta{F, 150.0, A + F * rho * rho, B, -2.0 * F * rho};
  SolveOptions o;
  o.nlowest = 3;
  const auto res = solve_model(m, theta, o, 2, 1);
  std::map<std::string, Eigen::VectorXd> by;
  for (std::size_t i = 0; i < m.blocks.size(); ++i) {
    CHECK(res[i].certified);
    by[m.blocks[i].id.key()] = res[i].eigenvalues;
  }
  BlockId a, b;
  a.two_j = b.two_j = 4;
  a.irrep = "sigma=1"; a.sub = "K=2";
  b.irrep = "sigma=2"; b.sub = "K=-2";
  CHECK_CLOSE((by[a.key()] - by[b.key()]).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

TEST_CASE("invalid specs are rejected") {
  TorsionRotorSpec s;
  s.n_fold = 0;
  CHECK_THROWS(models::compile_torsion_rotor(s));
  s.n_fold = 3;
  s.kp_coupling = true;  // needs j_max > 0
  CHECK_THROWS(models::compile_torsion_rotor(s));
  s.kp_coupling = false;
  s.v_orders = {0};
  CHECK_THROWS(models::compile_torsion_rotor(s));
}

TEST_CASE("results do not depend on the thread count") {
  TorsionRotorSpec s;
  s.n_fold = 3;
  s.m_max = 40;
  s.j_max = 2;
  const auto m = models::compile_torsion_rotor(s);
  SolveOptions o;
  o.nlowest = 5;
  o.method = SolveMethod::Lanczos;
  const std::vector<double> th{5.0, 300.0, 4.0, 1.0};
  const auto r1 = solve_model(m, th, o, 1, 42), r4 = solve_model(m, th, o, 4, 42);
  for (std::size_t i = 0; i < r1.size(); ++i)
    CHECK((r1[i].eigenvalues - r4[i].eigenvalues).cwiseAbs().maxCoeff() == 0.0);
}

RD_TEST_MAIN
