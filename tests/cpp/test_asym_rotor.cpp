// Asymmetric rotor (Watson): closed-form rigid levels, representation
// invariance, symmetric-top limit, trace sum rule, block structure, gauge.
#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#include "riss_drucker/models/asym_rotor.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;
using models::AsymRotorSpec;

namespace {
std::map<int, std::vector<double>> levels_by_j(const CompiledModel& m, const std::vector<double>& th) {
  SolveOptions o;
  o.method = SolveMethod::Dense;
  const auto res = solve_model(m, th, o, 2, 0);
  std::map<int, std::vector<double>> out;
  for (std::size_t i = 0; i < res.size(); ++i) {
    CHECK(res[i].certified);
    for (Eigen::Index k = 0; k < res[i].eigenvalues.size(); ++k)
      out[m.blocks[i].id.two_j / 2].push_back(res[i].eigenvalues[k]);
  }
  for (auto& [j, v] : out) std::sort(v.begin(), v.end());
  return out;
}
const double A = 9.4, B = 1.3, C = 1.13;
}  // namespace

TEST_CASE("rigid J = 1, 2 closed forms in every representation") {
  for (const char* ax : {"Ir", "IIr", "IIIr"}) {
    AsymRotorSpec s;
    s.j_max = 2;
    s.order = 2;
    s.axis = ax;
    const auto lv = levels_by_j(models::compile_asym_rotor(s), {A, B, C});
    CHECK(lv.at(0).size() == 1 && std::abs(lv.at(0)[0]) < 1e-14);
    std::vector<double> j1{B + C, A + C, A + B};
    std::sort(j1.begin(), j1.end());
    for (int i = 0; i < 3; ++i) CHECK_CLOSE(lv.at(1)[i], j1[i], 1e-13);
    const double r = 2.0 * std::sqrt((B - C) * (B - C) + (A - C) * (A - B));
    std::vector<double> j2{2 * (A + B + C) - r, A + B + 4 * C, A + 4 * B + C, 4 * A + B + C, 2 * (A + B + C) + r};
    std::sort(j2.begin(), j2.end());
    for (int i = 0; i < 5; ++i) CHECK_CLOSE(lv.at(2)[i], j2[i], 1e-12);
  }
}

TEST_CASE("trace sum rule and representation invariance with distortion") {
  AsymRotorSpec s;
  s.j_max = 12;
  s.order = 2;
  const auto lv = levels_by_j(models::compile_asym_rotor(s), {A, B, C});
  for (int J = 0; J <= 12; ++J) {
    double tr = 0.0;
    for (double e : lv.at(J)) tr += e;
    CHECK_CLOSE(tr, (A + B + C) * J * (J + 1.0) * (2 * J + 1.0) / 3.0, 1e-9);
    CHECK(static_cast<int>(lv.at(J).size()) == 2 * J + 1);
  }
}

TEST_CASE("symmetric-top limit: B = C gives B J(J+1) + (A - B) K^2 (Ir)") {
  AsymRotorSpec s;
  s.j_max = 6;
  s.order = 2;
  const auto lv = levels_by_j(models::compile_asym_rotor(s), {A, B, B});
  for (int J = 0; J <= 6; ++J) {
    std::vector<double> ref;
    for (int K = -J; K <= J; ++K) ref.push_back(B * J * (J + 1.0) + (A - B) * K * K);
    std::sort(ref.begin(), ref.end());
    for (int i = 0; i <= 2 * J; ++i) CHECK_CLOSE(lv.at(J)[i], ref[i], 1e-11);
  }
}

TEST_CASE("Wang block structure and compile certificates") {
  AsymRotorSpec s;
  s.j_min = 0;
  s.j_max = 5;
  s.order = 6;
  const auto m = models::compile_asym_rotor(s);
  std::map<int, int> total;
  for (const auto& b : m.blocks) total[b.id.two_j / 2] += static_cast<int>(b.dim());
  for (int J = 0; J <= 5; ++J) CHECK(total[J] == 2 * J + 1);
  CHECK(m.param_names.size() == 3 + 5 + 7);
  for (const auto& c : m.compile_certificates) CHECK(c.passed);
  CHECK(m.compile_certificates.size() == 6);
}

TEST_CASE("gauge: J2 is A + B + C, so theta + t(1,1,1,-1) leaves every level unchanged") {
  AsymRotorSpec s;
  s.j_max = 8;
  s.order = 4;
  s.extra_operators = {"J2"};
  const auto m = models::compile_asym_rotor(s);
  CHECK(m.param_names.back() == "J2");
  std::vector<double> th{A, B, C, 1e-5, 2e-5, 3e-4, 1e-6, 2e-5, 0.0};
  auto th2 = th;
  for (int k = 0; k < 3; ++k) th2[k] += 0.37;
  th2.back() -= 0.37;
  const auto l1 = levels_by_j(m, th), l2 = levels_by_j(m, th2);
  for (int J = 0; J <= 8; ++J)
    for (std::size_t i = 0; i < l1.at(J).size(); ++i) CHECK_CLOSE(l1.at(J)[i], l2.at(J)[i], 1e-10);
}

TEST_CASE("Hellmann-Feynman Jacobian matches finite differences") {
  AsymRotorSpec s;
  s.j_min = 7;
  s.j_max = 7;
  s.order = 4;
  const auto m = models::compile_asym_rotor(s);
  const std::vector<double> th{A, B, C, 1e-4, 2e-3, 3e-2, 1e-5, 4e-4};
  SolveOptions o;
  o.method = SolveMethod::Dense;
  const auto r0 = solve_model(m, th, o, 1, 0);
  for (std::size_t b = 0; b < m.blocks.size(); ++b) {
    const auto jac = hf_jacobian(m.blocks[b], r0[b].eigenvectors);
    for (std::size_t k = 0; k < th.size(); ++k) {
      const double h = 1e-6 * std::max(1.0, std::abs(th[k]));
      auto tp = th, tm = th;
      tp[k] += h;
      tm[k] -= h;
      const auto rp = solve_model(m, tp, o, 1, 0), rm = solve_model(m, tm, o, 1, 0);
      for (Eigen::Index i = 0; i < jac.rows(); ++i) {
        const double fd = (rp[b].eigenvalues[i] - rm[b].eigenvalues[i]) / (2 * h);
        CHECK_CLOSE(jac(i, static_cast<Eigen::Index>(k)), fd, 1e-5 * (1.0 + std::abs(fd)));
      }
    }
  }
}

TEST_CASE("octic operators: counts, certificates, l4 = J+^8 + J-^8 couples K and K+-8 only") {
  for (const char* red : {"A", "S"}) {
    AsymRotorSpec s;
    s.j_max = 9;
    s.order = 8;
    s.reduction = red;
    const auto m = models::compile_asym_rotor(s);
    CHECK(m.param_names.size() == 3 + 5 + 7 + 9);
    for (const auto& c : m.compile_certificates) CHECK(c.passed);
  }
  AsymRotorSpec s;
  s.j_min = s.j_max = 4;
  s.order = 8;
  s.reduction = "S";
  const auto m = models::compile_asym_rotor(s);
  const auto& names = m.param_names;
  const auto k = static_cast<std::size_t>(std::find(names.begin(), names.end(), "l4") - names.begin());
  // J = 4: J+^8 maps K = -4 to K = +4 only; <4|J+^8|-4> = prod_{K=-4}^{3} sqrt(20 - K(K+1)) = 8!.
  double maxabs = 0.0;
  for (const auto& b : m.blocks) {
    const auto d = b.ops[k].to_dense();
    maxabs = std::max(maxabs, d.cwiseAbs().maxCoeff());
  }
  CHECK_CLOSE(maxabs, 40320.0, 1e-6);
}

TEST_CASE("invalid specs") {
  AsymRotorSpec s;
  s.reduction = "X";
  CHECK_THROWS(models::compile_asym_rotor(s));
  s.reduction = "A";
  s.axis = "IVr";
  CHECK_THROWS(models::compile_asym_rotor(s));
  s.axis = "Ir";
  s.extra_operators = {"Jz"};
  CHECK_THROWS(models::compile_asym_rotor(s));
}

RD_TEST_MAIN
