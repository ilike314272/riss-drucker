// Open-shell rotor and LAM models: certificates, block structure, half-integer J.
#include <algorithm>
#include <cmath>

#include "riss_drucker/models/lam.hpp"
#include "riss_drucker/models/open_rotor.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;

TEST_CASE("open_rotor: half-integer J blocks, dimensions, certificates") {
  models::OpenRotorSpec s;
  s.two_s = 1;
  s.n_max = 4;
  const auto m = models::compile_open_rotor(s);
  for (const auto& c : m.compile_certificates) CHECK(c.passed);
  for (const auto& b : m.blocks) {
    CHECK(b.id.two_j % 2 == 1);                            // Kramers: half-integer J
    CHECK(b.dim() == (b.id.two_j + 1) * (s.two_s + 1));     // (2J+1)(2S+1) = sum_N (2N+1)
  }
  CHECK(m.blocks.front().id.two_j == 1 && m.blocks.back().id.two_j == 2 * 4 + 1);
  s.two_s = 2;
  s.spin_spin = true;
  const auto t = models::compile_open_rotor(s);
  CHECK(t.blocks.front().id.two_j == 0);
  CHECK(std::find(t.param_names.begin(), t.param_names.end(), "D_ss_N") != t.param_names.end());
}

TEST_CASE("inversion: parity blocks partition the basis; ram_torsion block sizes") {
  models::InversionSpec s;
  s.n_basis = 21;
  s.j_max = 2;
  s.rotation_coupling = true;
  const auto m = models::compile_inversion(s);
  for (const auto& c : m.compile_certificates) CHECK(c.passed);
  int per_jk = 0;
  for (const auto& b : m.blocks)
    if (b.id.two_j == 0) per_jk += static_cast<int>(b.dim());
  CHECK(per_jk == 21);
  CHECK(m.blocks.size() == 2u * (1 + 2 + 3));  // (J, K>=0) pairs x parity
  models::RamTorsionSpec r;
  r.n_fold = 3;
  r.m_max = 6;
  r.j_max = 2;
  const auto t = models::compile_ram_torsion(r);
  for (const auto& b : t.blocks) {
    const int nk = b.id.two_j + 1;
    CHECK(b.dim() % nk == 0);
  }
  CHECK_THROWS(models::compile_inversion(models::InversionSpec{2, 1.0, 4, 0, false}));
}

RD_TEST_MAIN
