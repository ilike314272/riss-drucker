// Benchmark per solver path (README 3.1): dense eigh, Lanczos and the inertia
// certificates (Sturm, Bunch-Kaufman), on torsion blocks of growing size.
#include <algorithm>
#include <chrono>
#include <cstdio>

#include "riss_drucker/models/torsion_rotor.hpp"
#include "riss_drucker/solver.hpp"

using namespace riss_drucker;
using Clock = std::chrono::steady_clock;

template <class F>
double seconds(F&& f, int reps = 3) {
  double best = 1e300;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    f();
    best = std::min(best, std::chrono::duration<double>(Clock::now() - t0).count());
  }
  return best;
}

int main() {
  for (int m_max : {300, 1500, 6000}) {
    models::TorsionRotorSpec spec;
    spec.n_fold = 3;
    spec.v_orders = {1, 2};  // pentadiagonal blocks: Bunch-Kaufman certificate
    spec.m_max = m_max;
    const auto model = models::compile_torsion_rotor(spec);
    const auto& blk = model.blocks.front();  // sigma=0,+ (cos) block
    const CsrMatrix h = assemble(blk, {5.0, 400.0, -30.0});
    const auto n = h.rows;

    SolveOptions o;
    o.nlowest = 10;
    for (auto m : {SolveMethod::Dense, SolveMethod::Lanczos}) {
      if (m == SolveMethod::Dense && n > 2500) continue;
      o.method = m;
      bool cert = false;
      const double t = seconds([&] { cert = solve_block(h, o).certified; });
      std::printf("{\"path\":\"%s\",\"dim\":%lld,\"nlowest\":10,\"seconds\":%.3e,\"certified\":%s}\n",
                  to_string(m), static_cast<long long>(n), t, cert ? "true" : "false");
    }
    if (n <= 2500) {
      const Eigen::MatrixXd d = h.to_dense();
      const double t = seconds([&] { (void)inertia_bunch_kaufman(d, 1000.0); });
      std::printf("{\"path\":\"bunch_kaufman\",\"dim\":%lld,\"seconds\":%.3e}\n", static_cast<long long>(n), t);
    }
    // Sturm on the tridiagonal V3-only block of the same size.
    spec.v_orders = {1};
    const auto tri = assemble(models::compile_torsion_rotor(spec).blocks.front(), {5.0, 400.0});
    const Eigen::VectorXd dg = tri.diagonal(), od = tri.subdiagonal();
    const double t = seconds([&] {
      for (int s = 0; s < 1000; ++s) (void)sturm_count(dg, od, 10.0 * s);
    }) / 1000.0;
    std::printf("{\"path\":\"sturm\",\"dim\":%lld,\"seconds_per_shift\":%.3e}\n", static_cast<long long>(tri.rows), t);
  }
  return 0;
}
