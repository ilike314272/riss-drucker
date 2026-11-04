// Benchmark gate for dispatched kernels (README section 4): each SIMD variant
// must show its speedup over the scalar baseline and its max deviation from it.
// Output: one JSON object per line.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "riss_drucker/build_info.hpp"
#include "riss_drucker/kernels.hpp"

using namespace riss_drucker;
using Clock = std::chrono::steady_clock;

template <class F>
double best_seconds(F&& f, int reps) {
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    f();
    best = std::min(best, std::chrono::duration<double>(Clock::now() - t0).count());
  }
  return best;
}

int main() {
  std::mt19937_64 rng(2024);
  std::normal_distribution<double> d;
  volatile double sink = 0;

  // dot: L2-resident and memory-bound sizes.
  for (std::size_t n : {std::size_t(4096), std::size_t(1) << 22}) {
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = d(rng);
    for (auto& v : y) v = d(rng);
    const int inner = n < 100000 ? 2000 : 10;
    double t_scalar = 0, ref = kernels::dot(SimdLevel::Scalar, n, x.data(), y.data());
    for (SimdLevel l : available_simd_levels()) {
      const double t = best_seconds([&] { for (int i = 0; i < inner; ++i) sink = sink + kernels::dot(l, n, x.data(), y.data()); }, 5) / inner;
      if (l == SimdLevel::Scalar) t_scalar = t;
      const double dev = std::abs(kernels::dot(l, n, x.data(), y.data()) - ref);
      std::printf("{\"kernel\":\"dot\",\"n\":%zu,\"level\":\"%s\",\"seconds\":%.3e,\"speedup\":%.2f,\"abs_dev_vs_scalar\":%.2e}\n",
                  n, to_string(l), t, t_scalar / t, dev);
    }
  }

  // CSR matvec: banded-random sparse matrix, ~32 nnz/row (Lanczos-like).
  {
    const std::size_t n = 200000, per_row = 32;
    std::vector<std::int64_t> rp{0};
    std::vector<std::int32_t> ci;
    std::vector<double> v;
    std::uniform_int_distribution<int> off(-2000, 2000);
    for (std::size_t r = 0; r < n; ++r) {
      std::vector<std::int32_t> cols;
      for (std::size_t k = 0; k < per_row; ++k)
        cols.push_back(static_cast<std::int32_t>(std::clamp<long>(long(r) + off(rng), 0, long(n) - 1)));
      std::sort(cols.begin(), cols.end());
      for (auto c : cols) { ci.push_back(c); v.push_back(d(rng)); }
      rp.push_back(static_cast<std::int64_t>(ci.size()));
    }
    std::vector<double> x(n), y(n), yref(n);
    for (auto& e : x) e = d(rng);
    kernels::csr_matvec(SimdLevel::Scalar, n, rp.data(), ci.data(), v.data(), x.data(), yref.data());
    double t_scalar = 0;
    for (SimdLevel l : available_simd_levels()) {
      const double t = best_seconds([&] { kernels::csr_matvec(l, n, rp.data(), ci.data(), v.data(), x.data(), y.data()); }, 10);
      if (l == SimdLevel::Scalar) t_scalar = t;
      double dev = 0;
      for (std::size_t i = 0; i < n; ++i) dev = std::max(dev, std::abs(y[i] - yref[i]));
      std::printf("{\"kernel\":\"csr_matvec\",\"n\":%zu,\"nnz\":%zu,\"level\":\"%s\",\"seconds\":%.3e,\"speedup\":%.2f,\"abs_dev_vs_scalar\":%.2e}\n",
                  n, ci.size(), to_string(l), t, t_scalar / t, dev);
    }
  }
  std::printf("{\"git_commit\":\"%s\",\"active_level\":\"%s\"}\n", build_info().git_commit.c_str(),
              to_string(active_simd_level()));
  return sink == 12345.0;  // keep sink alive
}
