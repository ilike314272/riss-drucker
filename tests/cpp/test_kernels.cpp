// Every available SIMD variant must agree with the scalar reference within a
// stated tolerance (README section 4, numerical safety).
#include <cstdint>
#include <random>
#include <vector>

#include "riss_drucker/kernels.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;

namespace {

// Tolerance for a length-n reduction of O(1) terms: summation-order and FMA
// differences are bounded by ~n * eps * sum|x_i y_i|.
double reduction_tol(std::size_t n, double abs_sum) {
  return 4.0 * static_cast<double>(n + 1) * 2.220446049250313e-16 * (abs_sum + 1.0);
}

std::vector<double> random_vec(std::size_t n, std::mt19937_64& rng) {
  std::normal_distribution<double> d;
  std::vector<double> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

struct Csr {
  std::size_t nrows, ncols;
  std::vector<std::int64_t> rp;
  std::vector<std::int32_t> ci;
  std::vector<double> v;
};

Csr random_csr(std::size_t nrows, std::size_t ncols, double density, std::mt19937_64& rng) {
  Csr m{nrows, ncols, {0}, {}, {}};
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::normal_distribution<double> d;
  for (std::size_t r = 0; r < nrows; ++r) {
    for (std::size_t c = 0; c < ncols; ++c)
      if (u(rng) < density) {
        m.ci.push_back(static_cast<std::int32_t>(c));
        m.v.push_back(d(rng));
      }
    m.rp.push_back(static_cast<std::int64_t>(m.ci.size()));
  }
  return m;
}

}  // namespace

TEST_CASE("dot: SIMD variants agree with scalar, all tail lengths") {
  std::mt19937_64 rng(1234);
  for (SimdLevel lvl : available_simd_levels()) {
    for (std::size_t n : {0u, 1u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 33u, 1000u, 100003u}) {
      auto x = random_vec(n, rng), y = random_vec(n, rng);
      double abs_sum = 0;
      for (std::size_t i = 0; i < n; ++i) abs_sum += std::abs(x[i] * y[i]);
      const double ref = kernels::dot(SimdLevel::Scalar, n, x.data(), y.data());
      const double got = kernels::dot(lvl, n, x.data(), y.data());
      CHECK_CLOSE(got, ref, reduction_tol(n, abs_sum));
    }
  }
}

TEST_CASE("axpy: SIMD variants agree with scalar") {
  std::mt19937_64 rng(99);
  for (SimdLevel lvl : available_simd_levels()) {
    for (std::size_t n : {0u, 1u, 3u, 7u, 8u, 13u, 1001u}) {
      auto x = random_vec(n, rng), y0 = random_vec(n, rng);
      auto yr = y0, yg = y0;
      kernels::axpy(SimdLevel::Scalar, n, 0.37, x.data(), yr.data());
      kernels::axpy(lvl, n, 0.37, x.data(), yg.data());
      for (std::size_t i = 0; i < n; ++i)
        CHECK_CLOSE(yg[i], yr[i], 4e-16 * (std::abs(yr[i]) + std::abs(0.37 * x[i]) + 1.0));
    }
  }
}

TEST_CASE("csr_matvec: SIMD variants agree with scalar and with dense product") {
  std::mt19937_64 rng(7);
  for (SimdLevel lvl : available_simd_levels()) {
    for (double density : {0.0, 0.05, 0.3, 1.0}) {
      const auto m = random_csr(53, 61, density, rng);
      const auto x = random_vec(m.ncols, rng);
      std::vector<double> yr(m.nrows), yg(m.nrows), yd(m.nrows, 0.0);
      kernels::csr_matvec(SimdLevel::Scalar, m.nrows, m.rp.data(), m.ci.data(), m.v.data(),
                          x.data(), yr.data());
      kernels::csr_matvec(lvl, m.nrows, m.rp.data(), m.ci.data(), m.v.data(), x.data(), yg.data());
      for (std::size_t r = 0; r < m.nrows; ++r) {
        double abs_sum = 0;
        for (auto k = m.rp[r]; k < m.rp[r + 1]; ++k) {
          yd[r] += m.v[k] * x[m.ci[k]];
          abs_sum += std::abs(m.v[k] * x[m.ci[k]]);
        }
        const double tol = reduction_tol(static_cast<std::size_t>(m.rp[r + 1] - m.rp[r]), abs_sum);
        CHECK_CLOSE(yg[r], yr[r], tol);
        CHECK_CLOSE(yr[r], yd[r], tol);
      }
    }
  }
}

TEST_CASE("default entry points use the active level and match it exactly") {
  std::mt19937_64 rng(3);
  auto x = random_vec(1234, rng), y = random_vec(1234, rng);
  const double a = kernels::dot(1234, x.data(), y.data());
  const double b = kernels::dot(active_simd_level(), 1234, x.data(), y.data());
  CHECK(a == b);  // same code path, same thread: bitwise identical
}

TEST_CASE("unavailable level throws instead of executing illegal instructions") {
  double x = 1.0;
  for (SimdLevel l : {SimdLevel::AVX2, SimdLevel::AVX512})
    if (!simd_level_available(l)) CHECK_THROWS(kernels::dot(l, 1, &x, &x));
}

RD_TEST_MAIN
