#include <vector>

#include "riss_drucker/cpu_features.hpp"
#include "test_harness.hpp"

using namespace riss_drucker;
using L = SimdLevel;

TEST_CASE("baseline CPU requirement: x86-64-v2") { CHECK(cpu_features().x86_64_v2); }

TEST_CASE("feature implications are consistent") {
  const auto& f = cpu_features();
  if (f.avx2) CHECK(f.avx);
  if (f.avx512f) CHECK(f.avx2);
}

TEST_CASE("available levels: scalar first, ascending, consistent with CPU") {
  const auto a = available_simd_levels();
  CHECK(!a.empty() && a.front() == L::Scalar);
  for (std::size_t i = 1; i < a.size(); ++i) CHECK(a[i - 1] < a[i]);
  if (simd_level_available(L::AVX512)) CHECK(cpu_features().avx512f && cpu_features().avx512dq);
  if (simd_level_available(L::AVX2)) CHECK(cpu_features().avx2 && cpu_features().fma);
#ifndef RISS_DRUCKER_HAVE_SIMD_DISPATCH
  CHECK(a.size() == 1);
#endif
}

TEST_CASE("resolve_simd_level: auto picks best, requests are clamped down") {
  const std::vector<L> all{L::Scalar, L::AVX2, L::AVX512};
  const std::vector<L> no512{L::Scalar, L::AVX2};
  const std::vector<L> only{L::Scalar};
  CHECK(resolve_simd_level(nullptr, all) == L::AVX512);
  CHECK(resolve_simd_level("", no512) == L::AVX2);
  CHECK(resolve_simd_level("AUTO", only) == L::Scalar);
  CHECK(resolve_simd_level("scalar", all) == L::Scalar);
  CHECK(resolve_simd_level("avx2", all) == L::AVX2);
  CHECK(resolve_simd_level("avx512", no512) == L::AVX2);
  CHECK(resolve_simd_level("avx2", only) == L::Scalar);
  CHECK_THROWS(resolve_simd_level("sse9", all));
}

TEST_CASE("active level is available") { CHECK(simd_level_available(active_simd_level())); }

RD_TEST_MAIN
