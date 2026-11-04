#include "riss_drucker/cpu_features.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace riss_drucker {

const CpuFeatures& cpu_features() {
  static const CpuFeatures f = [] {
    CpuFeatures c;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    // GCC/Clang check OS support (XGETBV) for AVX / AVX-512 state as well.
    c.x86_64_v2 = __builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("popcnt") &&
                  __builtin_cpu_supports("ssse3");
    c.avx = __builtin_cpu_supports("avx");
    c.avx2 = __builtin_cpu_supports("avx2");
    c.fma = __builtin_cpu_supports("fma");
    c.avx512f = __builtin_cpu_supports("avx512f");
    c.avx512dq = __builtin_cpu_supports("avx512dq");
#endif
    return c;
  }();
  return f;
}

const char* to_string(SimdLevel level) {
  switch (level) {
    case SimdLevel::Scalar: return "scalar";
    case SimdLevel::AVX2: return "avx2";
    case SimdLevel::AVX512: return "avx512";
  }
  return "unknown";
}

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
  return s;
}

SimdLevel parse_simd_level(const std::string& name) {
  const std::string n = lower(name);
  if (n == "scalar") return SimdLevel::Scalar;
  if (n == "avx2") return SimdLevel::AVX2;
  if (n == "avx512") return SimdLevel::AVX512;
  throw std::invalid_argument("unknown SIMD level '" + name + "' (expected scalar|avx2|avx512)");
}

std::vector<SimdLevel> available_simd_levels() {
  std::vector<SimdLevel> out{SimdLevel::Scalar};
#ifdef RISS_DRUCKER_HAVE_SIMD_DISPATCH
  const auto& f = cpu_features();
  if (f.avx2 && f.fma) out.push_back(SimdLevel::AVX2);
  if (f.avx2 && f.fma && f.avx512f && f.avx512dq) out.push_back(SimdLevel::AVX512);
#endif
  return out;
}

bool simd_level_available(SimdLevel level) {
  const auto a = available_simd_levels();
  return std::find(a.begin(), a.end(), level) != a.end();
}

SimdLevel resolve_simd_level(const char* request, const std::vector<SimdLevel>& available) {
  SimdLevel best = SimdLevel::Scalar;
  for (SimdLevel l : available) best = std::max(best, l);
  if (request == nullptr || *request == '\0' || lower(request) == "auto") return best;
  const SimdLevel want = parse_simd_level(request);
  SimdLevel chosen = SimdLevel::Scalar;
  for (SimdLevel l : available)
    if (l <= want) chosen = std::max(chosen, l);
  return chosen;
}

SimdLevel active_simd_level() {
  static const SimdLevel level =
      resolve_simd_level(std::getenv("RISS_DRUCKER_SIMD"), available_simd_levels());
  return level;
}

}  // namespace riss_drucker
