#pragma once
// Runtime CPU feature detection and SIMD level selection (README section 4).
#include <string>
#include <vector>

namespace riss_drucker {

struct CpuFeatures {
  bool x86_64_v2 = false;  // SSE4.2 + POPCNT + SSSE3: required by the baseline build
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  bool avx512dq = false;
};

// Detected once; includes OS support for the wider register state.
const CpuFeatures& cpu_features();

enum class SimdLevel : int { Scalar = 0, AVX2 = 1, AVX512 = 2 };

const char* to_string(SimdLevel level);
// Parses "scalar" | "avx2" | "avx512" (case-insensitive). Throws on anything else.
SimdLevel parse_simd_level(const std::string& name);

// Levels that are both compiled in and supported by this CPU, ascending.
std::vector<SimdLevel> available_simd_levels();
bool simd_level_available(SimdLevel level);

// Resolves a requested level (value of RISS_DRUCKER_SIMD, may be null/empty or
// "auto") against the available levels: the result is the highest available
// level not above the request. Pure function, exposed for testing.
SimdLevel resolve_simd_level(const char* request, const std::vector<SimdLevel>& available);

// Level used by the default kernel entry points. Fixed at first use from the
// environment variable RISS_DRUCKER_SIMD (default "auto").
SimdLevel active_simd_level();

}  // namespace riss_drucker
