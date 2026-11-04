#pragma once
// Build provenance recorded in every run record (README section 3).
#include <string>
#include <vector>

namespace riss_drucker {

struct BuildInfo {
  std::string version;
  std::string git_commit;
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string cxx_flags;      // baseline flags for all non-dispatched code
  std::string avx2_flags;     // flags applied to the AVX2 kernel TU only
  std::string avx512_flags;   // flags applied to the AVX-512 kernel TU only
  std::vector<std::string> linalg_backends;
  bool simd_dispatch = false;
  bool openmp = false;
  bool mkl = false;
  bool lapack = false;
};

const BuildInfo& build_info();

}  // namespace riss_drucker
