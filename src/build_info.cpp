#include "riss_drucker/build_info.hpp"

#include <sstream>

#include "riss_drucker/build_config.hpp"

namespace riss_drucker {

const BuildInfo& build_info() {
  static const BuildInfo info = [] {
    BuildInfo b;
    b.version = RISS_DRUCKER_VERSION_STRING;
    b.git_commit = RISS_DRUCKER_GIT_COMMIT;
    b.compiler_id = RISS_DRUCKER_COMPILER_ID;
    b.compiler_version = RISS_DRUCKER_COMPILER_VERSION;
    b.build_type = RISS_DRUCKER_BUILD_TYPE;
    b.cxx_flags = RISS_DRUCKER_CXX_FLAGS;
    b.avx2_flags = RISS_DRUCKER_AVX2_FLAGS;
    b.avx512_flags = RISS_DRUCKER_AVX512_FLAGS;
    std::stringstream ss(RISS_DRUCKER_LINALG_BACKENDS);
    for (std::string item; std::getline(ss, item, ',');) b.linalg_backends.push_back(item);
    b.simd_dispatch = RISS_DRUCKER_OPT_SIMD_DISPATCH;
    b.openmp = RISS_DRUCKER_OPT_OPENMP;
    b.mkl = RISS_DRUCKER_OPT_MKL;
    b.lapack = RISS_DRUCKER_OPT_LAPACK;
    return b;
  }();
  return info;
}

}  // namespace riss_drucker
