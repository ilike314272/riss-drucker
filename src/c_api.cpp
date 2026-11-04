#include "riss_drucker/c_api.h"

#include <exception>

#include "riss_drucker/build_info.hpp"
#include "riss_drucker/kernels.hpp"
#include "riss_drucker/linalg_backend.hpp"

using namespace riss_drucker;

extern "C" {

const char* rd_version(void) { return build_info().version.c_str(); }
const char* rd_git_commit(void) { return build_info().git_commit.c_str(); }
const char* rd_simd_level(void) { return to_string(active_simd_level()); }

double rd_dot(size_t n, const double* x, const double* y) { return kernels::dot(n, x, y); }
void rd_axpy(size_t n, double a, const double* x, double* y) { kernels::axpy(n, a, x, y); }
void rd_csr_matvec(size_t nrows, const int64_t* row_ptr, const int32_t* col_idx,
                   const double* values, const double* x, double* y) {
  kernels::csr_matvec(nrows, row_ptr, col_idx, values, x, y);
}

int rd_eigvalsh(size_t n, const double* a, double* w) {
  try {
    const Eigen::Map<const Eigen::MatrixXd> m(a, static_cast<Eigen::Index>(n),
                                              static_cast<Eigen::Index>(n));
    const auto r = linalg::eigh(Eigen::MatrixXd(m), false);
    for (size_t i = 0; i < n; ++i) w[i] = r.eigenvalues[static_cast<Eigen::Index>(i)];
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}

}  // extern "C"
