// Scalar reference kernels. Compiled with the portable baseline flags and
// -ffp-contract=off: this is the numerical reference for all SIMD variants.
#include "kernels_impl.hpp"

namespace riss_drucker::kernels::scalar {

double dot(std::size_t n, const double* x, const double* y) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
  return s;
}

void axpy(std::size_t n, double a, const double* x, double* y) {
  for (std::size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void csr_matvec(std::size_t nrows, const std::int64_t* row_ptr, const std::int32_t* col_idx,
                const double* values, const double* x, double* y) {
  for (std::size_t r = 0; r < nrows; ++r) {
    double s = 0.0;
    for (std::int64_t k = row_ptr[r]; k < row_ptr[r + 1]; ++k) s += values[k] * x[col_idx[k]];
    y[r] = s;
  }
}

}  // namespace riss_drucker::kernels::scalar
