// AVX2 + FMA kernels. This TU alone is compiled with -mavx2 -mfma.
// Called only after runtime detection confirms AVX2 and FMA.
#include <immintrin.h>

#include "kernels_impl.hpp"

namespace riss_drucker::kernels::avx2 {

static inline double hsum(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  lo = _mm_add_pd(lo, hi);
  __m128d sh = _mm_unpackhi_pd(lo, lo);
  return _mm_cvtsd_f64(_mm_add_sd(lo, sh));
}

double dot(std::size_t n, const double* x, const double* y) {
  __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
  __m256d a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    a0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), a0);
    a1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4), _mm256_loadu_pd(y + i + 4), a1);
    a2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 8), _mm256_loadu_pd(y + i + 8), a2);
    a3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 12), _mm256_loadu_pd(y + i + 12), a3);
  }
  for (; i + 4 <= n; i += 4)
    a0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), a0);
  double s = hsum(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
  for (; i < n; ++i) s += x[i] * y[i];
  return s;
}

void axpy(std::size_t n, double a, const double* x, double* y) {
  const __m256d va = _mm256_set1_pd(a);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(y + i, _mm256_fmadd_pd(va, _mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i)));
  for (; i < n; ++i) y[i] += a * x[i];
}

void csr_matvec(std::size_t nrows, const std::int64_t* row_ptr, const std::int32_t* col_idx,
                const double* values, const double* x, double* y) {
  for (std::size_t r = 0; r < nrows; ++r) {
    std::int64_t k = row_ptr[r];
    const std::int64_t end = row_ptr[r + 1];
    __m256d acc = _mm256_setzero_pd();
    for (; k + 4 <= end; k += 4) {
      const __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col_idx + k));
      const __m256d xv = _mm256_i32gather_pd(x, idx, 8);
      acc = _mm256_fmadd_pd(_mm256_loadu_pd(values + k), xv, acc);
    }
    double s = hsum(acc);
    for (; k < end; ++k) s += values[k] * x[col_idx[k]];
    y[r] = s;
  }
}

}  // namespace riss_drucker::kernels::avx2
