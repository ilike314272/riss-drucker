// AVX-512 (F + DQ) kernels. This TU alone is compiled with -mavx512f -mavx512dq.
// Called only after runtime detection confirms AVX-512F/DQ with OS support.
#include <immintrin.h>

#include "kernels_impl.hpp"

namespace riss_drucker::kernels::avx512 {

double dot(std::size_t n, const double* x, const double* y) {
  __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
  __m512d a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_fmadd_pd(_mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i), a0);
    a1 = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + 8), _mm512_loadu_pd(y + i + 8), a1);
    a2 = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + 16), _mm512_loadu_pd(y + i + 16), a2);
    a3 = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + 24), _mm512_loadu_pd(y + i + 24), a3);
  }
  for (; i + 8 <= n; i += 8)
    a0 = _mm512_fmadd_pd(_mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i), a0);
  if (i < n) {  // masked tail
    const __mmask8 m = static_cast<__mmask8>((1u << (n - i)) - 1u);
    a1 = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(m, x + i), _mm512_maskz_loadu_pd(m, y + i), a1);
  }
  return _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3)));
}

void axpy(std::size_t n, double a, const double* x, double* y) {
  const __m512d va = _mm512_set1_pd(a);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(y + i, _mm512_fmadd_pd(va, _mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i)));
  if (i < n) {
    const __mmask8 m = static_cast<__mmask8>((1u << (n - i)) - 1u);
    _mm512_mask_storeu_pd(
        y + i, m, _mm512_fmadd_pd(va, _mm512_maskz_loadu_pd(m, x + i), _mm512_maskz_loadu_pd(m, y + i)));
  }
}

void csr_matvec(std::size_t nrows, const std::int64_t* row_ptr, const std::int32_t* col_idx,
                const double* values, const double* x, double* y) {
  for (std::size_t r = 0; r < nrows; ++r) {
    std::int64_t k = row_ptr[r];
    const std::int64_t end = row_ptr[r + 1];
    __m512d acc = _mm512_setzero_pd();
    for (; k + 8 <= end; k += 8) {
      const __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_idx + k));
      const __m512d xv = _mm512_i32gather_pd(idx, x, 8);
      acc = _mm512_fmadd_pd(_mm512_loadu_pd(values + k), xv, acc);
    }
    double s = _mm512_reduce_add_pd(acc);
    for (; k < end; ++k) s += values[k] * x[col_idx[k]];
    y[r] = s;
  }
}

}  // namespace riss_drucker::kernels::avx512
