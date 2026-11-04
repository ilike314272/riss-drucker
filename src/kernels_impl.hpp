#pragma once
// Internal declarations of the per-ISA kernel variants.
// The SIMD translation units are compiled with ISA flags, so they must not
// instantiate templates or inline functions shared with baseline code (an ODR
// merge could otherwise leak AVX instructions into baseline paths). Keep them
// to plain functions over raw pointers.
#include <cstddef>
#include <cstdint>

#define RD_KERNEL_DECLS                                                                  \
  double dot(std::size_t n, const double* x, const double* y);                         \
  void axpy(std::size_t n, double a, const double* x, double* y);                      \
  void csr_matvec(std::size_t nrows, const std::int64_t* row_ptr,                      \
                  const std::int32_t* col_idx, const double* values, const double* x,  \
                  double* y);

namespace riss_drucker::kernels::scalar { RD_KERNEL_DECLS }
namespace riss_drucker::kernels::avx2 { RD_KERNEL_DECLS }
namespace riss_drucker::kernels::avx512 { RD_KERNEL_DECLS }

#undef RD_KERNEL_DECLS
