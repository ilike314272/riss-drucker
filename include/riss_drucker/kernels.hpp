#pragma once
// Hot vector kernels with a scalar reference and runtime-dispatched SIMD
// variants (README section 4). The scalar variant is the reference: every
// SIMD variant must agree with it within a relative tolerance (see tests).
//
// CSR layout: row_ptr has nrows+1 entries (int64), col_idx and values have
// row_ptr[nrows] entries; column indices are int32 (block dimensions < 2^31).
#include <cstddef>
#include <cstdint>

#include "riss_drucker/cpu_features.hpp"

namespace riss_drucker::kernels {

// Default entry points: use active_simd_level().
double dot(std::size_t n, const double* x, const double* y);
void axpy(std::size_t n, double a, const double* x, double* y);  // y += a*x
void csr_matvec(std::size_t nrows, const std::int64_t* row_ptr, const std::int32_t* col_idx,
                const double* values, const double* x, double* y);  // y = A x

// Explicit-level entry points (for tests and benchmarks). Throw
// std::invalid_argument if the level is not available on this build/CPU.
double dot(SimdLevel level, std::size_t n, const double* x, const double* y);
void axpy(SimdLevel level, std::size_t n, double a, const double* x, double* y);
void csr_matvec(SimdLevel level, std::size_t nrows, const std::int64_t* row_ptr,
                const std::int32_t* col_idx, const double* values, const double* x, double* y);

}  // namespace riss_drucker::kernels
