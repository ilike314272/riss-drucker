/* Optional C API for non-Python bindings (README section 4). */
#ifndef RISS_DRUCKER_C_API_H
#define RISS_DRUCKER_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* rd_version(void);
const char* rd_git_commit(void);
const char* rd_simd_level(void); /* active level name */

double rd_dot(size_t n, const double* x, const double* y);
void rd_axpy(size_t n, double a, const double* x, double* y);
void rd_csr_matvec(size_t nrows, const int64_t* row_ptr, const int32_t* col_idx,
                   const double* values, const double* x, double* y);

/* Dense symmetric eigenvalues (column-major n x n, lower triangle read).
 * Writes n ascending eigenvalues to w. Returns 0 on success, nonzero on error. */
int rd_eigvalsh(size_t n, const double* a, double* w);

#ifdef __cplusplus
}
#endif

#endif
