// Runtime dispatch between scalar / AVX2 / AVX-512 kernel variants.
#include <stdexcept>
#include <string>

#include "kernels_impl.hpp"
#include "riss_drucker/kernels.hpp"

namespace riss_drucker::kernels {
namespace {

struct Table {
  double (*dot)(std::size_t, const double*, const double*);
  void (*axpy)(std::size_t, double, const double*, double*);
  void (*csr_matvec)(std::size_t, const std::int64_t*, const std::int32_t*, const double*,
                     const double*, double*);
};

constexpr Table kScalar{scalar::dot, scalar::axpy, scalar::csr_matvec};
#ifdef RISS_DRUCKER_HAVE_SIMD_DISPATCH
constexpr Table kAvx2{avx2::dot, avx2::axpy, avx2::csr_matvec};
constexpr Table kAvx512{avx512::dot, avx512::axpy, avx512::csr_matvec};
#endif

const Table& table_for(SimdLevel level) {
  if (!simd_level_available(level))
    throw std::invalid_argument(std::string("SIMD level '") + to_string(level) +
                                "' is not available on this build/CPU");
  switch (level) {
#ifdef RISS_DRUCKER_HAVE_SIMD_DISPATCH
    case SimdLevel::AVX2: return kAvx2;
    case SimdLevel::AVX512: return kAvx512;
#endif
    default: return kScalar;
  }
}

const Table& active() {
  static const Table& t = table_for(active_simd_level());
  return t;
}

}  // namespace

double dot(std::size_t n, const double* x, const double* y) { return active().dot(n, x, y); }
void axpy(std::size_t n, double a, const double* x, double* y) { active().axpy(n, a, x, y); }
void csr_matvec(std::size_t nrows, const std::int64_t* rp, const std::int32_t* ci,
                const double* v, const double* x, double* y) {
  active().csr_matvec(nrows, rp, ci, v, x, y);
}

double dot(SimdLevel l, std::size_t n, const double* x, const double* y) {
  return table_for(l).dot(n, x, y);
}
void axpy(SimdLevel l, std::size_t n, double a, const double* x, double* y) {
  table_for(l).axpy(n, a, x, y);
}
void csr_matvec(SimdLevel l, std::size_t nrows, const std::int64_t* rp, const std::int32_t* ci,
                const double* v, const double* x, double* y) {
  table_for(l).csr_matvec(nrows, rp, ci, v, x, y);
}

}  // namespace riss_drucker::kernels
