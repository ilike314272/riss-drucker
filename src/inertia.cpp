#include "riss_drucker/inertia.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(RISS_DRUCKER_HAVE_LAPACK) || defined(RISS_DRUCKER_HAVE_MKL)
#define RD_HAVE_FORTRAN_LAPACK 1
extern "C" void dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv,
                        double* work, const int* lwork, int* info, std::size_t uplo_len);
#endif

namespace riss_drucker {
namespace {

void count_1x1(double d, Inertia& in) {
  in.min_abs_pivot = std::min(in.min_abs_pivot, std::abs(d));
  if (d < 0) ++in.n_neg;
  else if (d > 0) ++in.n_pos;
  else ++in.n_zero;
}

// Symmetric 2x2 pivot [[a, b], [b, c]]: count the signs of its eigenvalues.
void count_2x2(double a, double b, double c, Inertia& in) {
  const double mean = 0.5 * (a + c);
  const double rad = std::hypot(0.5 * (a - c), b);
  count_1x1(mean - rad, in);
  count_1x1(mean + rad, in);
}

Eigen::MatrixXd shifted_lower(const Eigen::MatrixXd& a, double sigma) {
  if (a.rows() != a.cols()) throw std::invalid_argument("inertia: matrix must be square");
  Eigen::MatrixXd s = a.triangularView<Eigen::Lower>();
  s.diagonal().array() -= sigma;
  return s;
}

}  // namespace

Inertia inertia_bunch_kaufman(const Eigen::MatrixXd& a, double sigma) {
  Eigen::MatrixXd A = shifted_lower(a, sigma);
  const Eigen::Index n = A.rows();
  const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
  Inertia in;
  in.method = "bunch_kaufman";

  Eigen::Index k = 0;
  while (k < n) {
    int kstep = 1;
    Eigen::Index kp = k;
    const double absakk = std::abs(A(k, k));
    double colmax = 0.0;
    Eigen::Index imax = k;
    for (Eigen::Index i = k + 1; i < n; ++i)
      if (std::abs(A(i, k)) > colmax) { colmax = std::abs(A(i, k)); imax = i; }

    if (std::max(absakk, colmax) == 0.0) {  // column k is exactly zero
      count_1x1(0.0, in);
      ++k;
      continue;
    }
    if (absakk < alpha * colmax) {
      double rowmax = 0.0;  // largest off-diagonal in row/column imax of the trailing block
      for (Eigen::Index j = k; j < imax; ++j) rowmax = std::max(rowmax, std::abs(A(imax, j)));
      for (Eigen::Index j = imax + 1; j < n; ++j) rowmax = std::max(rowmax, std::abs(A(j, imax)));
      if (absakk >= alpha * colmax * (colmax / rowmax)) {
        kp = k;
      } else if (std::abs(A(imax, imax)) >= alpha * rowmax) {
        kp = imax;
      } else {
        kp = imax;
        kstep = 2;
      }
    }

    const Eigen::Index kk = k + kstep - 1;
    if (kp != kk) {  // symmetric interchange of kk and kp in A(k:n, k:n), lower storage (as dsytf2)
      for (Eigen::Index i = kp + 1; i < n; ++i) std::swap(A(i, kk), A(i, kp));
      for (Eigen::Index j = kk + 1; j < kp; ++j) std::swap(A(j, kk), A(kp, j));
      std::swap(A(kk, kk), A(kp, kp));
      if (kstep == 2) std::swap(A(k + 1, k), A(kp, k));
    }

    const Eigen::Index m = n - k - kstep;  // trailing size
    if (kstep == 1) {
      const double d = A(k, k);
      count_1x1(d, in);
      if (m > 0) {
        const Eigen::VectorXd c = A.col(k).tail(m);
        A.bottomRightCorner(m, m).selfadjointView<Eigen::Lower>().rankUpdate(c, -1.0 / d);
      }
    } else {
      const double d11 = A(k, k), d21 = A(k + 1, k), d22 = A(k + 1, k + 1);
      count_2x2(d11, d21, d22, in);
      if (m > 0) {
        const Eigen::MatrixXd C = A.block(k + 2, k, m, 2);
        Eigen::Matrix2d D;
        D << d11, d21, d21, d22;
        const Eigen::MatrixXd W = C * D.inverse();
        A.bottomRightCorner(m, m).triangularView<Eigen::Lower>() -= W * C.transpose();
      }
    }
    k += kstep;
  }
  return in;
}

Inertia inertia_dense(const Eigen::MatrixXd& a, double sigma, linalg::Backend backend) {
  if (!linalg::backend_available(backend))
    throw std::invalid_argument(std::string("inertia: linalg backend '") + linalg::to_string(backend) +
                                "' was not built");
#ifdef RD_HAVE_FORTRAN_LAPACK
  if (backend == linalg::Backend::Lapack || backend == linalg::Backend::Mkl) {
    Eigen::MatrixXd A = shifted_lower(a, sigma);
    const int n = static_cast<int>(A.rows());
    Inertia in;
    in.method = "bunch_kaufman_dsytrf";
    if (n == 0) return in;
    std::vector<int> ipiv(static_cast<std::size_t>(n));
    int info = 0, lwork = -1;
    double wq = 0.0;
    const char uplo = 'L';
    dsytrf_(&uplo, &n, A.data(), &n, ipiv.data(), &wq, &lwork, &info, 1);
    lwork = std::max(1, static_cast<int>(wq));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsytrf_(&uplo, &n, A.data(), &n, ipiv.data(), work.data(), &lwork, &info, 1);
    if (info < 0) throw std::runtime_error("dsytrf: invalid argument " + std::to_string(-info));
    // info > 0: an exact zero pivot; the factorization is still complete.
    for (int k = 0; k < n;) {
      if (ipiv[static_cast<std::size_t>(k)] > 0) {
        count_1x1(A(k, k), in);
        k += 1;
      } else {
        count_2x2(A(k, k), A(k + 1, k), A(k + 1, k + 1), in);
        k += 2;
      }
    }
    return in;
  }
#endif
  return inertia_bunch_kaufman(a, sigma);
}

Inertia sturm_count(const Eigen::VectorXd& diag, const Eigen::VectorXd& offdiag, double sigma) {
  const Eigen::Index n = diag.size();
  if (n > 0 && offdiag.size() != n - 1)
    throw std::invalid_argument("sturm_count: offdiag must have n-1 entries");
  Inertia in;
  in.method = "sturm";
  if (n == 0) return in;
  const double emax2 = offdiag.size() ? offdiag.cwiseAbs2().maxCoeff() : 0.0;
  const double pivmin = DBL_MIN * std::max(1.0, emax2);
  double q = diag[0] - sigma;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (i > 0) q = (diag[i] - sigma) - offdiag[i - 1] * offdiag[i - 1] / q;
    if (std::abs(q) < pivmin) q = -pivmin;
    in.min_abs_pivot = std::min(in.min_abs_pivot, std::abs(q));
    if (q < 0) ++in.n_neg;
    else ++in.n_pos;
  }
  return in;
}

}  // namespace riss_drucker
