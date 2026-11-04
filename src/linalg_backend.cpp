#include "riss_drucker/linalg_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <Eigen/Eigenvalues>

#if defined(RISS_DRUCKER_HAVE_LAPACK) || defined(RISS_DRUCKER_HAVE_MKL)
#define RD_HAVE_FORTRAN_LAPACK 1
// Fortran LAPACK ABI (LP64). Both reference LAPACK/OpenBLAS and oneMKL (lp64)
// export dsyevd_. The trailing hidden arguments are the character lengths.
extern "C" void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
                        const int* lda, double* w, double* work, const int* lwork, int* iwork,
                        const int* liwork, int* info, std::size_t jobz_len, std::size_t uplo_len);
#endif

namespace riss_drucker::linalg {

const char* to_string(Backend b) {
  switch (b) {
    case Backend::Eigen: return "eigen";
    case Backend::Lapack: return "lapack";
    case Backend::Mkl: return "mkl";
  }
  return "unknown";
}

Backend parse_backend(const std::string& name) {
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
  if (n == "eigen") return Backend::Eigen;
  if (n == "lapack") return Backend::Lapack;
  if (n == "mkl") return Backend::Mkl;
  throw std::invalid_argument("unknown linalg backend '" + name + "' (expected eigen|lapack|mkl)");
}

std::vector<Backend> available_backends() {
  std::vector<Backend> out{Backend::Eigen};
#ifdef RISS_DRUCKER_HAVE_LAPACK
  out.push_back(Backend::Lapack);
#endif
#ifdef RISS_DRUCKER_HAVE_MKL
  out.push_back(Backend::Mkl);
#endif
  return out;
}

bool backend_available(Backend b) {
  const auto a = available_backends();
  return std::find(a.begin(), a.end(), b) != a.end();
}

Backend default_backend() {
  const char* env = std::getenv("RISS_DRUCKER_LINALG");
  if (env == nullptr || *env == '\0') return Backend::Eigen;
  const Backend b = parse_backend(env);
  if (!backend_available(b))
    throw std::runtime_error(std::string("RISS_DRUCKER_LINALG=") + env +
                             " requested but that backend was not built");
  return b;
}

namespace {

EighResult eigh_eigen(const Eigen::MatrixXd& a, bool vectors) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      a, vectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
  if (es.info() != Eigen::Success) throw std::runtime_error("Eigen eigh failed to converge");
  EighResult r;
  r.eigenvalues = es.eigenvalues();
  if (vectors) r.eigenvectors = es.eigenvectors();
  return r;
}

#ifdef RD_HAVE_FORTRAN_LAPACK
EighResult eigh_lapack(const Eigen::MatrixXd& a, bool vectors) {
  const int n = static_cast<int>(a.rows());
  Eigen::MatrixXd v = a;  // column-major copy, overwritten with eigenvectors
  Eigen::VectorXd w(n);
  const char jobz = vectors ? 'V' : 'N';
  const char uplo = 'L';
  int info = 0, lwork = -1, liwork = -1, iwork_query = 0;
  double work_query = 0.0;
  dsyevd_(&jobz, &uplo, &n, v.data(), &n, w.data(), &work_query, &lwork, &iwork_query, &liwork,
          &info, 1, 1);
  if (info != 0) throw std::runtime_error("dsyevd workspace query failed, info=" + std::to_string(info));
  lwork = static_cast<int>(work_query);
  liwork = std::max(1, iwork_query);
  std::vector<double> work(static_cast<std::size_t>(std::max(1, lwork)));
  std::vector<int> iwork(static_cast<std::size_t>(liwork));
  dsyevd_(&jobz, &uplo, &n, v.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork,
          &info, 1, 1);
  if (info != 0) throw std::runtime_error("dsyevd failed, info=" + std::to_string(info));
  EighResult r;
  r.eigenvalues = std::move(w);
  if (vectors) r.eigenvectors = std::move(v);
  return r;
}
#endif

}  // namespace

EighResult eigh(const Eigen::MatrixXd& a, Backend backend, bool compute_vectors) {
  if (a.rows() != a.cols()) throw std::invalid_argument("eigh: matrix must be square");
  if (!backend_available(backend))
    throw std::invalid_argument(std::string("eigh: linalg backend '") + to_string(backend) +
                                "' was not built");
  if (a.rows() == 0) return EighResult{};
  switch (backend) {
#ifdef RD_HAVE_FORTRAN_LAPACK
    case Backend::Lapack:
    case Backend::Mkl: return eigh_lapack(a, compute_vectors);
#endif
    default: return eigh_eigen(a, compute_vectors);
  }
}

double dense_eigh_bytes(long n) {
  // input copy + eigenvectors + O(n^2) workspace (dsyevd needs ~2n^2 doubles).
  const double nn = static_cast<double>(n) * static_cast<double>(n);
  return 8.0 * 4.0 * nn;
}

}  // namespace riss_drucker::linalg
