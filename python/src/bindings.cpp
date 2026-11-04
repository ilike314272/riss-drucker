// pybind11 module riss_drucker._core. Thin layer: validation and conversion
// only; all logic lives in the C++ library.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include "riss_drucker/build_info.hpp"
#include "riss_drucker/cpu_features.hpp"
#include "riss_drucker/kernels.hpp"
#include "riss_drucker/linalg_backend.hpp"

namespace py = pybind11;
using namespace riss_drucker;

namespace {

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

SimdLevel level_or_active(const std::optional<std::string>& level) {
  return level ? parse_simd_level(*level) : active_simd_level();
}

void require_1d(const py::buffer_info& b, const char* name) {
  if (b.ndim != 1) throw py::value_error(std::string(name) + " must be 1-D");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "riss_drucker core numerics (C++)";
  m.attr("__version__") = build_info().version;

  m.def("build_info", [] {
    const auto& b = build_info();
    py::dict d;
    d["version"] = b.version;
    d["git_commit"] = b.git_commit;
    d["compiler_id"] = b.compiler_id;
    d["compiler_version"] = b.compiler_version;
    d["build_type"] = b.build_type;
    d["cxx_flags"] = b.cxx_flags;
    d["avx2_flags"] = b.avx2_flags;
    d["avx512_flags"] = b.avx512_flags;
    d["linalg_backends"] = b.linalg_backends;
    d["simd_dispatch"] = b.simd_dispatch;
    d["openmp"] = b.openmp;
    d["mkl"] = b.mkl;
    d["lapack"] = b.lapack;
    return d;
  }, "Compile-time provenance (git commit, compiler, flags, options).");

  m.def("cpu_features", [] {
    const auto& f = cpu_features();
    py::dict d;
    d["x86_64_v2"] = f.x86_64_v2;
    d["avx"] = f.avx;
    d["avx2"] = f.avx2;
    d["fma"] = f.fma;
    d["avx512f"] = f.avx512f;
    d["avx512dq"] = f.avx512dq;
    return d;
  });

  m.def("simd_levels_available", [] {
    std::vector<std::string> out;
    for (auto l : available_simd_levels()) out.emplace_back(to_string(l));
    return out;
  });
  m.def("simd_level_active", [] { return std::string(to_string(active_simd_level())); });

  m.def("dot", [](DArr x, DArr y, std::optional<std::string> level) {
    auto bx = x.request(), by = y.request();
    require_1d(bx, "x"); require_1d(by, "y");
    if (bx.size != by.size) throw py::value_error("x and y must have the same length");
    const SimdLevel l = level_or_active(level);
    const double* px = x.data();
    const double* py_ = y.data();
    py::gil_scoped_release nogil;
    return kernels::dot(l, static_cast<std::size_t>(bx.size), px, py_);
  }, py::arg("x"), py::arg("y"), py::arg("level") = py::none());

  m.def("axpy", [](double a, DArr x, DArr y, std::optional<std::string> level) {
    auto bx = x.request(), by = y.request();
    require_1d(bx, "x"); require_1d(by, "y");
    if (bx.size != by.size) throw py::value_error("x and y must have the same length");
    DArr out(by.size);
    std::copy(y.data(), y.data() + by.size, out.mutable_data());
    const SimdLevel l = level_or_active(level);
    kernels::axpy(l, static_cast<std::size_t>(bx.size), a, x.data(), out.mutable_data());
    return out;
  }, py::arg("a"), py::arg("x"), py::arg("y"), py::arg("level") = py::none(),
     "Returns y + a*x (inputs unchanged).");

  m.def("csr_matvec",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr,
           py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> indices,
           DArr data, DArr x, std::optional<std::string> level) {
          auto bp = indptr.request(), bi = indices.request(), bd = data.request(), bx = x.request();
          require_1d(bp, "indptr"); require_1d(bi, "indices"); require_1d(bd, "data"); require_1d(bx, "x");
          if (bp.size < 1) throw py::value_error("indptr must have nrows+1 entries");
          if (bi.size != bd.size) throw py::value_error("indices and data must have equal length");
          const auto* rp = indptr.data();
          const std::size_t nrows = static_cast<std::size_t>(bp.size - 1);
          if (rp[0] != 0 || rp[nrows] != bi.size) throw py::value_error("indptr inconsistent with nnz");
          for (std::size_t r = 0; r < nrows; ++r)
            if (rp[r + 1] < rp[r]) throw py::value_error("indptr must be non-decreasing");
          const auto* ci = indices.data();
          for (py::ssize_t k = 0; k < bi.size; ++k)
            if (ci[k] < 0 || ci[k] >= bx.size) throw py::value_error("column index out of range");
          DArr y(static_cast<py::ssize_t>(nrows));
          const SimdLevel l = level_or_active(level);
          kernels::csr_matvec(l, nrows, rp, ci, data.data(), x.data(), y.mutable_data());
          return y;
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("x"),
        py::arg("level") = py::none(), "y = A x for A in CSR form (scipy-compatible arrays).");

  m.def("linalg_backends", [] {
    std::vector<std::string> out;
    for (auto b : linalg::available_backends()) out.emplace_back(linalg::to_string(b));
    return out;
  });
  m.def("default_linalg_backend", [] { return std::string(linalg::to_string(linalg::default_backend())); });

  m.def("eigh",
        [](const Eigen::MatrixXd& a, std::optional<std::string> backend, bool vectors) {
          const auto b = backend ? linalg::parse_backend(*backend) : linalg::default_backend();
          linalg::EighResult r;
          {
            py::gil_scoped_release nogil;
            r = linalg::eigh(a, b, vectors);
          }
          if (vectors) return py::make_tuple(r.eigenvalues, r.eigenvectors);
          return py::make_tuple(r.eigenvalues, py::none());
        },
        py::arg("a"), py::arg("backend") = py::none(), py::arg("vectors") = true,
        "Symmetric eigendecomposition (lower triangle read). Returns (w, V).");

  m.def("dense_eigh_bytes", &linalg::dense_eigh_bytes, py::arg("n"));
}
