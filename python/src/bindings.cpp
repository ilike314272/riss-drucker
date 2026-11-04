// pybind11 module riss_drucker._core. Thin layer: validation and conversion
// only; all logic lives in the C++ library.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include "riss_drucker/angular.hpp"
#include "riss_drucker/build_info.hpp"
#include "riss_drucker/cpu_features.hpp"
#include "riss_drucker/kernels.hpp"
#include "riss_drucker/inertia.hpp"
#include "riss_drucker/linalg_backend.hpp"
#include "riss_drucker/model.hpp"
#include "riss_drucker/models/asym_rotor.hpp"
#include "riss_drucker/models/torsion_rotor.hpp"
#include "riss_drucker/solver.hpp"

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

using I64Arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using I32Arr = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

CsrMatrix csr_from_arrays(I64Arr indptr, I32Arr indices, DArr data, std::int64_t ncols) {
  auto bp = indptr.request(), bi = indices.request(), bd = data.request();
  require_1d(bp, "indptr"); require_1d(bi, "indices"); require_1d(bd, "data");
  if (bp.size < 1) throw py::value_error("indptr must have nrows+1 entries");
  if (bi.size != bd.size) throw py::value_error("indices and data must have equal length");
  std::vector<Triplet> t;
  t.reserve(static_cast<std::size_t>(bd.size));
  const auto* rp = indptr.data();
  const std::int64_t nrows = bp.size - 1;
  if (rp[0] != 0 || rp[nrows] != bi.size) throw py::value_error("indptr inconsistent with nnz");
  for (std::int64_t r = 0; r < nrows; ++r) {
    if (rp[r + 1] < rp[r]) throw py::value_error("indptr must be non-decreasing");
    for (auto k = rp[r]; k < rp[r + 1]; ++k) t.push_back({r, indices.data()[k], data.data()[k]});
  }
  return CsrMatrix::from_triplets(nrows, ncols < 0 ? nrows : ncols, std::move(t));
}

py::tuple csr_to_arrays(const CsrMatrix& m) {
  return py::make_tuple(py::array_t<std::int64_t>(m.row_ptr.size(), m.row_ptr.data()),
                        py::array_t<std::int32_t>(m.col_idx.size(), m.col_idx.data()),
                        py::array_t<double>(m.values.size(), m.values.data()),
                        py::make_tuple(m.rows, m.cols));
}

py::dict inertia_dict(const Inertia& in) {
  py::dict d;
  d["n_neg"] = in.n_neg; d["n_zero"] = in.n_zero; d["n_pos"] = in.n_pos;
  d["min_abs_pivot"] = in.min_abs_pivot; d["method"] = in.method;
  return d;
}

py::dict cert_dict(const Certificate& c) {
  py::dict d;
  d["name"] = c.name; d["passed"] = c.passed; d["value"] = c.value; d["bound"] = c.bound;
  d["detail"] = c.detail;
  return d;
}

// Dominant basis function per eigenvector (assignment by max |c|^2).
py::dict result_dict(const BlockResult& r, const std::vector<std::string>* labels, bool keep_vectors) {
  py::dict d;
  d["method"] = r.method;
  d["eigenvalues"] = r.eigenvalues;
  d["residuals"] = r.residuals;
  py::list certs;
  for (const auto& c : r.certificates) certs.append(cert_dict(c));
  d["certificates"] = certs;
  d["certified"] = r.certified;
  d["window"] = py::make_tuple(r.window_lo, r.window_hi);
  d["nev_requested"] = r.nev_requested;
  d["widen_steps"] = r.widen_steps;
  d["boundary_extended"] = r.boundary_extended;
  d["lanczos_runs"] = r.lanczos_runs;
  d["lanczos_mode"] = r.lanczos_mode;
  d["matvecs"] = r.matvecs;
  d["gap_ratio_mean"] = r.gap_ratio_mean;
  py::dict cost;
  cost["dim"] = r.cost.dim; cost["nnz"] = r.cost.nnz; cost["bandwidth"] = r.cost.bandwidth;
  cost["dense_bytes"] = r.cost.dense_bytes; cost["lanczos_bytes"] = r.cost.lanczos_bytes;
  cost["recommended"] = r.cost.recommended; cost["inertia_method"] = r.cost.inertia_method;
  d["cost"] = cost;
  if (r.eigenvectors.size()) {
    py::list dom, weight;
    for (Eigen::Index i = 0; i < r.eigenvectors.cols(); ++i) {
      Eigen::Index arg = 0;
      const double w = r.eigenvectors.col(i).cwiseAbs2().maxCoeff(&arg);
      dom.append(labels && arg < static_cast<Eigen::Index>(labels->size()) ? py::cast((*labels)[arg]) : py::cast(arg));
      weight.append(w);
    }
    d["assignment"] = dom;
    d["assignment_weight"] = weight;
  }
  if (keep_vectors) d["eigenvectors"] = r.eigenvectors;
  return d;
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

  // ---- M1: certificates, solver, torsion-rotor model -----------------------
  py::register_exception<BudgetError>(m, "BudgetError", PyExc_MemoryError);

  m.def("inertia", [](const Eigen::MatrixXd& a, double sigma, std::optional<std::string> backend) {
    const auto b = backend ? linalg::parse_backend(*backend) : linalg::default_backend();
    return inertia_dict(inertia_dense(a, sigma, b));
  }, py::arg("a"), py::arg("sigma"), py::arg("backend") = py::none(),
     "Inertia of (A - sigma I) by Bunch-Kaufman LDL^T (lower triangle read).");
  m.def("sturm_count", [](const Eigen::VectorXd& d, const Eigen::VectorXd& e, double sigma) {
    return inertia_dict(sturm_count(d, e, sigma));
  }, py::arg("diag"), py::arg("offdiag"), py::arg("sigma"));

  py::class_<LanczosOptions>(m, "LanczosOptions")
      .def(py::init<>())
      .def_readwrite("max_basis", &LanczosOptions::max_basis)
      .def_readwrite("max_runs", &LanczosOptions::max_runs)
      .def_readwrite("tol", &LanczosOptions::tol)
      .def_readwrite("seed", &LanczosOptions::seed);

  py::class_<SolveOptions>(m, "SolveOptions")
      .def(py::init<>())
      .def_property("method", [](const SolveOptions& o) { return std::string(to_string(o.method)); },
                    [](SolveOptions& o, const std::string& s) { o.method = parse_solve_method(s); })
      .def_property("backend", [](const SolveOptions& o) { return std::string(linalg::to_string(o.backend)); },
                    [](SolveOptions& o, const std::string& s) { o.backend = linalg::parse_backend(s); })
      .def_readwrite("nlowest", &SolveOptions::nlowest)
      .def_readwrite("window_lo", &SolveOptions::window_lo)
      .def_readwrite("window_hi", &SolveOptions::window_hi)
      .def_readwrite("dense_max_dim", &SolveOptions::dense_max_dim)
      .def_readwrite("inertia_dense_max_dim", &SolveOptions::inertia_dense_max_dim)
      .def_readwrite("memory_budget_bytes", &SolveOptions::memory_budget_bytes)
      .def_readwrite("lanczos", &SolveOptions::lanczos)
      .def_readwrite("residual_tol", &SolveOptions::residual_tol)
      .def_readwrite("orthogonality_tol", &SolveOptions::orthogonality_tol)
      .def_readwrite("max_widen", &SolveOptions::max_widen)
      .def_readwrite("degeneracy_tol", &SolveOptions::degeneracy_tol)
      .def_readwrite("shift_invert", &SolveOptions::shift_invert)
      .def_readwrite("keep_vectors", &SolveOptions::keep_vectors);

  m.def("solve_csr",
        [](I64Arr indptr, I32Arr indices, DArr data, const SolveOptions& opts,
           std::optional<Eigen::MatrixXd> warm) {
          const CsrMatrix h = csr_from_arrays(indptr, indices, data, -1);
          SolveOptions o = opts;
          o.keep_vectors = true;
          BlockResult r;
          {
            py::gil_scoped_release nogil;
            r = solve_block(h, o, warm ? &*warm : nullptr);
          }
          return result_dict(r, nullptr, opts.keep_vectors);
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("options"),
        py::arg("warm") = py::none(), "Certified solve of one symmetric CSR block.");

  py::class_<CompiledModel>(m, "CompiledModel")
      .def_readonly("name", &CompiledModel::name)
      .def_readonly("spec_json", &CompiledModel::spec_json)
      .def_readonly("param_names", &CompiledModel::param_names)
      .def_property_readonly("num_blocks", [](const CompiledModel& cm) { return cm.blocks.size(); })
      .def_property_readonly("compile_certificates", [](const CompiledModel& cm) {
        py::list l;
        for (const auto& c : cm.compile_certificates) l.append(cert_dict(c));
        return l;
      })
      .def_property_readonly("blocks", [](const CompiledModel& cm) {
        py::list l;
        for (const auto& b : cm.blocks) {
          py::dict d;
          d["key"] = b.id.key(); d["two_j"] = b.id.two_j; d["irrep"] = b.id.irrep;
          d["spin_species"] = b.id.spin_species; d["electronic_state"] = b.id.electronic_state;
          d["sub"] = b.id.sub; d["dim"] = b.dim(); d["basis_labels"] = b.basis_labels;
          l.append(d);
        }
        return l;
      })
      .def("operator", [](const CompiledModel& cm, std::size_t block, std::size_t param) {
        return csr_to_arrays(cm.blocks.at(block).ops.at(param));
      }, py::arg("block"), py::arg("param"), "CSR arrays (indptr, indices, data, shape) of O_k in a block.")
      .def("jacobian", [](const CompiledModel& cm, std::size_t block, const Eigen::MatrixXd& vecs) {
        return hf_jacobian(cm.blocks.at(block), vecs);
      }, py::arg("block"), py::arg("eigenvectors"),
         "Hellmann-Feynman Jacobian <psi_i|O_k|psi_i> (rows: levels, columns: parameters).")
      .def_static("from_parts",
          [](std::string name, std::string spec_json, std::vector<std::string> param_names,
             py::list blocks, py::list certs) {
            CompiledModel cm;
            cm.name = std::move(name);
            cm.spec_json = std::move(spec_json);
            cm.param_names = std::move(param_names);
            for (auto item : blocks) {
              py::dict b = item.cast<py::dict>();
              CompiledBlock cb;
              cb.id.two_j = b["two_j"].cast<int>();
              cb.id.irrep = b["irrep"].cast<std::string>();
              cb.id.spin_species = b["spin_species"].cast<std::string>();
              cb.id.electronic_state = b["electronic_state"].cast<int>();
              cb.id.sub = b["sub"].cast<std::string>();
              cb.basis_labels = b["basis_labels"].cast<std::vector<std::string>>();
              for (auto op : b["ops"].cast<py::list>()) {
                py::tuple t = op.cast<py::tuple>();
                auto shape = t[3].cast<py::tuple>();
                CsrMatrix c = csr_from_arrays(t[0].cast<I64Arr>(), t[1].cast<I32Arr>(), t[2].cast<DArr>(),
                                              shape[1].cast<std::int64_t>());
                if (c.rows != shape[0].cast<std::int64_t>()) throw py::value_error("operator shape mismatch");
                cb.ops.push_back(std::move(c));
              }
              if (cb.ops.size() != cm.param_names.size())
                throw py::value_error("block " + cb.id.key() + " has " + std::to_string(cb.ops.size()) +
                                      " operators for " + std::to_string(cm.param_names.size()) + " parameters");
              cm.blocks.push_back(std::move(cb));
            }
            for (auto item : certs) {
              py::dict c = item.cast<py::dict>();
              cm.compile_certificates.push_back({c["name"].cast<std::string>(), c["passed"].cast<bool>(),
                                                 c["value"].cast<double>(), c["bound"].cast<double>(),
                                                 c["detail"].cast<std::string>()});
            }
            return cm;
          },
          py::arg("name"), py::arg("spec_json"), py::arg("param_names"), py::arg("blocks"),
          py::arg("compile_certificates"), "Rebuild a compiled model (used by the operator cache).")
      .def("assemble", [](const CompiledModel& cm, std::size_t block, const std::vector<double>& theta) {
        return csr_to_arrays(assemble(cm.blocks.at(block), theta));
      }, py::arg("block"), py::arg("theta"), "CSR arrays of H(theta) = sum_k t_k O_k in a block.")
      .def("solve",
           [](const CompiledModel& cm, const std::vector<double>& theta, const SolveOptions& opts,
              int threads, std::uint64_t seed, std::optional<std::vector<Eigen::MatrixXd>> warm) {
             SolveOptions o = opts;
             o.keep_vectors = true;  // needed for assignments; dropped below if not requested
             std::vector<BlockResult> res;
             {
               py::gil_scoped_release nogil;
               res = solve_model(cm, theta, o, threads, seed, warm ? &*warm : nullptr);
             }
             py::list out;
             for (std::size_t i = 0; i < res.size(); ++i)
               out.append(result_dict(res[i], &cm.blocks[i].basis_labels, opts.keep_vectors));
             return out;
           },
           py::arg("theta"), py::arg("options"), py::arg("threads") = 1, py::arg("seed") = 0,
           py::arg("warm") = py::none(), "Solve every block (parallel across blocks).");

  // ---- M2: angular algebra, asymmetric rotor, Jacobian, model from parts ----
  auto exact_tuple = [](const angular::ExactValue& v) {
    return py::make_tuple(v.sign, v.num, v.den, v.value);
  };
  m.def("wigner3j_2", &angular::wigner3j, "3j symbol from doubled arguments (2j, 2m).");
  m.def("wigner6j_2", &angular::wigner6j, "6j symbol from doubled arguments.");
  m.def("wigner9j_2", &angular::wigner9j, "9j symbol from doubled arguments.");
  m.def("clebsch_gordan_2", &angular::clebsch_gordan, "<j1 m1 j2 m2|J M> from doubled arguments.");
  m.def("wigner3j_exact_2", [exact_tuple](int a, int b, int c, int d, int e, int f) {
    return exact_tuple(angular::wigner3j_exact(a, b, c, d, e, f));
  }, "(sign, num, den, value): value = sign * sqrt(num/den), num/den as decimal strings.");
  m.def("wigner6j_exact_2", [exact_tuple](int a, int b, int c, int d, int e, int f) {
    return exact_tuple(angular::wigner6j_exact(a, b, c, d, e, f));
  });
  m.def("cg_orthogonality_defect", &angular::cg_orthogonality_defect, py::arg("two_jmax"));
  m.def("cg_completeness_defect", &angular::cg_completeness_defect, py::arg("two_jmax"));
  m.def("sixj_orthogonality_defect", &angular::sixj_orthogonality_defect, py::arg("two_jmax"));

  m.def("compile_asym_rotor",
        [](int j_min, int j_max, std::string reduction, std::string axis, int order,
           std::vector<std::string> extra) {
          models::AsymRotorSpec s;
          s.j_min = j_min; s.j_max = j_max; s.reduction = std::move(reduction);
          s.axis = std::move(axis); s.order = order; s.extra_operators = std::move(extra);
          return models::compile_asym_rotor(s);
        },
        py::arg("j_min") = 0, py::arg("j_max") = 10, py::arg("reduction") = "A", py::arg("axis") = "Ir",
        py::arg("order") = 4, py::arg("extra_operators") = std::vector<std::string>{},
        "Compile the Watson asymmetric rotor (see models/asym_rotor.hpp).");

  m.def("compile_torsion_rotor",
        [](int n_fold, std::vector<int> v_orders, int m_max, int j_max, bool kp_coupling) {
          models::TorsionRotorSpec s;
          s.n_fold = n_fold; s.v_orders = std::move(v_orders); s.m_max = m_max;
          s.j_max = j_max; s.kp_coupling = kp_coupling;
          return models::compile_torsion_rotor(s);
        },
        py::arg("n_fold") = 3, py::arg("v_orders") = std::vector<int>{1}, py::arg("m_max") = 30,
        py::arg("j_max") = 0, py::arg("kp_coupling") = false,
        "Compile the M1 torsion + rigid-rotor model (see models/torsion_rotor.hpp).");
}
