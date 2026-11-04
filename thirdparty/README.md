# Vendored dependencies

Every dependency lives here, one directory each, with its own `CMakeLists.txt`
wrapper. Nothing is fetched from the network at build time (README section 3).
Upstream sources sit unmodified under `<dep>/upstream/`, trimmed to what the
build needs. Platform patches, if ever needed, go in `<dep>/patches/`.

| Dependency | Version | Upstream commit | License | Trimmed to | CMake target |
|---|---|---|---|---|---|
| Eigen | 3.4.0 | 3147391d946bb4b6c68edd901f2add6ac1f31f8c | MPL2 (built with `EIGEN_MPL2_ONLY`) | `Eigen/`, license files | `riss_drucker::eigen` |
| pybind11 | 2.13.6 | a2e59f0e7065404b44dfe92a28aca47ba1378dc4 | BSD-3-Clause | `include/`, `tools/`, `pybind11/`, `CMakeLists.txt`, license | `pybind11::module` (via `pybind11_add_module`) |

Optional system backends (not vendored, behind CMake options, see
`docs/conventions.md`): LAPACK/OpenBLAS (`RISS_DRUCKER_LAPACK`), oneMKL
(`RISS_DRUCKER_MKL`).

To update a dependency: replace `upstream/`, update this table, rebuild, and
run the full test suite on both linalg backends.
