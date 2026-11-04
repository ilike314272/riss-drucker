# Wrapper for the vendored pybind11 2.13.6 (see thirdparty/README.md).
# Provides pybind11_add_module() and the pybind11::* targets.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(upstream pybind11)
