# Header-only wrapper for the vendored Eigen 3.4.0 (see thirdparty/README.md).
add_library(riss_drucker_eigen INTERFACE)
add_library(riss_drucker::eigen ALIAS riss_drucker_eigen)
target_include_directories(riss_drucker_eigen SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/upstream)
# Refuse any LGPL-licensed Eigen code paths.
target_compile_definitions(riss_drucker_eigen INTERFACE EIGEN_MPL2_ONLY)
