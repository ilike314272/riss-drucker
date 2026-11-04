cmake_minimum_required(VERSION 3.18)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" RD_VERSION LIMIT_COUNT 1)
project(riss_drucker VERSION ${RD_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options (README section 4) -------------------------------------------
option(RISS_DRUCKER_BUILD_TESTS   "Build C++ tests"                          ON)
option(RISS_DRUCKER_BUILD_BENCH   "Build benchmark targets"                  ON)
option(RISS_DRUCKER_PY            "Build the pybind11 module"                ON)
option(RISS_DRUCKER_SIMD_DISPATCH "Build AVX2/AVX-512 kernels, pick at runtime" ON)
option(RISS_DRUCKER_OPENMP        "Enable OpenMP (block-level parallelism)"  OFF)
option(RISS_DRUCKER_MKL           "Enable the oneMKL linalg backend"         OFF)
option(RISS_DRUCKER_LAPACK        "Enable the LAPACK/OpenBLAS linalg backend" OFF)
option(RISS_DRUCKER_CUDA          "CUDA paths (not v0; must stay OFF)"       OFF)

if(RISS_DRUCKER_CUDA)
  # TODO(riss-drucker-v1): GPU paths (README 3.1). Not part of v0.
  message(FATAL_ERROR "RISS_DRUCKER_CUDA is reserved for a later phase; nothing in v0 uses a GPU.")
endif()
if(RISS_DRUCKER_MKL AND RISS_DRUCKER_LAPACK)
  message(FATAL_ERROR "Pick one of RISS_DRUCKER_MKL / RISS_DRUCKER_LAPACK: mixing BLAS/OpenMP runtimes is unsafe.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(RissDruckerFlags)

add_subdirectory(thirdparty/eigen)

# ---- Core library ---------------------------------------------------------
set(RD_CORE_SOURCES
  src/build_info.cpp
  src/cpu_features.cpp
  src/kernels_scalar.cpp
  src/kernels_dispatch.cpp
  src/linalg_backend.cpp
  src/c_api.cpp
  src/block_id.cpp
  src/sparse.cpp
  src/inertia.cpp
  src/lanczos.cpp
  src/solver.cpp
  src/model.cpp
  src/models/torsion_rotor.cpp
)
add_library(riss_drucker_core STATIC ${RD_CORE_SOURCES})
add_library(riss_drucker::core ALIAS riss_drucker_core)
target_include_directories(riss_drucker_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
find_package(Threads REQUIRED)
target_link_libraries(riss_drucker_core PUBLIC riss_drucker::eigen Threads::Threads)
rd_apply_baseline_flags(riss_drucker_core)

# Runtime-dispatched SIMD kernels: ISA flags per translation unit only.
if(RISS_DRUCKER_SIMD_DISPATCH)
  target_sources(riss_drucker_core PRIVATE src/kernels_avx2.cpp src/kernels_avx512.cpp)
  set_source_files_properties(src/kernels_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "${RD_AVX2_FLAGS}")
  set_source_files_properties(src/kernels_avx512.cpp PROPERTIES
    COMPILE_OPTIONS "${RD_AVX512_FLAGS}")
  target_compile_definitions(riss_drucker_core PUBLIC RISS_DRUCKER_HAVE_SIMD_DISPATCH=1)
endif()

# Linear-algebra backends (Eigen is always available and is the default).
set(RD_LINALG_BACKENDS "eigen")
if(RISS_DRUCKER_LAPACK)
  find_package(LAPACK REQUIRED)
  target_link_libraries(riss_drucker_core PUBLIC LAPACK::LAPACK)
  target_compile_definitions(riss_drucker_core PUBLIC RISS_DRUCKER_HAVE_LAPACK=1)
  list(APPEND RD_LINALG_BACKENDS "lapack")
endif()
if(RISS_DRUCKER_MKL)
  # oneMKL ships its own CMake config (MKLConfig.cmake). Check license and
  # offline availability on Kaggle before relying on it (README section 4).
  set(MKL_THREADING sequential CACHE STRING "")
  set(MKL_INTERFACE lp64 CACHE STRING "")
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(riss_drucker_core PUBLIC MKL::MKL)
  target_compile_definitions(riss_drucker_core PUBLIC RISS_DRUCKER_HAVE_MKL=1)
  list(APPEND RD_LINALG_BACKENDS "mkl")
endif()
if(RISS_DRUCKER_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(riss_drucker_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(riss_drucker_core PUBLIC RISS_DRUCKER_HAVE_OPENMP=1)
endif()

# Build provenance baked into the library (README section 3: reproducibility).
include(RissDruckerBuildInfo)
rd_generate_build_info()

# ---- Python module ----------------------------------------------------------
if(RISS_DRUCKER_PY)
  add_subdirectory(thirdparty/pybind11)
  # NO_EXTRAS: no LTO, so code from the ISA-flagged kernel TUs can never be
  # inlined across the dispatch boundary into baseline code.
  pybind11_add_module(_core MODULE NO_EXTRAS python/src/bindings.cpp)
  target_link_libraries(_core PRIVATE riss_drucker::core)
  rd_apply_baseline_flags(_core)
  if(DEFINED RD_PY_OUTPUT_DIR)
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${RD_PY_OUTPUT_DIR})
  endif()
endif()

# ---- Tests and benchmarks -------------------------------------------------
if(RISS_DRUCKER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
if(RISS_DRUCKER_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()
