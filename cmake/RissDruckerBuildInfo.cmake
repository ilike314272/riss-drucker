# Generates generated/riss_drucker/build_config.hpp with provenance fields that
# every run record carries (git commit, compiler, flags, options).

function(rd_generate_build_info)
  set(RD_GIT_COMMIT "unknown")
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE RD_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_dirty)
      set(RD_GIT_COMMIT "${RD_GIT_COMMIT}-dirty")
    endif()
  endif()
  # Source shipped without .git (e.g. a Kaggle dataset): use a stamped file,
  # created with `git rev-parse HEAD > GIT_COMMIT` before uploading.
  if(RD_GIT_COMMIT STREQUAL "unknown" AND EXISTS "${CMAKE_SOURCE_DIR}/GIT_COMMIT")
    file(STRINGS "${CMAKE_SOURCE_DIR}/GIT_COMMIT" RD_GIT_COMMIT LIMIT_COUNT 1)
  endif()
  if(DEFINED ENV{RISS_DRUCKER_GIT_COMMIT})
    set(RD_GIT_COMMIT "$ENV{RISS_DRUCKER_GIT_COMMIT}")
  endif()

  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
  string(REPLACE ";" " " _base "${RD_BASELINE_FLAGS}")
  string(REPLACE ";" " " _avx2 "${RD_AVX2_FLAGS}")
  string(REPLACE ";" " " _avx512 "${RD_AVX512_FLAGS}")
  set(RD_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} ${_base}")
  string(STRIP "${RD_CXX_FLAGS_STR}" RD_CXX_FLAGS_STR)
  set(RD_AVX2_FLAGS_STR "${_avx2}")
  set(RD_AVX512_FLAGS_STR "${_avx512}")
  string(REPLACE ";" "," RD_LINALG_BACKENDS_STR "${RD_LINALG_BACKENDS}")
  foreach(_opt SIMD_DISPATCH OPENMP MKL LAPACK)
    if(RISS_DRUCKER_${_opt})
      set(RD_OPT_${_opt} 1)
    else()
      set(RD_OPT_${_opt} 0)
    endif()
  endforeach()
  configure_file(${CMAKE_SOURCE_DIR}/cmake/build_config.hpp.in
                 ${CMAKE_BINARY_DIR}/generated/riss_drucker/build_config.hpp @ONLY)
endfunction()
