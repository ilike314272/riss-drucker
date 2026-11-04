# Compiler flags (README sections 3 and 4).
#  * Portable baseline: -O3 -march=x86-64-v2 (never -march=native).
#  * ISA flags (AVX2/AVX-512) only on the dispatched kernel translation units.
#  * No -ffast-math anywhere. -ffp-contract=off so the scalar reference is not
#    silently FMA-contracted; SIMD kernels use explicit FMA intrinsics.

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  message(FATAL_ERROR "riss-drucker v0 targets x86-64 (got ${CMAKE_SYSTEM_PROCESSOR}).")
endif()

foreach(_var CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  if("${${_var}}" MATCHES "march=native|ffast-math|Ofast")
    message(FATAL_ERROR "${_var} contains a forbidden flag (-march=native, -ffast-math or -Ofast): '${${_var}}'")
  endif()
endforeach()

set(RD_BASELINE_FLAGS -march=x86-64-v2 -ffp-contract=off -fno-fast-math)
set(RD_AVX2_FLAGS   -mavx2 -mfma)
set(RD_AVX512_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
set(RD_WARNING_FLAGS -Wall -Wextra -Wpedantic)

function(rd_apply_baseline_flags target)
  target_compile_options(${target} PRIVATE
    ${RD_BASELINE_FLAGS} ${RD_WARNING_FLAGS} $<$<CONFIG:Release>:-O3>)
endfunction()
