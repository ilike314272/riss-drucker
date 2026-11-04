foreach(t test_kernels test_cpu_dispatch test_linalg test_build_info
          test_sparse test_inertia test_solver test_torsion test_angular test_asym_rotor)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE riss_drucker::core)
  rd_apply_baseline_flags(${t})
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# Force each SIMD level through the env override and rerun the kernel tests.
foreach(lvl scalar avx2 avx512)
  add_test(NAME test_kernels_env_${lvl} COMMAND test_kernels)
  set_tests_properties(test_kernels_env_${lvl} PROPERTIES ENVIRONMENT "RISS_DRUCKER_SIMD=${lvl}")
endforeach()

# Solver and model results must not depend on the SIMD level either.
foreach(t test_solver test_torsion)
  add_test(NAME ${t}_env_scalar COMMAND ${t})
  set_tests_properties(${t}_env_scalar PROPERTIES ENVIRONMENT "RISS_DRUCKER_SIMD=scalar")
endforeach()

# Portable-flag verification: no AVX/AVX-512 instructions outside the
# dispatched kernel translation units.
find_package(Python COMPONENTS Interpreter QUIET)
find_program(RD_OBJDUMP objdump)
if(Python_Interpreter_FOUND AND RD_OBJDUMP)
  add_test(NAME check_portable_isa
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_portable_isa.py
            --objdump ${RD_OBJDUMP} $<TARGET_FILE:riss_drucker_core>)
endif()
