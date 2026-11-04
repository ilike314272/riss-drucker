add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE riss_drucker::core)
rd_apply_baseline_flags(bench_kernels)

add_executable(bench_eigh bench_eigh.cpp)
target_link_libraries(bench_eigh PRIVATE riss_drucker::core)
rd_apply_baseline_flags(bench_eigh)

add_executable(bench_solvers bench_solvers.cpp)
target_link_libraries(bench_solvers PRIVATE riss_drucker::core)
rd_apply_baseline_flags(bench_solvers)
