# Conventions

Single source of truth for conventions referenced by `ModelSpec` and the result
schema (README section 10). Each section names the milestone that fixes it; a
section marked *open* must be decided, and recorded here, before that milestone
merges.

## 1. Build and floating point (fixed in M0)

| Rule | Where enforced |
|---|---|
| Baseline flags: `-O3 -march=x86-64-v2 -ffp-contract=off -fno-fast-math` | `cmake/RissDruckerFlags.cmake` |
| `-march=native`, `-ffast-math`, `-Ofast` in any user flags are a configure error | same file (`FATAL_ERROR`) |
| ISA flags only on dispatched kernel TUs: `kernels_avx2.cpp` (`-mavx2 -mfma`), `kernels_avx512.cpp` (`-mavx512f -mavx512dq -mavx2 -mfma`) | `CMakeLists.txt`, `set_source_files_properties` |
| No AVX-class instruction outside those TUs | `scripts/check_portable_isa.py` (CTest `check_portable_isa`) |
| Python module built without LTO (`NO_EXTRAS`) so no kernel code can be inlined across the dispatch boundary | `CMakeLists.txt` |
| SIMD TUs contain no templates, STL, or shared inline functions: plain functions over raw pointers | `src/kernels_impl.hpp` |
| x86-64 only in v0 | `cmake/RissDruckerFlags.cmake` |

`-ffp-contract=off` means the scalar reference is never silently contracted to
FMA. SIMD variants use explicit FMA intrinsics, so they round differently from
the reference by design.

**SIMD tolerance.** A length-n reduction may differ from the scalar reference by
at most `4 (n+1) ε (Σ|x_i y_i| + 1)` (ε = 2.22e-16). Element-wise kernels
(axpy) differ only by the single FMA rounding, `≤ 4e-16 (|y|+|ax|+1)`. These
bounds are what `tests/cpp/test_kernels.cpp` and `tests/python/test_smoke.py`
check. Certificates (Sturm/inertia counts, residual bounds) are always computed
on the scalar/reference path at the final accepted θ (README section 4).

**SIMD selection.** At first use, the core picks the highest level that is both
compiled in and supported by the CPU and OS (`__builtin_cpu_supports`, which
also checks XGETBV). `RISS_DRUCKER_SIMD=scalar|avx2|avx512|auto` caps it; asking
for an unsupported level falls back to the next lower one. The active level goes
into every run record.

## 2. Linear algebra backend (fixed in M0)

- `riss_drucker::linalg` is the single backend layer. Eigen (vendored, MPL2-only
  via `EIGEN_MPL2_ONLY`) is always built and is the default.
- `RISS_DRUCKER_LAPACK=ON` links system LAPACK (reference or OpenBLAS) through
  CMake `FindLAPACK`. `RISS_DRUCKER_MKL=ON` links oneMKL (`lp64`, sequential
  threading) through `MKLConfig.cmake`. The two are mutually exclusive.
- Both Fortran backends call `dsyevd_` (LP64, hidden string-length arguments
  passed). Only the lower triangle is read, on every backend.
- Runtime choice: `RISS_DRUCKER_LINALG=eigen|lapack|mkl`. Asking for a backend
  that was not built is an error, never a silent fallback.
- A wheel built with LAPACK links the image's `liblapack.so.3`. It is not
  manylinux: it is tied to the image it was built on (README 3.1, "Wheel and ABI").

## 3. Threading (fixed in M0, used from M1)

- One OpenMP runtime: GNU `libgomp` (GCC) when `RISS_DRUCKER_OPENMP=ON`. MKL is
  linked sequential, so `libiomp5` never enters the process.
- Block-level parallelism is the default. BLAS gets one thread per block. Set
  `riss_drucker.config.blas_thread_env(1)` in the environment *before* importing
  numpy, scipy, or riss_drucker.
- `RISS_DRUCKER_THREADS` (default 1) is the block-worker count. It is always
  explicit and always recorded.

## 4. Memory, cache, and checkpoints (fixed in M0, used from M1)

- `RISS_DRUCKER_MEM_BUDGET` (for example `8G`). The default is half of
  min(cgroup limit, MemAvailable), measured at runtime and never hardcoded.
  `RuntimeConfig.check_memory()` raises `MemoryError` with a message before any
  allocation that would exceed it. `dense_eigh_bytes(n) = 32 n²` (matrix,
  eigenvectors, workspace).
- `RISS_DRUCKER_CACHE`: defaults to `/kaggle/working/cache` on Kaggle and
  `~/.cache/riss_drucker` elsewhere.
- Every result file is written atomically (`io_utils.atomic_write_*`: temporary
  file, fsync, `os.replace`).

## 5. Sparse layout (fixed in M0)

CSR with `int64` row pointers, `int32` column indices and `float64` values, the
same as scipy's `indptr`/`indices`/`data`. Block dimension must be below 2³¹.

## 6. Provenance (fixed in M0)

Run records (`riss_drucker.run_record`) carry: package version, git commit
(`-dirty` if the tree was modified), compiler and flags, build options, CPU
features, active SIMD level, linalg backend, runtime config (threads, budget,
cache), spec hash, RNG seeds, Python and platform, and optionally the full
`env_probe` report. When the source has no `.git` (for example a Kaggle dataset),
stamp it before upload with `git rev-parse HEAD > GIT_COMMIT`, or set
`RISS_DRUCKER_GIT_COMMIT`.

## 7. Certificates and solver semantics (fixed in M1)

- **Counts are half-open.** `inertia_count` compares the number of returned
  eigenvalues in [lo, hi) with n_neg(H − hi I) − n_neg(H − lo I).
- **Inertia method per block** (`estimate_cost`): Sturm count if the block is
  tridiagonal (bandwidth ≤ 1). Otherwise dense Bunch-Kaufman LDLᵀ, provided
  dim ≤ `inertia_dense_max_dim` (default 6000) and 8·dim² fits the memory
  budget. Otherwise the count is *unavailable*: window solves then refuse to run,
  and lowest-k solves come back uncertified.
- **Bunch-Kaufman:** α = (1+√17)/8, lower storage, symmetric interchanges as in
  LAPACK `dsytf2`. In-house for the Eigen backend, `dsytrf` for lapack/mkl. Inertia
  comes from the signs of the 1×1 and 2×2 pivot eigenvalues (Sylvester).
  `min_abs_pivot` is reported as a diagnostic of how close the shift is to an
  eigenvalue.
- **Widening.** On a count mismatch both finite edges move outward by
  δ_a = 1e3·ε·max(‖H‖∞, 1)·4^a, for attempt a = 0…`max_widen`. The accepted
  window is reported.
- **Lowest-k mode** becomes the window [−∞, hi), with hi at the midpoint of the gap
  above level k. Levels within `degeneracy_tol`·‖H‖∞ (default 1e-10) of level k
  are included, so a multiplet is never split (`boundary_extended`).
- **Lanczos:** full reorthogonalization (two MGS passes with the dispatched
  kernels), locking from the bottom of the spectrum, and fresh deflated restarts
  until the inertia count is met. That is how exact degeneracies are recovered.
- **Shift-invert** is the default on the Lanczos path. Lanczos runs on
  −(H − σI)⁻¹ with σ = Gershgorin lower bound − 1e-6·‖H‖∞. That makes the
  shifted matrix positive definite, so a sparse Cholesky (Eigen
  `SimplicialLLT`) needs no pivoting. Values are mapped back by Rayleigh
  quotients with H. If the factorization fails, Lanczos runs plain.
- **Where certificates are evaluated.** Residuals ‖Hx − λx‖/‖x‖ use the
  scalar reference kernel. The defaults are residual ≤ 1e-8·‖H‖∞ and |VᵀV − I| ≤ 1e-8.
- **Per-block seeds** are `seed XOR FNV-1a(BlockId.key())`, so results do not
  depend on the thread count. The tests check bitwise equality across thread counts.

## 8. Physics conventions

| Item | Status | Milestone |
|---|---|---|
| Energy unit for all internal quantities: cm⁻¹ (MHz converted at ingestion, factor recorded) | fixed | M1 |
| Torsion basis e^{imφ}/√(2π), \|m\| ≤ m_max; V(φ) = Σ_L V_L (1 − cos Lφ)/2 with L = k·n_fold; cosine terms only (φ → −φ symmetric potentials) | fixed | M1 |
| Torsion blocks: σ = m mod n_fold; cos/sin parity combinations (\|m⟩ ± \|−m⟩)/√2 when 2σ ≡ 0 (mod n_fold) and the block has no K·p term. Labels `sigma=s` or `sigma=s,±` | fixed | M1 |
| Reduced rotor: prolate symmetric top A K² + B (J(J+1) − K²), K a good quantum number; torsion-rotation through D_Kp K p only. Gauge: F(p − ρK)² ⇒ D_Kp = −2Fρ, A absorbs Fρ² | fixed (M1 baseline) | M1 |
| BlockId key: `J2=<2J>\|irrep=..\|spin=..\|el=<state>\|sub=k=v;..`. 2J allows half-integer J; `el` is reserved for Phase B | fixed | M1 |
| Rotor basis: symmetric-top \|J K M⟩, Condon-Shortley phases; Wang combinations for asymmetric rotor | *open* | M2 |
| Axis system (Iʳ / IIIʳ, ...) and Watson reduction (A/S), recorded per spec | *open* | M2 |
| 3j/6j/9j phase convention and recursion direction | *open* | M2 |
| Exact angular algebra: GMP / Boost.Multiprecision in C++ vs SymPy/`fractions` in Python, chosen by benchmark | *open* | M2 |
| Isoscalar-factor multiplicity labels (gauge): construction and ordering | *open* | M8 |
| Uncertainty convention: 1σ internally; 2σ sources converted at ingestion | proposed | M4 |
