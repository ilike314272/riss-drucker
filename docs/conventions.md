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

## 8. Exact angular algebra (decided in M2, by benchmark)

**Decision:** an in-house C++ exact engine (`include/riss_drucker/angular.hpp`).
Racah sums for 3j and 6j are evaluated with prime-factorized factorials and a
small in-house big integer. The result is exact as sign·√(num/den) and is
rounded to double once. 9j is a long-double sum of products of exact 6j
values. CG coefficients use the Condon-Shortley phase. SymPy is the cross-check
in tests only. No GMP or Boost dependency.

Benchmark (`benchmarks/bench_angular.py`, `benchmarks/baselines/m2_angular_dev_container.jsonl`;
dev container, times include the Python call):

| Candidate | j ≤ 6, per call | j = 40–80, per call | Max rel. error vs SymPy (40 digits) |
|---|---|---|---|
| C++ exact (chosen) | 3.0 µs | 27 µs | 1.1e-16 |
| Python exact (stdlib ints) | 16 µs | 131 µs | 1.4e-16 |
| SymPy | 194 µs | 1.1 ms | 1.1e-16 |
| Naive double Racah | 7.6 µs | 21 µs | 4.0e-2 |

Naive double precision is ruled out by cancellation at large j. SymPy is too slow
for compile loops. The pure-Python exact path is correct but 5× slower than C++
and would put the compile step outside the C++ core.

Sum-rule certificates: CG orthogonality and completeness (2j ≤ 6 at every rotor
compile, 2j ≤ 8 in tests) and 6j orthogonality (tests). Bound 1e-13.

## 9. Asymmetric rotor (fixed in M2)

- **Basis:** symmetric-top \|J K⟩ with ⟨K+1\|J₊\|K⟩ = √(J(J+1) − K(K+1)), real and
  positive. Wang functions (\|K⟩ ± \|−K⟩)/√2 for K > 0, with \|0⟩ in "+".
  Blocks per J: `E+`, `E-`, `O+`, `O-` (K even/odd, ±), so `irrep` is the Wang label.
- **Representations:** (x, y, z) = Iʳ (b, c, a), IIʳ (c, a, b), IIIʳ (a, b, c). The
  rigid term is A J_a² + B J_b² + C J_c².
- **Reductions and signs:** exactly as in `models/asym_rotor.hpp` (Watson A and S,
  quartic and sextic, H = Σ t_k O_k with the conventional signs, e.g.
  O_DeltaJ = −J⁴, O_d1 = +J²(J₊² + J₋²)). Units are cm⁻¹. A literature constant in MHz
  is converted at ingestion (M4).
- **Labels J(Ka,Kc):** a block's parity class follows from C₂ about z, (−1)^K, and
  about x, γ(−1)^J. In the principal axes these are (−1)^Ka for a, (−1)^(Ka+Kc) for b,
  (−1)^Kc for c. The n-th level of a block (ascending energy) gets the n-th label of
  its class in increasing τ = Ka − Kc. This is tested against a global τ sort in all
  three representations.
- **Compile certificates** (failure aborts the compile): Hermiticity; total symmetry
  (no coupling between Wang blocks); time reversal (even degree in J); Wigner-Eckart
  check of the ladder elements against the 3j engine; CG orthogonality and completeness.

## 10. ModelSpec, hash and operator cache (fixed in M2)

- `ModelSpec.hash()` = SHA-256 of the canonical JSON: sorted keys, no whitespace,
  ASCII, integral floats as ints, other floats as `repr`. The JSON includes kind,
  structure, group chain, electronic states (reserved, Phase B), the conventions
  reference and `OPERATOR_LIBRARY_VERSION`. Bump that version whenever an operator
  definition, sign or phase changes.
- Cache: `<RISS_DRUCKER_CACHE>/operators/<hash>/manifest.json` plus `block_NNNNN.npz`
  (CSR arrays per operator), with a per-file SHA-256 in the manifest, written
  atomically and the manifest last. A mismatch in spec, hash, library version, format
  or checksum raises `CacheError`. Nothing is silently recompiled; `refresh=True`
  is the explicit override.

## 11. Fit and gauge (fixed in M2)

- Residuals (E_obs − E_calc)/σ. Jacobian by Hellmann-Feynman, ⟨ψ|O_k|ψ⟩ on the
  scalar reference path. An independent Gaussian prior adds rows (μ − θ)/s.
- Column scaling D (Jacobian column norms) is frozen at the starting point. The SVD of
  J·D⁻¹ gives rank r, using singular values > 1e-9·s_max. Levenberg-Marquardt steps
  and Σ_θ = D⁻¹ V_r S_r⁻² V_rᵀ D⁻¹ live in the identifiable subspace.
- **Null directions** are reported in parameter space (normalized, largest component
  positive). Parameters with a component > 1e-6 on any null direction get σ = NaN.
  The gauge is fixed exactly: w·θ keeps its starting value, with w = D·v_null
  (`gauge_constraints`).
- Σ_θ is not rescaled by χ²/dof unless `scale_cov_by_chi2=True`. Pulls on synthetic
  data are tested to be N(0, 1).
- σ_pred per level = √(gᵀ Σ_θ g), using the Hellmann-Feynman g of that level.

## 12. Probabilistic layer (fixed in M3)

- **Windows:** σ_i² = g_iᵀ Σ_θ g_i, using the Hellmann-Feynman g on the reference
  path; window E_i ± kσ_i, default k = 2.
- **Near crossing:** σ_i / (nearest gap in the same block) > 0.3. Only blocks
  holding flagged levels are re-solved at θ ~ N(θ̂, Σ_θ). Null directions are not
  moved: samples use the eigendecomposition of Σ_θ.
- **Weyl fence (interpretation):** a window is "beyond the Weyl fence" when
  kσ_i > ½ × the local mean level spacing. That spacing comes from a smoothed
  counting function N(E) of the block (cubic fit; linear below 8 levels). Past it,
  a level's identity is not individually trustworthy (the ε·ρ horizon idea).
  Separately, `extrapolated` marks levels outside the data-supported J range or
  above the highest observed energy plus 5·max σ_obs.
- **Truncation:** for torsion, a second-order estimate Σ_n \|⟨n\|V\|ψ_i⟩\|²/(E_i − E_n⁰)
  over neglected \|m\| > m_max. It is an estimate, not a rigorous bound; the tests
  check it tracks the actual shift within ×2–5. The rotor needs none (J is exact).
- **Line strengths:** S = (2J+1)(2J'+1)\|Σ μ_q c' c (−1)^K' (J' 1 J; −K' q K)\|²,
  with μ_±1 = ∓(μ_x ± iμ_y)/√2 in the representation axes and S in units of
  \|μ\|². Certified by the sum rule Σ_f S = (2J+1)\|μ\|² (1e-12) and by exact C₂
  selection rules (a: Δ(Ka,Kc) parity (0,1), b: (1,1), c: (1,0)). Intensities are
  relative absorption with c₂ = 1.438776877 cm K. Q comes from the model's own
  levels, so it is truncated at j_max.
- **Blends:** predicted or true lines closer than the resolution are merged (chained)
  into one line with an intensity-weighted position, summed intensity and id `a+b`.
  The synthetic resolution is fwhm/2.
- **Assignment:** gate z² < 10.83 (χ²₁, 0.999). LR = position likelihood / ρ_j × intensity
  log-normal likelihood / background log-intensity density. Global intensity scale =
  median log ratio over mutually unique pairs. Detection probability p_d = 0.9. The
  credible (ambiguity) set covers 0.95. Monotone DP is per series (ΔJ, Ka″ → Ka′) in
  predicted-position order; conflicts across series go to the higher posterior and are
  flagged. Chance coincidence uses 1 − exp(−2δρ) with ρ from ±25 median spacings;
  Benjamini-Hochberg at q = 0.05 runs across the band.
- **Outlier mixture:** (1 − w)N(0, σ_i²) + wN(0, σ_b²) with σ_b = 100 × median σ by
  default. EM uses effective σ_i/√(responsibility). Probabilities are always reported;
  observations are never dropped.
- **Coverage:** leave-one-band-out, z = (E_obs − E_pred)/√(σ_pred² + σ_obs²), with
  empirical vs nominal fractions and binomial standard errors.
- **⟨r⟩ references:** Poisson 2 ln 2 − 1, GOE 0.5307, GUE 0.5996, GSE 0.6744.

## 13. Data layer (fixed in M4)

- **Units:** records keep the paper's units and uncertainty definition. Conversion
  happens once, at use: 1 cm⁻¹ = 29979.2458 MHz; 2σ → 1σ by halving. An
  `unknown` definition cannot be used in a fit.
- **Effective-Hamiltonian-derived data** (HITRAN positions, values computed from a
  paper's own Hamiltonian) carry `source: ehamil_derived` and are refused as fit input
  unless `allow_ehamil_derived=True`.
- **Network inversion:** weights 1/σ². Each component is anchored at the given level
  (MARVEL ground state at 0) or, if none is given, at its first level; such energies are
  reported as floating.
  - *Outlier detection:* Huber IRLS on capped precision, σ_eff = max(σ, median σ).
  - *Clean solves:* full-precision solves on the clean set, re-judging every line against
    \|r\|/σ > 5 (worst first; lines that fit again are un-flagged).
  - Lines with a negative wavenumber in a MARVEL file are MARVEL deletions and are ignored.
  - A level with one clean transition is unconfirmed. A level reachable only through flagged
    lines keeps its robust estimate, marked `isolated_by_outliers`, with no uncertainty.
  - Outlier indices refer to the caller's input list.
- **Octic Watson terms:** + sign for all octic terms, like the sextic terms
  (`models/asym_rotor.hpp`).
- **Order continuation and the robust-fit guard:** fit high orders by continuation
  (`fit_by_order`). `fit_robust` warns, and is not `trustworthy`, if its starting fit did
  not converge or if the outlier fraction is > 0.25.
- **Round trip:** a constants record is validated when its constants reproduce the
  paper's lines/levels with rms ≤ 2 × (paper rms, or median reference σ) and no
  residual > 5 × that scale. Otherwise it is quarantined with the reason. The paper's
  sign and name conventions are mapped through `convention.name_map`.

## 14. Physics conventions

| Item | Status | Milestone |
|---|---|---|
| Energy unit for all internal quantities: cm⁻¹ (MHz converted at ingestion, factor recorded) | fixed | M1 |
| Torsion basis e^{imφ}/√(2π), \|m\| ≤ m_max; V(φ) = Σ_L V_L (1 − cos Lφ)/2 with L = k·n_fold; cosine terms only (φ → −φ symmetric potentials) | fixed | M1 |
| Torsion blocks: σ = m mod n_fold; cos/sin parity combinations (\|m⟩ ± \|−m⟩)/√2 when 2σ ≡ 0 (mod n_fold) and the block has no K·p term. Labels `sigma=s` or `sigma=s,±` | fixed | M1 |
| Reduced rotor in the torsion model: prolate symmetric top A K² + B (J(J+1) − K²), K a good quantum number; torsion-rotation through D_Kp K p only. Gauge: F(p − ρK)² ⇒ D_Kp = −2Fρ, A absorbs Fρ² | fixed (M1 baseline) | M1 |
| BlockId key: `J2=<2J>\|irrep=..\|spin=..\|el=<state>\|sub=k=v;..`. 2J allows half-integer J; `el` is reserved for Phase B | fixed | M1 |
| Rotor basis, Wang blocks, representation, Watson A/S operators, J(Ka,Kc) labels | fixed (section 9) | M2 |
| 3j/6j/9j: exact Racah sums (no recursion needed at this precision), Condon-Shortley phase | fixed (section 8) | M2 |
| Exact angular algebra: in-house C++ (benchmark above) | fixed | M2 |
| Isoscalar-factor multiplicity labels (gauge): construction and ordering | *open* | M8 |
| Uncertainty convention: 1σ internally; 2σ sources converted at ingestion | proposed | M4 |
