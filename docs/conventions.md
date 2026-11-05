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

## 14. Open-shell rotor (fixed in M8 work)

- **Basis:** at fixed J the Hund's case (a) product |J P⟩|S Σ⟩ (P, Σ are molecule-fixed
  projections), which is complete: (2J+1)(2S+1) = Σ_N (2N+1). The molecule-fixed J
  obeys anomalous commutation and is represented as (J_x, −J_y, J_z) of the normal
  matrices. S is normal. N = J − S (anomalous). All operators are real in this basis
  (certificate `reality`).
- **Operators** (cm⁻¹, see `models/open_rotor.hpp`):
  - rotor A N_a² + B N_b² + C N_c²;
  - spin-rotation ε_αα N_α S_α (diagonal tensor);
  - spin-spin D_ss (S_z² − S²/3) + E_ss (S_x² − S_y²) in representation axes;
  - Watson A quartic distortion of N;
  - Brown-Sears Δ_N^S N²(N·S);
  - spin-spin distortion ½{N², S_z² − S²/3}.
- **Linear Σ limit:** B = C with A large; ε = γ; D_ss = 2λ; Δ_N^S = γ_D; D_ss_N = 2λ_D.
- **Half-integer J:** a doubled 2J, so odd 2S gives the Kramers / GSE class. The spec's
  group chain records the double group (`D2^(2)`).
- **Compile certificates:** Hermiticity; reality; N² eigenvalues N(N+1) with multiplicity
  2N+1; N·S eigenvalues equal both the 6j-based value and (J(J+1) − N(N+1) − S(S+1))/2.
- **Labels** (`riss_drucker.open_shell.assign`): N from ⟨N²⟩, Ka from ⟨N_a²⟩. F_i is
  numbered from the largest J for given N (F₁: J = N + S). `purity` reports N mixing.

## 15. Large-amplitude motion models (fixed in M8 work)

- **Inversion (`inversion`):** H = G p² + V₂x² + V₄x⁴ + V₆x⁶ in a dimensionless coordinate
  x, plus symmetric-top rotation B(J(J+1) − K²) + C K², plus inversion-rotation coupling
  Bx2·x²(J(J+1) − K²) + Cx2·x²K².
  - Basis: harmonic-oscillator functions of q = x/basis_scale. Polynomial operators are
    built in a basis 8 functions larger, then truncated, so the matrix elements are exact.
  - Blocks: (J, K ≥ 0, parity `+`/`−`). K > 0 levels are doubly degenerate (±K).
- **RAM torsion (`ram_torsion`):** H = A J_a² + B J_b² + C J_c² + D_ab(J_aJ_b + J_bJ_a) + F p²
  + D_pa p J_a + Σ_L V_L(1 − cos Lγ)/2 in |J K⟩|m⟩, with blocks (J, σ = m mod n).
  - Gauge: F(p − ρJ_a)² ⇒ D_pa = −2Fρ, and A absorbs Fρ².
  - With B = C and D_ab = 0 it reproduces `torsion_rotor` exactly. With ρ = 0 and V = 0 it
    equals the principal-axis rotor + F m².

## 16. Physics conventions

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

## 17. Effective vibrational Hamiltonians and ML interfaces (fixed in M7 infrastructure work)

- **Terms are text, never matrices.** A term is `"<param>: <expr>"` in the grammar of
  `rd.grammar`. `n<mode>` is Σ_c a†_c a_c over the mode's components (without the ½).
  Ladder operators act on harmonic-oscillator states, with ⟨n+1|a†|n⟩ = √(n+1). The
  coefficient multiplies the operator exactly as written, so `k: a1+ a2 a2 + h.c.` has
  ⟨0,2|H|1,0⟩ = √2·k.
- **Canonical form.** Normal ordered (creators to the left), with sorted factors and the
  largest coefficient scaled to +1. The canonical string is hashed (SHA-256).
  Two terms with the same hash in one spec are a compile error ("same operator").
- **`sym(...)`** is the group average (1/|G|) Σ_g g·O. If it vanishes, the term is rejected,
  never silently dropped.
- **Degenerate components** use Cartesian real bases: F2 is (x, y, z) and transforms as
  (x, y, z) under Td's signed permutations. E is ((2z² − x² − y²)/√6, (x² − y²)/√2) for
  components (x, y). C2v irreps are A1, A2, B1, B2, with B1 symmetric under σ(xz).
- **Polyads.** P = Σ w_i n_i, with the weights set per mode. Blocks are labelled `irrep=P=<p>`.
  A term that changes P is a compile error.
- **Proposers** return a list of term strings. Lines that do not parse are dropped by the
  proposer. Terms that parse but fail a certificate raise `GrammarError` at compile time.
- **Datasets:** an `npz` shard per `shard_size` samples plus `manifest.json` (canonical spec,
  prior, feature definitions, seed, block keys, library version and commit). Level features are the lowest `n_levels` eigenvalues per
  block, in block order. Moment features are [center, halfwidth, μ_0 … μ_{N−1}] per block,
  with the KPM moments normalized by the dimension (exact) or by R·dim (stochastic).
- **Point sets** (`pointset/1` JSON): energies in hartree as computed. Conversion to cm⁻¹
  uses `HARTREE_TO_CM` = 219474.6313705 at ingestion.

## 18. Dynamic builder, symmetry groups and algebraic certificates (fixed in M7/M8 work)

**Groups (`rd.symmetry`).** A group is the closure of its generators. Each element carries an
equivalent rotation R (proper), its SU(2) lift u, a parity (−1 for E* type), a permutation of
identical nuclei and an affine action γ → sγ + c on each large-amplitude coordinate.
- Operator action: g·J_a = Σ_b R_ba J_b, g·a_c = Σ_c' ρ(g)_c'c a_c', g·f(γ) = f(sγ + c). Hence
  R(gh) = R(g)R(h), ρ(gh) = ρ(g)ρ(h), and h_gh = h_h ∘ h_g for coordinate maps.
- Point groups are given by point operations O. Then R = det(O)·O, parity = det(O), and the
  vector irrep (F2 in Td, E in C3v) is O itself.
- Irreps come from decomposing the regular representation with a random commutant element,
  split over ℂ when the real piece is not absolutely irreducible. They are named from a stored
  character table on the generators; unnamed ones get G<d>_<k>, spinor ones S<d>_<k>.
- Integer results (multiplicities, Frobenius-Schur indicators, spin weights) are rounded from
  floats. A non-integer is an error.
- Double groups (`double=True`) put the SU(2) lift in the element identity. They are needed
  whenever one space has half-integer angular momentum, even if the product is single valued.
- Gauge elements (generator flag `gauge=True`) are redundancies of the coordinates, not
  symmetries. Only irreps trivial on the gauge subgroup are physical blocks.
  - H₂O₂ example: the frame is z = O1→O2 and x = the bisector of the OH projections, with the
    torsion t the full HOOH dihedral on [0, 4π). The gauge element (t → t + 2π, C2z) makes
    K-odd states use antiperiodic torsional functions.
  - In G8(H2O2), swap = C2x with t → t, and E* = C2y with t → −t.

**Spaces and bases (`rd.build`).**
- **Rotation:** |J k⟩, k = −J..J ascending. J_mf = (Jx, −Jy, Jz) of the standard matrices
  (anomalous, section 14). U(g) = D^J(c u c†) with c the lift of C2y, so that
  U J_mf,a U† = Σ_b R_ba J_mf,b.
- **Spin:** |S s⟩ with standard matrices; U = D^S(u).
- **Electronic:** components `<state><x|y|z>` with U = ρ_Γ(g).
- **Vibrational:** occupation states over Cartesian components. U on n quanta is Sym^n(ρ).
  Truncation is by polyad P = Σ w n (conserved, blocks) or by total quanta (`nmax`).
- **Torsion:** e^{i(m/κ)γ}, |m| ≤ m_max (in units of 1/κ), κ = 1 for period 2π and 2 for 4π.
  U|m⟩ = e^{i(m/κ)c}|sm⟩.
- **Oscillator coordinate:** HO basis; U|n⟩ = sⁿ|n⟩.
- **Kronecker order:** rotation first, then spin, then the declared order.

**Time reversal.** Θ = U_Θ K with U_Θ = exp(iπJ_y) and exp(iπS_y) on the rotation and spin
spaces, |m⟩ → |−m⟩ on torsion, and identity elsewhere. Θ includes the space-fixed π rotation,
so Θ² = +1 whenever rotation is present: blocks are real. A spin space without a rotation space
is refused.

**Compilation and certificates.** Terms are canonicalized before checking: cross-space atoms
commute, bosons are normal ordered, and electronic products are reduced. A term must pass
- Hermiticity, Θ-evenness and symmetry under every generator (symbolic, then again on the
  matrices at every J);
- polyad conservation;
- independence of all term matrices over the compiled blocks.

Per J and polyad sector:
- the product-space character must reproduce the sector dimension;
- the projected dimension of each irrep must equal its character count;
- the Θ-fixed real basis must have the block dimension;
- block matrices must be real;
- U must implement the grammar's group action on every atom and form a homomorphism;
- Θ must act on the atoms with the stated parities.

Complex-type irreps are blocked as Γ+Γ* (real, separably degenerate). Quaternionic irreps with
Θ² = +1 are refused. Block keys: `J2=<2J>|irrep=<Γ>|...|sub=P=<p>`.

**Operator dictionary and Molien counts (`rd.algebra`).**
- A multidegree gives, per space:
  - boson degree n;
  - J degree n;
  - torsion (k in units of 1/κ, p power m);
  - oscillator degree;
  - finite spaces (spin, electronic) in full.
- The leading symbol is the commutative image on infinite spaces and the exact matrix on
  finite spaces. The dictionary keeps a maximal independent set of symbols of
  sym(word) + h.c. and sym(i·word) + h.c.
- Its size must equal (D ± T)/2. Here D = 1/|G| Σ_g Π χ_s(g) and T = 1/|G| Σ_g Π τ_s(g), where
  τ_s is the trace of (†∘Θ)∘g. The sign is + for time-reversal-even terms (Hamiltonian) and −
  for odd ones (generators K, with S = iK). The per-space traces are:

| Space | χ_s | τ_s |
|---|---|---|
| bosons | [z⁰] h_n over c and a weighted z^{±w} | h_{n/2}(λ²) for even n |
| J | h_n(R) | (−1)ⁿ h_n(R) |
| spin | \|χ^S\|² | (−1)^{2S} χ^S(u²) |
| electronic | \|χ_el\|² | χ_el(g²) |
| torsion | (1+s) cos(kc) sᵐ, or sᵐ for k = 0 | (−1)ᵐ χ |
| oscillator | (n+1)sⁿ | sⁿ for even n |

**Redundancy.**
- The test is dim(span{O_k} ∩ span{[S_j, H₀]}) at the current parameters. Generators K have
  the requested degrees. H₀ defaults to the terms with at most two atoms per word, at their
  current values.
- The intersection uses floating-point ranks with relative tolerance 10⁻⁹, order by order:
  linearized, not a closed-form normal form.
- Fisher rank ≤ n_terms − redundant is a result certificate.

**Reaction-path Hamiltonian (`rd.abinitio.rph`).**
- The generalized inertia tensor is built from the body-frame coordinates r(γ) on the
  extended path, with Fourier fits of order K in γ/2.
- H = ½ Σ π_i G_ij π_j + V' + V + ΔZPE, with ħ²/(amu Å²) = 33.7152583 cm⁻¹.
- V' = ħ²[−(G_tt L)'/8 + G_tt L²/32], L = (ln det I)'. This is the exact Podolsky
  pseudopotential for a single internal coordinate; the J–p cross terms reduce to
  J_a{G_at, p}/2.
- Each coefficient function is expanded only in the Fourier functions for which the term is
  invariant; the compiler's own symmetry test decides which.
- ΔZPE comes from Hessians projected on translations, rotations and the path tangent.
- Angles are the actual dihedral of each relaxed geometry, not the scan label.
