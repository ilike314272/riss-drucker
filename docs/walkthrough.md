# Build walkthrough

How to build riss-drucker out, milestone by milestone, following the README
(build order **3 → 1 gold set → 4 alongside → 1 automated → 2**). Each
milestone ends with passing tests and a runnable notebook, and its diffs are
scoped to that milestone.

Status legend: ✅ done · ▶ next · ○ planned

---

## M0 Skeleton ✅

**What exists**

```
CMakeLists.txt, cmake/          options, portable flags, build provenance
thirdparty/{eigen,pybind11}/    vendored, each with its own CMake wrapper
include/riss_drucker/           public C++ headers (+ c_api.h)
src/                            cpu features, dispatch, scalar/AVX2/AVX-512 kernels, linalg backend
python/src/bindings.cpp         pybind11 module riss_drucker._core
python/riss_drucker/            config, run records, atomic I/O, env_probe
tests/cpp, tests/python         C++ (CTest) and pytest suites
benchmarks/                     bench_kernels, bench_eigh, baselines/
scripts/                        kaggle_build.sh, env_probe.py, check_portable_isa.py
notebooks/00_smoke_test.ipynb   thin driver
.github/workflows/ci.yml        gcc/clang × eigen/lapack × simd on/off, wheel, fresh install
```

**M0 exit criteria and how each is checked**

| Criterion | Evidence |
|---|---|
| Portable flags verified | `check_portable_isa` CTest disassembles the core library: AVX-class code only in the dispatched TUs. Configure fails on `-march=native` or `-ffast-math`. |
| Scalar and SIMD agree within tolerance | `test_kernels` (all tails, all levels) plus reruns with `RISS_DRUCKER_SIMD` forced to each level; pytest does the same through Python |
| Eigen-only and LAPACK backends build and pass the same tests | CI matrix; `test_linalg` loops over all built backends |
| MKL backend builds (if available) | `-DRISS_DRUCKER_MKL=ON`; not in CI (needs oneMKL, see open item below) |
| Wheel built in one session installs in a fresh one | CI `wheel` → `wheel-fresh-install`; locally `kaggle_build.sh --wheel DIR` |
| Smoke-test notebook | `notebooks/00_smoke_test.ipynb` |

**Local quick start**

```bash
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
```

```bash
scripts/kaggle_build.sh            # probe, build wheel, install, smoke test
```

**Open M0 items**
- Run `env_probe.py` on a real Kaggle session and record its CPU flags. Re-record
  `benchmarks/baselines/` there; the current baseline is from a dev container.
- oneMKL: check license and redistribution terms and offline availability on Kaggle before using it.

---

## M1 Exact reduced baseline ✅

**What exists**

```
include/riss_drucker/block_id.hpp         BlockId (2J, irrep, spin, el, sub) + stable FNV-1a hash
include/riss_drucker/sparse.hpp           CsrMatrix, OperatorSum (fixed pattern, H = sum t_k O_k)
include/riss_drucker/inertia.hpp          Bunch-Kaufman (in-house + dsytrf), Sturm count
include/riss_drucker/lanczos.hpp          full reorth, locking/deflated restarts, LinearOperator
include/riss_drucker/solver.hpp           solve_block: dense | Lanczos (shift-invert), certificates
include/riss_drucker/model.hpp            CompiledModel, solve_model (parallel across blocks)
include/riss_drucker/models/torsion_rotor.hpp   Fourier torsion + symmetric-top rotor (+ K p)
python/riss_drucker/solve.py              rd.solve / rd.solve_csr / Solution (levels, summary)
benchmarks/bench_solvers.cpp              dense / Lanczos / Bunch-Kaufman / Sturm baselines
notebooks/01_torsion_baseline.ipynb       barrier sweep, Mathieu check, certificates, K structure
```

**Python**

```python
model = rd.compile_torsion_rotor(n_fold=3, v_orders=[1, 2], m_max=60, j_max=0)
sol = rd.solve(model, {"F": 5.3, "V3": 400.0, "V6": -20.0}, nlowest=6)   # or window=(lo, hi)
sol.certified, sol.failed_certificates(), sol.levels()[:3], sol.summary()
```

**M1 exit criteria and evidence**

| Criterion | Evidence |
|---|---|
| Dense and Lanczos solves | `test_solver`: both paths match the reference on random banded matrices; `test_m1_torsion::test_dense_and_lanczos_paths_agree` |
| Sturm/inertia certificates | `test_inertia`: BK (both backends) and Sturm agree with eigenvalue counts, including forced 2×2 pivots and singular shifts. Every result carries `inertia_count`, `residual`, `orthogonality` and `hermiticity` |
| A window is accepted only if the counts match; otherwise widen | `test_solver`: edge on an eigenvalue (widened), degenerate boundary (kept whole), missing degenerate copies (extra Lanczos runs forced by the count) |
| Free-rotor limit | `test_torsion` (C++), `test_free_rotor_limit_and_assignments` |
| Harmonic limit | large-barrier Mathieu expansion to O(1/q), all species, tolerance 1e-3·(v+1) cm⁻¹ |
| Exact reference | Mathieu characteristic values for A₁/A₂ (scipy) to 1e-7·V; independent periodic sinc-DVR for all species to 1e-8 |
| Warm start = cold start | C++ and Python tests |
| Failed certificate surfaced | starved Lanczos returns `certified=False` with `lanczos_convergence` failed; no exception |
| Results independent of threads and SIMD level | bitwise equality, 1 vs 4 threads; solver/torsion tests rerun with `RISS_DRUCKER_SIMD=scalar` |

**Benchmarks** (`benchmarks/baselines/m1_dev_container.jsonl`, dev container). For the
pentadiagonal torsion block, dim 2001, lowest 10: dense 9.9 s (Eigen) and Lanczos
0.63 s (shift-invert), of which 0.53 s is the dense Bunch-Kaufman certificate.
Plain Lanczos took 24.6 s on that stiff spectrum, which is why shift-invert is
the default.

**Follow-ups found in M1** (not blocking)
- A banded/sparse inertia count, so large non-tridiagonal blocks can be certified
  without a dense O(n³) factorization (now the dominant cost on the Lanczos path).
  Options: banded Bunch-Kaufman, or a sparse LDLᵀ with symmetric indefinite pivoting.
- Sturm counts vectorized across many shifts (SIMD, structure-of-arrays), as listed in README section 4.
- Cache `OperatorSum` per block for fitting loops (it is rebuilt per call now; cheap at M1 sizes).
- Check which MARVEL/ExoMol data exist for H2O2 before using it as the first
  real torsion test. H2O2 needs `n_fold=1` with V1/V2/V3 terms (unequal cis and trans barriers).

---

## M2 Operator library, small group ▶

- `ModelSpec` with canonical `hash()`: sorted, normalized JSON → SHA-256.
  Stability tests across processes and platforms.
- Asymmetric rotor in the Watson A and S reductions, plus centrifugal distortion
  operators. Blocks by (J, D2 irrep).
- Angular algebra: 3j/6j/9j by stable recursion with closed-form tests. Make the
  exact-arithmetic decision (GMP / Boost.Multiprecision vs SymPy) **by
  benchmark** and record it in `conventions.md`.
- Sum-rule certificates at compile time (orthogonality and completeness,
  Hermiticity, total symmetry, time reversal). Any failure aborts the compile.
- Cache: per-block CSR binary + JSON manifest {spec hash, library version}.
  A mismatch is an error. Atomic writes keyed by (spec hash, block ID).
- Hellmann-Feynman Jacobian (⟨ψ|O_k|ψ⟩ from the certified eigenvectors), then
  LM/Gauss-Newton with a Gaussian prior.
  Parameter recovery and gauge tests on synthetic data (`synth.py` starts here).
- Start `schemas/result.schema.json` here (certificates, θ̂, Σ_θ with gauge directions).

## M3 Probabilistic layer ○
Per-level σ_i² = g_iᵀ Σ_θ g_i, Weyl fence, near-crossing flags and resampling,
χ²-gated assignment with DP alignment, ambiguity sets and BH, outlier mixture,
and coverage checks split by band. β-Hermite / Brody samplers with the ⟨r⟩
tests (0.386 / 0.531 / 0.600).

## M4 Data store and gold set ○
`schemas/record.schema.json` (source ∈ measured|marvel|computed|ehamil_derived;
the fit refuses `ehamil_derived` by default). ExoMol `.states`/`.trans` and MARVEL
parsers with filtering. ~20 hand-extracted gold-set papers (numbers and
citations only). Round-trip validator: quarantine anything that doesn't reproduce.
MARVEL-style (DᵀWD)E = DᵀWν consistency solve. Calculated-vs-MARVEL error prior.
Candidate molecules: H2CO or H2S.

## M5 Multi-fidelity ab initio ○
File-boundary interface (Parquet/JSON), a GP Δ-surface with predictive variance,
propagated level variance, and a model-error hyperparameter. No double counting.

## M6 Automated extraction + analysis widgets ○
An offline LLM extractor that outputs the record schema, evaluated against the gold set.
Part 4 plots (Loomis-Wood, Fortrat, residual-vs-J, corner plots, calibration).
Part 4 widgets can start alongside M2/M3 as soon as there are results to plot.

## M7 Structure proposer ○
Sparse Bayesian selection over the enumerable operator dictionary first, then a
grammar-constrained proposer for compositional choices. Coupling-case diagnostic.

## M8 and later phases ○
Td polyads (methane), open-shell and double groups, LAM PI-group blocks, then
Phases B–F. The interfaces reserved for them so far: an electronic-state slot in
`BlockId`/`ModelSpec` (M1), time-indexed parameters in the run record/result
schema (`TODO(riss-drucker-v1)` in `runinfo.py`), and `RISS_DRUCKER_CUDA` (configure error until then).
