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

## M1 Exact reduced baseline ▶

Goal: a torsion + rigid-rotor Hamiltonian with every solver path and
certificate in place, checked against analytic and exact references.

**C++ components**
1. `BlockId` (J, irrep, spin species, and a reserved electronic-state slot for
   Phase B) plus a canonical string form. It goes into cache keys and the result schema.
2. `SparseOperator` (CSR, uses the M0 kernels) and `Hamiltonian = Σ t_k O_k`
   assembled as a sparse sum with the sparsity pattern fixed per block.
3. Torsion block: Fourier basis e^{imφ}, `F p²` diagonal, `V_n/2 (1 − cos nφ)`
   couples m ↔ m±n. Blocks by m mod n (for n = 3: A and E species).
4. Rigid rotor: symmetric top `B J(J+1) + (A−B) K²` (diagonal; the asymmetric
   rotor comes in M2).
5. Solvers: dense `eigh` (M0 backend layer) and Lanczos with full
   reorthogonalization, a window [σ_lo, σ_hi], and a warm start.
6. Certificates:
   - Sturm count for the symmetric tridiagonal/banded case (vectorize across
     shifts later; scalar reference first).
   - LDLᵀ inertia of (H − σI) for general blocks. *Decision:* Bunch-Kaufman
     (`dsytrf` on LAPACK, in-house for Eigen). A plain LDLᵀ without pivoting
     can break down on indefinite matrices.
   - Residual bound ‖Hx − λx‖/‖x‖.
   - Window acceptance: inertia count == number returned; otherwise widen and rerun.
7. Per-block cost estimate (dimension, bandwidth, ⟨r⟩) feeding the
   dense-vs-Lanczos choice and `RuntimeConfig.check_memory`.

**Tests to write first** (README section 12)
- Free rotor (V = 0): E = F m², twofold degeneracy for m ≠ 0.
- Harmonic limit (V ≫ F): low levels → n√(F V)(v + ½) plus anharmonic correction.
- Exact reference across all regimes: Mathieu characteristic values (the V₃-only torsion problem maps onto the Mathieu equation).
- Lanczos eigenvalues equal dense within tolerance. Inertia count equals the returned count. Warm start equals cold start.
- A forced failed certificate shows up in the result, never absorbed silently.

**Notebook:** `01_torsion_baseline.ipynb`, levels vs barrier height
from free rotor to harmonic, using an H2O2-like F and V. Check which MARVEL/ExoMol
data exist for H2O2 before going further.

---

## M2 Operator library, small group ○

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
- Hellmann-Feynman Jacobian, then LM/Gauss-Newton with a Gaussian prior.
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
