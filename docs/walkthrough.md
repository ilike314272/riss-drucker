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

## M2 Operator library, small group ✅

**What exists**

```
include/riss_drucker/angular.hpp          exact 3j/6j (prime factorization + big integer), 9j, CG, sum rules
include/riss_drucker/models/asym_rotor.hpp  Watson A/S, quartic/sextic, Ir/IIr/IIIr, Wang blocks
include/riss_drucker/model.hpp            + hf_jacobian (Hellmann-Feynman)
python/riss_drucker/spec.py               ModelSpec + canonical SHA-256 hash
python/riss_drucker/cache.py              per-block CSR .npz + manifest; mismatch raises CacheError
python/riss_drucker/rotor.py              J(Ka,Kc) labels from the C2 parity rule
python/riss_drucker/fit.py                LM fit, prior, gauge null directions, Sigma_theta
python/riss_drucker/synth.py              synthetic assigned levels with ground truth
python/riss_drucker/results.py            result record + schema validation
schemas/result.schema.json                result schema (README 9), slots for later milestones
benchmarks/bench_angular.py               the exact-arithmetic decision benchmark
notebooks/02_asymmetric_rotor_fit.ipynb   cache, fit, recovery, gauge, schema
```

**Python**

```python
spec = rd.ModelSpec.asym_rotor(j_max=20, reduction="A", axis="Ir", order=4)
model, info = rd.compile_cached(spec)                     # info["hit"]
obs = rd.synth.synthesize_levels(model, theta_true, noise=1e-4, labels=rd.rotor.asym_top_labels(model))
res = rd.fit(model, obs, theta_start, prior=rd.Prior(mean={...}, sigma={...}), fixed=[...])
res.theta, res.sigma, res.null_directions, res.certified
rd.validate_result(rd.fit_result_record(res, spec, model=model))
```

**M2 exit criteria and evidence**

| Criterion | Evidence |
|---|---|
| Asymmetric rotor, Watson reduction | J = 1, 2 closed forms in all three representations, trace sum rule, symmetric-top limit, representation invariance of labelled levels |
| 3j/6j/9j vs closed forms, exact values, symmetries | `test_angular` (closed forms, permutations, 9j→6j, large j); `test_m2_angular` (exact equality with SymPy for 150 random 3j and 60 6j; 9j to 1e-13) |
| Sum-rule certificates | CG orthogonality and completeness, 6j orthogonality; rotor compile certificates (Hermiticity, total symmetry, time reversal, Wigner-Eckart) abort the compile on failure |
| Cache and hash-invalidation tests | round trip (bitwise-identical operators and levels); structural change → new hash/directory; tampered hash, library version or file → `CacheError`; hash stable across processes |
| Parameter recovery on synthetic data | Watson A quartic, J ≤ 20: pulls within ±4.5σ, χ²/dof ≈ 1; pull distribution over 12 noise seeds has mean ≈ 0, std ≈ 1 |
| Gauge test | redundant `J2` operator: one null direction (1,1,1,−1)/2 found, levels invariant along it, gauge held exactly (w·θ unchanged), invariant combination recovered |
| Exact-arithmetic decision by benchmark | conventions section 8 |

**Follow-ups found in M2** (not blocking)
- Log-magnitude and sign coordinates for parameters spanning decades (README 5.4). Column
  scaling already conditions the problem; these matter once priors span decades (M5).
- Cache `OperatorSum` and the per-block Jacobian operator products inside fitting loops.
- Fits solve every block; solving only the blocks with observations is a simple speedup.
- Map Wang labels to named D2 irreps (A, Ba, Bb, Bc) and nuclear-spin species when a real
  molecule needs spin statistics (M4).
- Isoscalar factors and multiplicity labels: not needed until degenerate groups (M8).

---

## M3 Probabilistic layer ✅

**What exists**

```
python/riss_drucker/spectra.py     line strengths (3j), sum-rule + selection-rule checks, intensities, Q(T)
python/riss_drucker/posterior.py   level windows, near-crossing flags + resampling, Weyl/support fences,
                                   torsion truncation estimate, leave-one-band-out coverage
python/riss_drucker/assign.py      predicted lines with sigma, blends, gating, LR posteriors,
                                   monotone DP per series, ambiguity sets, Poisson null, BH
python/riss_drucker/robust.py      outlier mixture (EM around the fit)
python/riss_drucker/stats.py       beta-Hermite, COE/CUE, Brody (+ fit), gap ratio, superposition P(0)
python/riss_drucker/synth.py       + synthesize_spectrum (selection rules, Boltzmann, noise, blends, spurious)
python/riss_drucker/fit.py         + transitions as observations (E_u - E_l), solves only used blocks
schemas/result.schema.json         + windows, near-crossing resampling, assignments, coverage
notebooks/03_probabilistic_layer.ipynb   spectrum -> assignment -> line fit -> windows -> coverage
```

**M3 exit criteria and evidence**

| Criterion | Evidence |
|---|---|
| Hellmann-Feynman windows | window σ equals the fit's σ_pred for observed levels; support/extrapolation and Weyl flags tested |
| Near-crossing resampling | flagged only when σ/gap > 0.3; only those blocks are re-solved; quantiles reported |
| Joint assignment with intensities | 3 synthetic spectra (pure a-type; mixed a/b/c with 41 spurious peaks): ≥ 99% correct, wrong ≤ expected + 3√expected + 3, few spurious peaks assigned |
| Line strengths | sum rule to 1e-12 in Iʳ/IIʳ/IIIʳ; exact C₂ selection rules; J = 0 strengths = μ_α²; prolate Hönl-London |
| Outlier mixture | 5% injected outliers: all flagged (p > 0.9), > 97% of clean below 0.1, fraction recovered, pulls < 4.5 (plain fit χ²/dof > 10) |
| Coverage checks split by band | correct model: empirical within 4 SE of nominal; quartic fit of sextic data extrapolated to high J: 95% interval covers < 80% |
| Statistics (README 12) | β-Hermite ⟨r⟩ for β = 0, 1, 2, 4 and COE/CUE within 0.012; Brody q recovered; superposition P(0) = 1 − Σf_i² within 0.06 |
| Closed loop | spectrum → assignment → fit to lines recovers all constants (pulls < 5) |

**M3 gaps, closed afterwards** (`tests/python/test_m3_gaps.py`)
- Peak finding + Voigt fitting from raw spectra (`peaks.py`). Matched-filter detection, joint
  Voigt fits per cluster with a local linear baseline, covariance uncertainties. Isolated-line
  pulls ≈ N(0, 1), no spurious peaks at 5σ, doublets resolved down to about 1.5 FWHM.
- Blends in the line fit: a blended peak = intensity-weighted mean of its member transitions,
  with a weighted Hellmann-Feynman Jacobian.
- Self-calibrating assignment: EM estimates p_d, the background fraction and the background
  intensity model from unexplained peaks. Over 16 synthetic spectra: 4 wrong vs 8.6 expected
  (was ~4× conservative), none with posterior > 0.99.
- Model-error hyperparameter τ: `fit(model_error=τ)` and `posterior.calibrate_model_error`
  (ML from cross-fitted held-out residuals). Quartic fit of sextic data: 95% coverage 7% → 99%,
  calibrated on an independent split. A correct model gives τ ≈ 0.

**Still open from M3**
- A scalar τ over-covers when the model error is concentrated (e.g. high K): a structured
  τ(J, K) model belongs with M5's multi-fidelity error model.
- ε_Ω·ρ horizon per block, once perturbative orders are explicit (M8 polyads).
- Truncation estimates for models other than torsion (none needed for the rigid rotor).

---

## M4 Data store and gold set — infrastructure ✅, first real data ✅, gold set ▶

**What exists**

```
schemas/record.schema.json            level | transition | constants; source; provenance; units; convention
python/riss_drucker/data/records.py   unit (MHz/GHz -> cm-1) and 2 sigma -> 1 sigma conversion, schema check,
                                      ehamil_derived refused as fit input by default
python/riss_drucker/data/exomol.py    .def.json fields, .states (filters), streamed .trans (+ S(T) from A, cutoffs)
python/riss_drucker/data/marvel.py    MARVEL transitions/energies readers (n_qn configurable)
python/riss_drucker/data/network.py   (D^T W D) E = D^T W nu per component, anchors, outlier down-weighting,
                                      unconfirmed (single-transition) levels, floating components
python/riss_drucker/data/error_prior.py  calculated-vs-empirical statistics; sigma(J) = sqrt(a^2 + (b J(J+1))^2)
python/riss_drucker/data/roundtrip.py    compile + constants (name map, signs, units) -> reproduce the paper -> quarantine
python/riss_drucker/data/store.py        JSONL record store + quarantine -> fit Observations
scripts/fetch_data.py, scripts/kaggle_datasets.py   provenance-recording downloads; Kaggle via the official CLI
docs/data_sources.md                     survey: H2CO MARVEL (first), H2S MARVEL, ExoMol AYTY/AYT2/APTY/MM,
                                         H2O2 torsion, CH4 MARVEL, CH4+ (Merkt group)
gold/README.md                           extraction protocol; empty so far
notebooks/04_data_layer.ipynb            real files if present, synthetic demonstration otherwise
```

**Evidence** (`tests/python/test_m4_data.py`; all fixtures are synthetic files in the documented formats):
ExoMol read/filter/intensity round trip, including bz2; MARVEL read/write; network inversion
recovers levels with calibrated per-level pulls over 40 noise realizations; injected bad lines
flagged; b-type-only data splits into two components with one floating (ortho/para-like);
error-prior σ(J) recovered; round trip validates correct constants and quarantines a sign flip,
an MHz-as-cm⁻¹ unit error and a ×10 digit error; store → fit recovers constants, with
ehamil_derived refused.

**Real data** (`docs/case_studies/`)
- **H₂CO, ExoMol AYTY** (`scripts/case_h2co.py`): 544 ground-state MARVEL levels (J ≤ 38),
  Watson A octic by order continuation; robust χ²/dof 0.48, certified, full rank. The Ka = 9
  stack is offset (~6σ rms) while Ka = 10 and 11 fit, and octic terms do not remove it, so it
  is a data/network feature (needs the MARVEL H₂CO transitions; the publisher blocks scripted
  access). Calculated-vs-MARVEL error prior from 5,111 states: band-dependent biases up to
  −0.15 cm⁻¹.
- **H₂S, MARVEL** (`scripts/case_h2s.py`): independent inversion of 44,325 transitions
  reproduces all 7,436 published levels (max \|Δ\| ≤ 1.2×10⁻⁴ cm⁻¹, 98% within MARVEL σ).
  Two solver flaws exposed by real data are fixed (precision dynamic range; MARVEL's
  negative-wavenumber deletions). The Watson fit shows the slow convergence expected for a
  light molecule; the new robust-fit guard reports it instead of masking it.
- Octic Watson terms (A and S reductions), `fit_by_order` (order continuation), and
  `fit_robust` warnings (non-converged start, outlier fraction > 0.25).

**Next in M4**
- MARVEL H₂CO transitions (supplement, publisher site) to resolve the Ka = 9 question.
- Gold set: H₂CO microwave/THz constants (Müller & Lewen 2017; CDMS) run through the
  round-trip validator; then the H₂O₂ torsion and CH₄⁺ papers.
- Resummed (Padé/Euler) rotor Hamiltonian for light molecules (H₂S/H₂O).
- Faster `.states`/`.trans` parsing before the 100 AYTY transition files.

---

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
