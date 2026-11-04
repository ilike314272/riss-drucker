# riss-drucker

"Crack printer" for amortized inferencing for fundamental physical details of general polyatomic spectroscopic problems.

Pipeline from molecular spectra and ab initio calculations to a **quantum, time-independent effective Hamiltonian** with calibrated uncertainties. C++ numerics, Python orchestration, runnable on Kaggle notebooks.

> Instructions for Claude Code: read this file fully before writing code. Follow the working agreements in section 10 and respect the scope in section 1.

**Status:** M0 (skeleton), M1 (torsion + rigid-rotor baseline with certified dense/Lanczos solves) and M2 (exact angular algebra, Watson asymmetric rotor, ModelSpec hash and operator cache, Hellmann-Feynman fit with gauge detection) and M3 (windows, near-crossing resampling, line strengths, joint assignment with BH control, outlier mixture, coverage by band, level statistics, plus peak finding/Voigt fits, blends, EM-calibrated posteriors and a model-error hyperparameter) complete. M4 data-layer infrastructure is in place; real data ingestion is waiting on network access to the data hosts (see `docs/data_sources.md`). See [`docs/walkthrough.md`](docs/walkthrough.md) for the milestone-by-milestone build guide and [`docs/conventions.md`](docs/conventions.md) for conventions. Quick start: `scripts/kaggle_build.sh` (probe, build wheel offline, install, smoke test) or `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build`.

## 1. Scope

**In scope (v0):** H(θ) = Σ_k t_k O_k built from a fixed, symmetry-adapted operator set; block diagonalization per (J, irrep); Bayesian fit to level/line data; levels, mixing, assignments and calibrated windows as outputs; literature/data ingestion with provenance; analysis and visualization.

**Deferred to later phases (planned, not v0; do not build yet, but do not design them out):** time dependence and classical/positional degrees of freedom, MD and QM/MM partitioning, ergodicity/recurrence diagnostics, Jahn-Teller and multi-surface vibronic models (including CH4+), MQDT/Rydberg states, an in-house correlated ab initio engine (use existing codes through a file boundary for now). See section 11 for the phase plan. Keep the interfaces (ModelSpec, block IDs, result schema) general enough that these can be added without breaking the cache or schemas, e.g. leave room in ModelSpec for extra electronic states and in the result schema for time-indexed parameters.

**Design rules**
- Formulae are deterministic and exact. Numbers feeding them are distributions.
- Angular-momentum algebra (3j/6j/9j, isoscalar factors, O_k matrices) is computed **once per model specification** and cached. A Bayesian update changes θ only, never the O_k.
- Certificates (inertia counts, residual bounds, sum rules, round trips) are deterministic code. Probabilistic statements (windows, assignment confidence) sit on top and are reported with calibration status.
- ML proposes, physics verifies. No learned component is the source of record for a physical quantity.

## 2. The four parts

| Part | Name | Language | Role |
|---|---|---|---|
| 1 | Data: calculate, gather, scan | Python (+ LLM extractor) | Databases, papers, ab initio point sets into a validated, provenance-tagged store |
| 2 | Structure proposer | Python (LLM + tools) | Proposes effective-Hamiltonian structure (operators, polyads, coupling scheme); scored by fitting, not by token likelihood |
| 3 | Core numerics | C++ (+ Python bindings) | Operator library, solver, certificates, Jacobians, symbolic/numerical angular algebra |
| 4 | Analysis and visualization | Python | Spectroscopic plots, diagnostics, interactive notebooks |

Build order: **3 → 1 (hand-built gold set) → 4 alongside → 1 (automated extraction) → 2.** Part 3's round-trip validator is what makes automated extraction trustworthy, and Part 2 needs both the operator library and validated data.

## 3. Environment constraints (Kaggle)

- Notebooks are thin drivers. All logic lives in the library, installed as a wheel or built in-session.
- Internet may be off. Vendor every dependency under `thirdparty/`: one directory per dependency, each with its own `CMakeLists.txt` wrapper, platform patches in a subdirectory only if needed. No network fetches at build time.
- Inputs are read-only under `/kaggle/input`. Write only to `/kaggle/working`. Cache dir is configurable (`RISS_DRUCKER_CACHE`, default `/kaggle/working/cache`).
- Hardware varies between sessions: never build with `-march=native`. Build portable code at `-O3 -march=x86-64-v2` and add AVX2/AVX-512 kernels through runtime dispatch (section 4), so a wheel built in one session runs in another and still uses Intel SIMD where the CPU has it.
- Verify current Kaggle CPU/RAM/disk/session limits before sizing anything; do not hardcode them. Thread count (`RISS_DRUCKER_THREADS`) and memory budget are explicit parameters.
- Reproducibility: every run records model-spec hash, git commit, compiler flags, thread count and RNG seeds.

### 3.1 Compute model (CPU-first)

- **The core is CPU code.** Operator compilation, block diagonalization, certificates, Jacobians and fitting are CPU calculations in double precision (FP64). Kaggle notebooks may offer a GPU, but nothing in v0 depends on one, and every result must be reproducible on CPU alone.
- **Where a GPU could help later (behind `RISS_DRUCKER_CUDA`, off by default, not v0):** dense `eigh` on large blocks, batched small-block solves, matvecs for Lanczos and stochastic trace/moment estimation, and training of the Part 2 proposer and ab initio Δ-models. Verify the FP64 throughput of whatever GPU Kaggle assigns before porting anything, since many GPUs have very weak FP64. Mixed precision is acceptable for ML training, never for the Hamiltonian core or certificates.
- **Parallelism:** blocks (J, irrep, spin species) are independent, so parallelize across blocks (OpenMP or a thread pool), not inside small blocks. Prevent oversubscription: when running block-level parallelism, force the BLAS/LAPACK backend to one thread per block (and set it explicitly via `RISS_DRUCKER_THREADS`, not by library defaults). Large single blocks may use threaded BLAS instead.
- **Memory:** a dense block of dimension N costs 8N² bytes (N = 20,000 is about 3.2 GB), so windowed Lanczos/Davidson on sparse CSR is the default above a few thousand. Per-block cost estimates (section 5.3) decide dense versus iterative. Enforce a configurable memory budget and fail early with a clear message, not an OOM kill.
- **Linear algebra backend:** abstract it behind one `linalg_backend` layer. Default is Eigen's own kernels (vendorable, no system dependency). Optional LAPACK/OpenBLAS behind `RISS_DRUCKER_LAPACK` for speed; because bundled NumPy/SciPy BLAS is not linkable, either vendor OpenBLAS under `thirdparty/` or use whatever the Kaggle image provides after the probe below confirms it. Tests must pass on both backends.
- **Exact angular algebra:** rational-with-square-root arithmetic needs big integers (vendor GMP or Boost.Multiprecision, or do the exact step in Python with SymPy/`fractions` and export floating-point results). Decide in M2 by benchmark, and record the choice in `docs/conventions.md`.
- **Determinism:** threaded reductions are not bitwise reproducible. Fix seeds, and write tests with tolerances rather than exact equality. Record thread count and backend in every run.
- **Checkpoint and resume:** Kaggle sessions have time limits and `/kaggle/working` is ephemeral. Jobs are idempotent and keyed by (spec hash, θ hash, block ID). Write per-block results atomically, resume from what exists, and publish caches and wheels as a Kaggle dataset or notebook output version between sessions.
- **Wheel and ABI:** a wheel is tied to the Python version and the image's libraries. Rebuild it when the Kaggle image changes. `scripts/env_probe.py` (run in M0 and at the start of every notebook) records Python version, compiler (`g++`/`cmake` availability and versions), BLAS/LAPACK, core count, RAM, disk, GPU presence and installed packages, and fails fast if a required piece is missing.
- **Offline rule for LLM and ab initio work:** submission-style notebooks must not call external APIs. The Part 1 LLM extractor and heavy ab initio runs happen offline (locally or on other hardware), and their outputs (records, point sets) are shipped as Kaggle datasets. Never commit API keys; use Kaggle secrets where a key is unavoidable.
- **Local development:** build and test locally with the same CMake and `thirdparty/` vendoring pattern as your other C++ projects, and use Kaggle for the larger runs. CI builds the scalar and SIMD-dispatched paths with the Eigen-only and LAPACK backends.
- **Performance:** add a benchmark target per solver path (dense, Lanczos, Sturm) with a recorded baseline. Profile before optimizing. SIMD kernels (section 4) and any GPU path need a benchmark that shows the gain and agreement with the scalar CPU reference.

## 4. Project form and Intel CPU acceleration

**Form:** a C/C++/Python library. The numerical core is C++ (with an optional `extern "C"` API header for other bindings), exposed to Python through pybind11. Python orchestrates; Kaggle notebooks are thin drivers. Python import name and C++ namespace: `riss_drucker`. No fixed directory tree is prescribed: create directories as milestones need them, and keep these conventions: third-party code under `thirdparty/` (section 3), tests under `tests/`, conventions in `docs/conventions.md`, `scripts/` for `kaggle_build.sh` and `env_probe.py`, and `schemas/` for the record and result schemas.

**Target hardware: Intel x86-64 CPUs with hardware acceleration.** SIMD is expected, not avoided. The constraint is that one wheel must stay portable across Kaggle sessions whose CPUs may differ, so acceleration is chosen at runtime instead of baked in with `-march=native`.

- **Baseline build:** compile portable code at `-O3 -march=x86-64-v2` so it runs on any session.
- **Runtime dispatch:** build hot kernels in multiple variants (baseline, AVX2+FMA, and AVX-512 where the CPU has it) and select at load time with CPU feature detection (function multi-versioning, `__builtin_cpu_supports`, or a vendored SIMD library such as Highway or xsimd). Apply ISA flags per translation unit, never globally. Verify which instruction sets the Kaggle CPUs actually expose with `env_probe.py`, because AVX-512 is not guaranteed.
- **Where SIMD pays off here:** sparse CSR matvec in Lanczos/Davidson, dense GEMM and `eigh` inside blocks, batched Sturm/inertia sequences (vectorize across many shifts at once), Hellmann-Feynman expectation values ⟨ψ|O_k|ψ⟩, Chebyshev/stochastic-trace moment recurrences, and the dot products and axpys in reorthogonalization.
- **Vendor libraries:** oneMKL (BLAS, LAPACK, sparse BLAS) behind `RISS_DRUCKER_MKL` is the natural Intel-optimized backend. Check its license and redistribution terms and how it can be obtained offline on Kaggle before depending on it (vendoring rule in section 3 still applies). Eigen's built-in kernels stay the portable default; OpenBLAS stays available behind `RISS_DRUCKER_LAPACK`. Avoid mixing OpenMP runtimes (GNU `libgomp` vs Intel `libiomp5`) in one process, since that causes oversubscription and crashes. Pick one threading layer and set it explicitly.
- **Compiler:** GCC or Clang first. Intel oneAPI `icx` is optional and must not be required to build. Do not use `-ffast-math` anywhere that touches certificates, Sturm counts or inertia counts.
- **Numerical safety:** FMA and different summation orders change rounding. Every accelerated path must match the scalar reference within a stated tolerance, tests compare against the scalar path, and certificates are always evaluated with the reference-precision path at the final accepted θ.
- **Memory behaviour:** align arrays (64-byte), block for cache in dense kernels, and prefer structure-of-arrays layouts for batched shifts. Measure before tuning.
- **Benchmarks gate acceleration:** each dispatched kernel needs a benchmark showing the speedup over the baseline variant on the target CPU. Keep the scalar kernel as the reference and as the fallback.

**CMake options:** `RISS_DRUCKER_BUILD_TESTS`, `RISS_DRUCKER_PY`, `RISS_DRUCKER_SIMD_DISPATCH` (default ON), `RISS_DRUCKER_OPENMP`, `RISS_DRUCKER_MKL`, `RISS_DRUCKER_LAPACK`, `RISS_DRUCKER_CUDA` (later, default OFF).

## 5. Part 3: core numerics

### 5.1 ModelSpec (discrete structure, hashed)
Point or PI group and subgroup chain, mode content and symmetries, polyad scheme, vibrational truncation, J_max, coupling scheme, axis system, reduction/gauge (e.g. Watson A/S), operator list (name, order, symmetry label, tensor rank), optional LAM block (coordinate, Fourier/grid size, barrier parametrization). `ModelSpec.hash()` is canonical (sorted, normalized) and is the cache key. Any change to a field is a model-structure change: cache invalidated, model comparison required. Never change structure inside the fit loop.

### 5.2 OperatorLibrary (compile once, cache)
- `compile(spec) -> {block_id: [sparse O_k]}`.
- 3j/6j/9j via stable recursion direction; exact arithmetic (rational with square roots) where cheap, with SymPy cross-checks in Python. Isoscalar factors by projection operators; multiplicity labels are a gauge, so fix and record the convention.
- Every compile runs sum-rule certificates: orthogonality and completeness of CG/isoscalar factors, Hermiticity, total symmetry of each O_k, time-reversal invariance. Failure aborts compile.
- Cache: per-block CSR binary plus JSON manifest with spec hash and library version. A hash mismatch is an error, not a warning.

### 5.3 Solver
- H(θ) = Σ t_k O_k as a sparse sum.
- Dense `eigh` for small blocks. Lanczos or Davidson/LOBPCG with warm start for windows. Reorthogonalize Lanczos at these sizes.
- Certificates: Sturm count (tridiagonal/banded), LDLᵀ inertia count of (H − σI) otherwise, residual bound ‖Hx − λx‖/‖x‖. A window is accepted only if the inertia count matches the number of returned eigenvalues; otherwise widen and rerun.
- Per-block cost estimate: dimension, bandwidth, gap ratio ⟨r⟩ from a few low levels.

### 5.4 Jacobian and fit
- Hellmann-Feynman Jacobian from converged eigenvectors. Never autodiff through the eigensolver.
- Levenberg-Marquardt / Gauss-Newton with a Gaussian prior on θ (ab initio mean and covariance plus a model-error hyperparameter). Whitened, gauge-fixed coordinates; Σ_θ by pseudo-inverse along gauge null directions; log-magnitude and sign coordinates for parameters spanning decades.
- Level-line incidence matrix D (signed, sparse): solve (DᵀWD)E = DᵀWν per connected component with an anchored level, for MARVEL-style consistency and cycle-based outlier flagging before the Hamiltonian fit.

### 5.5 Posterior and pushforward
- Output θ̂, joint Σ_θ, samples or sigma points.
- Per-level σ_i² = g_iᵀ Σ_θ g_i; windows E_i ± kσ_i inside a Weyl fence; flag levels with σ_i/gap above ~0.3 as near-crossing and resample those only.
- Assignment: χ²-gated candidates, joint position + intensity likelihood, monotone alignment (DP) within a block, ambiguity sets, expected misassignment count, Benjamini-Hochberg across a band. Poisson null for chance coincidences, P(≥1) = 1 − exp(−2δρ).
- Outlier mixture (narrow Gaussian + broad component) for unmodeled perturbers.
- Truncation check: second-order neglected-shift bound per level.

### 5.6 Moments and basis diagnostics
- Spectral moments / Chebyshev moments per block (stochastic trace estimation, no diagonalization): gauge-invariant fingerprint and ML feature. Moments are basis-invariant, so they do **not** identify Hund's cases.
- **Coupling-case diagnostic:** build H in each candidate coupling basis and compute the off-diagonal fraction ‖H_offdiag‖²/‖H‖². The best-matching basis is the appropriate case. Report as a ratio or distribution over cases, as a function of J (cases can change with J). Published case labels are weak labels only.

### 5.7 Synthetic data (`synth.py`)
Sample ModelSpec → sample θ → compile/solve → apply selection rules, Boltzmann and spin weights → add noise (Gaussian position noise ∝ FWHM/SNR, log-normal intensity noise). Level-sequence generators for sanity checks: β-Hermite tridiagonal sampler (β = 1, 2, 4; β → 0 for Poisson), Brody. Ground-truth assignments stored with every set.

### 5.8 Ab initio interface
Use existing codes (e.g. PySCF, Psi4, or what is available/licensed) through a file boundary: geometries in, energies/dipoles/diabatic elements out (Parquet or JSON). Multi-fidelity correction (start with a GP, predictive variance required) lives in `abinitio/`. Converge on gauge-invariant observables (levels, combination differences), not PES RMSE. Do not treat ab initio training data as the target the posterior must match (double counting).

## 6. Part 1: data gathering and extraction

### 6.1 Sources
- **ExoMol:** `.states` (energies in cm⁻¹, state ID, quantum numbers, uncertainty where given) and `.trans` (upper/lower IDs, Einstein A, wavenumber); definition files `.def.json`; master file `exomol.json`. Lists can have billions of transitions: ingest `.states` fully, `.trans` filtered by intensity cutoff, J range or band.
- **ExoMolHR:** CSV of high-accuracy lines (requires explicit uncertainties). PyExoCross for cross sections.
- **MARVEL datasets:** assigned transitions into consistent levels with propagated uncertainties (e.g. H2CO, C2, SO, H2CS, methane studies exist). Preferred fit input.
- **MoLLIST** experimental diatomic line lists.
- **HITRAN / HITEMP / GEISA:** useful for intensities and broadening. These are outputs of fitted effective Hamiltonians, so do **not** treat their positions as independent measurements.
- **Not yet verified, check before use:** JPL/CDMS submillimeter catalogs, PNNL cross sections, IAO LMS spectra, W@DIS (measured transitions without intensities).
- **Papers:** observed transitions/levels, effective-Hamiltonian constants, fit statistics. Prefer machine-readable supplementary files (MARVEL-format inputs, line-list tables) over PDF tables.
- Check each source's license/redistribution terms before uploading anything to Kaggle, including private datasets.

### 6.2 Record schema (`schemas/record.schema.json`)
Every record carries:
- `source`: `measured` | `marvel` | `computed` | `ehamil_derived`. The fit refuses `ehamil_derived` as independent data by default.
- Provenance: DOI, table/page, extraction method, extraction confidence.
- Units and uncertainty definition (1σ vs 2σ), assignment, J, symmetry, nuclear-spin species.
- For constants: **convention metadata** (operator definitions, reduction, sign conventions, units MHz vs cm⁻¹, which parameters were fixed or floated), model specification (group, operator list, coupling scheme), fit statistics (rms, line count, J/band range).

### 6.3 Extraction workflow
1. **Gold set first:** hand-extract ~20 papers into `gold/`. This defines the schema and is the evaluation set for the automated extractor.
2. **Automated extractor:** LLM-based, outputs the record schema with confidence.
3. **Round-trip validation (mandatory):** compile the extracted spec, plug in the extracted constants, reproduce the paper's own line list or reported residuals with the Part 3 code. If the numbers do not reproduce, the extraction or the convention is wrong; the record is quarantined, not ingested. This catches OCR errors, sign flips and unit mistakes.
4. Store derived numeric data plus citation, not paper text. Prefer arXiv/open-access sources; check publisher terms before bulk ingestion.
5. Real data is small and clustered by molecule family: use it for validation and conventions. Use synthetic data for bulk training. Split by band or molecule, never random lines.

### 6.4 Calculated-vs-empirical error prior
Where ExoMol provides both computed and MARVEL-replaced levels, extract the difference as an empirical distribution of computed error versus J, band and symmetry. This feeds the ab initio model-error hyperparameter.

## 7. Part 2: structure proposer

- Proposes structure only: operator list, polyad scheme, coupling scheme, LAM treatment, as a typed string over a controlled grammar (attribute/type checking for symmetry, Hermiticity, time-reversal invariance, order bound). The compiler turns it into exact matrices. The proposer never writes angular algebra or numeric coefficients.
- Retrieval is from a **typed library**: operator dictionary per group chain (orders, symmetry labels, compiled O_k), baseline templates (Watson rigid rotor, polyad schemes, spin-rotation blocks, torsion-rotation block), with metadata on valid regimes and degeneracies.
- Inputs are structured diagnostics, not raw residuals: blocks with large residuals, residual patterns versus J/K, near-crossing flags, intensity-borrowing signatures, Fisher rank (identifiability), moment fingerprints, coupling-case diagnostic.
- Loop: propose delta → compile → fit with ab initio prior → score by evidence or cross-validation by band → accept only if identifiable and significant. Keep a structure posterior (model averaging over a few candidates), not a single string.
- Canonical form and numerical deduplication (compare levels at random θ) so equivalent models do not split probability mass.
- If the allowed operators at order Ω are enumerable from the group algebra, start with sparse Bayesian selection (ARD / horseshoe / spike-and-slab) over the dictionary. Add the generative proposer only for compositional choices (polyads, resonances, coupling scheme, LAM treatment).
- Training data: synthetic (structure → θ → simulate → spectrum/structure pair) plus round-trip-validated literature records. Validate on held-out molecule families.

## 8. Part 4: analysis and visualization

Standard spectroscopic plots, mapped to outputs, as interactive notebook widgets (Plotly/Bokeh):
- Loomis-Wood plots, Fortrat diagrams, reduced-term-value plots (perturbation spotting)
- combination-difference and residual-versus-J plots
- level windows with near-crossing flags, assignment ambiguity sets
- spacing histograms with Brody fits per block, Δ₃/Σ² curves, gap ratio
- Voigt-convolved simulated vs measured spectra
- corner plots of the θ posterior with gauge directions marked
- calibration plots: nominal vs empirical coverage, binned by J, polyad, σ/gap

## 9. Result schema (`schemas/result.schema.json`)

Model spec + hash; θ̂, Σ_θ (gauge directions marked), samples; ab initio baseline and correction stored separately; per-level and per-line predictive distributions with near-crossing flags; assignment posteriors with ambiguity sets; certificate outcomes (pass/fail + numeric bound); coverage report; supported (J, polyad) range and horizon flag (ε_Ω·ρ per block); coupling-case diagnostic per J.

## 10. Working agreements for Claude Code

- Surgical edits to existing files. Do not regenerate a file to change part of it.
- Read a file before editing it. Keep diffs small and scoped to one milestone.
- Tests first for each component. Certificates are code, not comments.
- Never silently absorb a failed certificate: surface it in the result schema.
- No dependencies outside `thirdparty/`. Each new dependency gets its own CMake wrapper.
- `TODO(riss-drucker-vN)` markers only for items listed as deferred in section 1; do not implement them yet, but keep interfaces compatible with them.
- CPU-first: never make a result depend on a GPU; any accelerated path must match the CPU path within tolerance and live behind a CMake option.
- Units, axis system, reduction and multiplicity conventions live in `docs/conventions.md` and are referenced by the spec.
- Never copy paper text into the repo. Store extracted numbers and citations only.

## 11. Milestones

Each milestone ends with passing tests and a runnable notebook.

- **M0 Skeleton:** CMake, thirdparty wrappers, pybind module, CI, `kaggle_build.sh`, `env_probe.py`, smoke-test notebook; portable flags verified; scalar and runtime-dispatched SIMD paths both build and agree within tolerance; Eigen-only, LAPACK and (if available) MKL backends build; wheel built in one session installs in a fresh one.
- **M1 Exact reduced baseline:** 1D Fourier torsion + rigid rotor; dense and Lanczos solves; Sturm/inertia certificates; barrier regime check against analytic limits (free rotor, harmonic).
- **M2 Operator library, small group:** asymmetric rotor (Watson reduction), sum-rule certificates, cache and hash-invalidation tests, parameter recovery on synthetic data.
- **M3 Probabilistic layer:** Hellmann-Feynman windows, near-crossing resampling, joint assignment with intensities, outlier mixture, coverage checks split by band.
- **M4 Data store and gold set:** record schema, ExoMol/MARVEL parsers, gold-set papers, round-trip validator, calculated-vs-MARVEL error prior.
- **M5 Multi-fidelity ab initio input:** Δ-surface with variance → variational solve → propagated level variance → Bayesian update with model-error hyperparameter; prior-posterior shift diagnostic.
- **M6 Automated extraction + analysis widgets.**
- **M7 Structure proposer:** sparse-selection baseline first, then grammar-constrained generative proposer; coupling-case diagnostic.
- **M8 (later):** degenerate-mode chains (Td-type polyads), open-shell spin terms and double group (Kramers/GSE class), LAM PI-group blocks.

### Later phases (planned after the time-independent quantum core is solid)

Each depends on the earlier milestones and reuses the same operator library, cache, certificates and result schema.

- **Phase B, vibronic and excited states:** multi-surface diabatic models (Jahn-Teller, Renner-Teller), diabatic coupling learning, CH4+-type systems. Needs: ModelSpec extended with electronic states, EOM/multireference ab initio inputs, and the horizon gate to hand over to statistics above the resolved regime.
- **Phase C, partitioned molecules:** active quantum region plus classical environment (QM/MM, system-bath), with environment-driven parameter distributions and embedding corrections. Needs: parameter distributions per conformer in the result schema.
- **Phase D, time and position dependence:** θ(t) from trajectories, Markov-state or HMM exchange models, frequency-correlation and lineshape analysis, ergodicity/recurrence diagnostics added to the certificate set.
- **Phase E, beyond bound states:** MQDT/Rydberg and autoionization, complex-eigenvalue statistics.
- **Phase F, in-house ab initio:** only if the file-boundary approach becomes limiting.

Suggested first molecules: H2O2 for the torsion baseline (M1/M8), H2CO or H2S for the asymmetric rotor and data store (M2/M4), methane for Td polyads (M8). Check what MARVEL/ExoMol data exist before committing to each.

## 12. Tests

- **Unit:** 3j/6j/9j vs closed forms and recursion-stability cases; sum rules; Hermiticity; symmetry of every O_k.
- **Solver:** eigenvalues vs dense reference; inertia count equals returned count; warm start equals cold start.
- **Fit:** parameter recovery on synthetic data; gauge test (equivalent parameter sets give identical levels; Σ_θ has the expected null directions).
- **Statistics:** β-Hermite sampler reproduces ⟨r⟩ ≈ 0.386 (Poisson), 0.531 (GOE), 0.600 (GUE); superposition test P(0) = 1 − Σf_i².
- **Calibration:** empirical coverage of windows vs nominal on held-out bands.
- **Data:** record-schema validation; round-trip reproduction of every gold-set paper; extraction accuracy against the gold set.
- **Regression:** cache round trip; spec hash stability across processes and platforms.

## 13. Known risks

- **Gauge freedom** (contact transformations, multiplicity labels, axis system, reduction choice): report invariants first, parameters second.
- **Convention mismatch across papers:** constants from different operator sets/conventions are not comparable. Always carry convention metadata.
- **Parameter explosion:** gate each block by ε_Ω·ρ; blocks past ~1 are reported statistically, not fitted level by level.
- **Differentiable `eigh` instability** near degeneracies: Hellmann-Feynman only.
- **Prior double counting:** training data for the surface must not also be the posterior's target.
- **Effective-Hamiltonian-derived data used as measurement:** enforced by the `source` field.
- **Low fitted Brody q is ambiguous** (weak chaos vs missing/mislabeled levels): do not interpret physically without an independent level count (inertia, density of states).
- **Extrapolation** beyond fitted J/polyad: polynomial effective Hamiltonians diverge; cap with the Weyl fence and flag.
- **Thread oversubscription** (block-level threads times BLAS threads): set both explicitly and benchmark.
- **Kaggle session limits and ephemeral storage:** checkpoint per block, publish caches/wheels between sessions, never assume the working directory survives.
- **Image drift:** the Kaggle image (Python, compiler, libraries) changes over time; `env_probe.py` output is part of every run record, and the wheel is rebuilt rather than reused across image changes.
- **GPU FP64 weakness:** do not move the Hamiltonian core to a GPU without a measured FP64 benefit and agreement with the CPU path.
- **Memory blowup in dense blocks:** estimate before allocating; fall back to windowed iterative solves.
