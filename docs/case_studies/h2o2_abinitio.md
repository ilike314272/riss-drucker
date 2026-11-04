# Case study: H₂O₂ torsion from multi-fidelity ab initio

Reproduce with `python scripts/case_h2o2_abinitio.py --out DIR --relaxed` (PySCF + geomeTRIC,
CPU, a few minutes; `--reuse` re-reads the cached `h2o2_{low,high}.json` point sets).
Derived numbers are in `h2o2_abinitio_results.json`. Reference levels: Hunt et al.,
*J. Chem. Phys.* 42, 1931 (1965).

## Pipeline

1. Relaxed scan of the HOOH dihedral: every coordinate except the dihedral is optimized at
   MP2/cc-pVDZ (frozen-dihedral constraint).
2. Low fidelity: MP2/cc-pVDZ at 19 points (0°…180°, step 10°). High fidelity: CCSD(T)/cc-pVTZ
   single points at 7 of those geometries (step 30°). Both are written in the `pointset/1`
   JSON format (`rd.abinitio.io`), the file boundary between ab initio codes and the library.
3. `DeltaModel`: a periodic GP on the low-level points plus a periodic GP on the
   high − low difference. The predictive variance is the **sum** of both GP variances.
4. V(φ) → cosine series to order 6 → `torsion_rotor` levels. The internal-rotation constant
   F comes from the relaxed minimum geometry.
5. 200 joint GP samples of V(φ) are pushed through the same solve, which gives the
   propagated level standard deviation.

## Results (cm⁻¹, above the ground level)

| Level | Δ-model | ±σ (GP) | MP2/DZ only | Experiment |
|---|---|---|---|---|
| tunnelling splitting | 10.88 | 0.001 | 28.6 | 11.4 |
| 2 | 244.9 | 0.004 | 216.2 | 254.6 |
| 3 | 353.6 | 0.015 | 368.4 | 370.9 |
| 4 | 544.2 | 0.007 | 565.3 | 569.5 |
| 5 | 740.4 | 0.013 | 778.6 | 775.9 |

Barriers: trans 374 cm⁻¹, cis 2605 cm⁻¹. Leave-one-out z-scores of the seven high-level
points are −1.26, 0.24, −0.41, 1.69, −0.22, −0.18 and 1.14. These are consistent with unit
variance, so the GP is honest about **interpolation** error.

## What this shows and what it does not

- The Δ-correction fixes the qualitative failure of the cheap level: its tunnelling splitting
  is off by 2.5×. The Δ-model splitting is within 5% of experiment.
- The remaining errors are 4–5% on the excited levels, three to four orders of magnitude above the GP σ. That gap
  is **model error**: a 1D torsion with a fixed F, CCSD(T) in a triple-ζ basis, and no coupling
  to the OH stretches or bends. It is not interpolation error. The GP variance does not
  include it, and it must not be read as a total uncertainty.
- In the Bayesian update against spectra, model error enters through the separate
  model-error hyperparameter τ (`rd.fit`, M3), so it is not counted twice.
- An earlier version summed only the Δ-GP variance. That hid the low-fidelity interpolation
  error (caught by the synthetic coverage test in `test_ml_infrastructure.py`). The
  low-fidelity GP variance is now included.
