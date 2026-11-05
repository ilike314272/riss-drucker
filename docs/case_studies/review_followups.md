# Response to the method review

A methodology review of the case studies (Nov 2026) raised 13 findings. This page records what
was checked, what changed, and what the checks showed. Several claims in the earlier write-ups were
weakened as a result, and two explanations (one from the review, one from the CH3D study) turned out
wrong.

Numbers come from `scripts/review_followups.py` (results in `review_followups.json`, generated
summary in `review_followups_results.md`). Rerun them with:

```bash
python scripts/review_followups.py all --data DATA_DIR --pdf JCP_126_144305.pdf
```

`DATA_DIR` holds `ch3d_jpl/c017003.cat`, `h2o2_jpl/c034004.cat` and `h2co_cdms/c030501.cat`. Add
`--llm --model opus` to also run Claude on the hidden-truth synthetic molecule.

## Status of each finding

| # | Finding | Verdict | What changed |
|---|---|---|---|
| 1 | Catalogue levels are model output | **Confirmed, and worse than stated** | CH3D: 0 of 203 lines measured. Provenance is reported by `levels_from_cat`, `measured_only=True` refuses predicted-only files, and the new `levels_from_measured` builds term values from measured lines by network inversion. CH3D is now described as a model-reduction test. |
| 2 | CH3D misfit: test the ΔK = 3 operators | **Tested: refuted**, and the octic explanation too | See below |
| 3 | Evidence uses one arbitrary prior | Valid | Not yet changed in `autobuild`; the held-out and K-stack splits below now carry the selection evidence |
| 4 | Correlated inputs fitted as independent | Valid | Not yet changed |
| 5 | "Full rank" is not identifiability | Valid | K-stack hold-out added for CH3D; the H2CO Ka split is not done yet |
| 6 | CH₄⁺ statistics overstated | **Confirmed** | Per-isotopologue σ estimated (CH₄⁺ ≈ 1.1 cm⁻¹, not 0.5). The frame-rotation claim is rewritten as "suggestive for CH₄⁺, no evidence for CD₄⁺". |
| 7 | δ not tested against its sensitivity | Confirmed | Uncertainty budget below |
| 8 | Docs and verifier drifted | Confirmed | cc-pVTZ first-principles values recorded and checked; stale tables replaced; the verifier is described as a reproducibility check |
| 9 | Claude evidence thin, first-principles scoring circular | **Confirmed by the live run** | Dictionary baseline and hidden-truth synthetic molecule added (below); the circularity is documented |
| 10 | H2O2 headline omits the ZPE row | Valid | Headline now quotes both: 0.15–1.6% without ZPE, 0.5–2.0% with it (splitting 10.0 vs 11.4) |
| 11 | H2S solver tuned on its own data | Valid | Not yet changed |
| 12 | Error prior may be biased | Valid | Not yet changed |
| 13 | No coverage checks | Done for CH3D and CH₄⁺ | Results below; one surprise in the review's disfavour |

## 1. Provenance

| Catalogue | Lines | Measured | Levels from measured lines (network inversion) |
|---|---|---|---|
| CH3D, JPL 17003 | 203 | **0** | none: every level is SPCAT output |
| H₂O₂, JPL 34004 | 38,357 | 301 | 521 levels in 220 components (largest 11), each anchored on one catalogue E″ |
| H₂CO, CDMS 30501 | 7,016 | 407 | 507 levels in 133 components (largest 32); 12 lines flagged as outliers |

Where measured levels exist, the catalogue reproduces them to 3×10⁻⁵ cm⁻¹ rms. The catalogues are
faithful to their data, but the measured lines form small disconnected pieces. Most level energies
therefore still depend on the catalogue model, and studies built on catalogue levels test
reproduction of a published Hamiltonian.

## 2. CH3D: the ΔK = 3 test (and the octic one)

| Model | χ²/dof | Held-out rms / cm⁻¹ | max \|residual\|, J = 22–23, K = 0, 3 |
|---|---|---|---|
| autobuild model (10 terms) | 0.463 | 1.15×10⁻³ | 3.9×10⁻³ |
| + the three ΔK = 3 operators (seeded away from 0) | 0.463 | 4.9×10⁻³ | 2.7×10⁻² |
| + best single octic term (of 10) | 0.41–0.46 | 1.07–1.57×10⁻³ | 3.1–4.0×10⁻³ |

- **ΔK = 3 terms:** autobuild rejected them as removable by a contact transformation, the standard
  C3v reduction (it keeps the ΔK = ±6 term). Forced in, they leave χ² unchanged and make extrapolation
  4× worse. The review's memory of an "h₃-type term" refers to the ΔK = 6 operator.
- **A blind spot this exposed:** a ΔK = 3 operator has zero first derivative at zero coefficient.
  It has no expectation value in a pure-K state, so it acts at second order. A gradient fit started
  at 0 cannot move it, and a first-order score test cannot see it. autobuild's trial fits start from
  0 too, so terms like this can only be judged fairly with seeded fits. Here the contact-
  transformation filter rejected them first, which was correct.
- **Octic terms:** none removes the misfit. The residual at J = 22–23 compares our extrapolation
  with JPL's, because those "data" are themselves predictions.

## 13. Coverage of held-out predictions

| Prediction | n | Within 1σ | Within 2σ | rms z | Verdict |
|---|---|---|---|---|---|
| CH3D J > 12 extrapolation, Fisher σ_pred | 129 | 0.61 | 0.99 | 0.96 | calibrated |
| CH₄⁺ leave-one-position-out, σ = 0.5 (as used) | 21 | 0.24 | 0.62 | 2.49 | intervals 2.5× too narrow |
| CH₄⁺, σ = 1.08 (free-fit χ²/dof = 1) | 21 | 0.67 | 0.95 | 1.15 | calibrated |
| CD₄⁺, σ = 0.5 (as used) | 19 | 0.63 | 0.95 | 1.15 | about right |
| CD₄⁺, σ = 0.32 (free fit) | 19 | 0.37 | 0.68 | 1.8 | too narrow: the free-fit σ omits parameter error |

Calibrated values are 0.68, 0.95 and 1. The review expected CH3D's Fisher intervals to be orders of
magnitude too narrow; they are not. Held-out rms (10⁻³) against fit rms (10⁻⁵) is the wrong
comparison, because the predicted σ grows steeply with J. For CH₄⁺ the review was right: a common
σ = 0.5 cm⁻¹ was wrong, and the lines behave as if σ ≈ 1.1 cm⁻¹.

## 5. Harder splits

| Held out | Result |
|---|---|
| CH3D, one whole K stack (all J) | K = 0–2, 4, 5, 7–9: rms ≈ 5×10⁻⁵ cm⁻¹. **K = 3: 1.6×10⁻³; K = 6: 2×10⁻⁴.** The C3v K = 3 (and 6) A1/A2 splittings cannot be predicted without data on them. |
| CH₄⁺ / CD₄⁺, one whole N⁺ group (rms, cm⁻¹) | reference 2.43 / 0.79; Claude's four-coupling models 2.55 / 0.78; **Claude's single Coriolis term 1.70 / 0.81** |

The N⁺-group split is the more meaningful test of the Claude proposals. Only the physically
motivated Coriolis term generalizes better for CH₄⁺; the richer models do not.

## 7. Uncertainty budget for δ

See `review_followups_results.md` (section 5) for the full table. Each variant changes one input
and recomputes the converged reaction-plane splitting:
- the CCSD(T) correction ×0 and ×2 (it is applied along the path only);
- the zero-point energy of the remaining modes ×0.9 and ×1.1;
- the harmonic valley instead of the ab initio plane;
- the basis set.

UHF ⟨S²⟩ at the minimum and saddle measures spin contamination.

## 9. Baselines

See `review_followups_results.md` (section 3). On CH₄⁺, every symmetry-allowed first- and
second-order rotation–tunneling operator is tried, singly and by greedy forward selection to four
extra terms. They are scored exactly like Claude's proposals, against experiment and against the
first-principles levels. The hidden-truth synthetic molecule is water with three undisclosed
quartic/sextic terms and 10⁻⁵ cm⁻¹ noise, fitted to J ≤ 6 and scored on J = 7–9. Claude cannot
have read the answer to it.
