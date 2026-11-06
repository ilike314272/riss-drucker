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
| 7 | δ not tested against its sensitivity | Confirmed | Uncertainty budget below: the measured δ lies inside it |
| 8 | Docs and verifier drifted | Confirmed | cc-pVTZ first-principles values recorded and checked; stale tables replaced; the verifier is described as a reproducibility check |
| 9 | Claude evidence thin, first-principles scoring circular | **Confirmed on CH₄⁺**: the plain dictionary matches or beats Claude; the synthetic blind test was biased toward the dictionary and is inconclusive | Dictionary baseline and hidden-truth synthetic molecule added (below); the circularity is documented |
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

Each row changes one input and recomputes the converged reaction-plane splitting (cm⁻¹,
CH₄⁺ / CD₄⁺). Measured: 16.4–17.1 / 1.4–1.74.

| Variant | cc-pVDZ | cc-pVTZ |
|---|---|---|
| Base (ab initio plane, CCSD(T) correction along the path) | 13.4 / 1.00 | 13.1 / 0.92 |
| CCSD(T) correction ×0 (pure MP2) | 11.1 / 0.78 | 10.0 / 0.63 |
| CCSD(T) correction ×2 | 16.3 / 1.32 | 17.5 / 1.38 |
| ZPE of the remaining modes ×0.9 | 12.5 / 0.94 | 12.1 / 0.85 |
| ZPE of the remaining modes ×1.1 | 14.4 / 1.07 | 14.3 / 0.99 |
| Harmonic valley instead of the ab initio plane | 9.1 / 0.87 | 8.3 / 0.72 |

- **The measurement is inside the budget.** The largest single uncertainty is the CCSD(T)
  correction, which is applied along the path only. Its plausible range alone moves δ(CH₄⁺) from
  10 to 17.5 cm⁻¹. "80% of the measured δ" is therefore within the method's uncertainty, neither a
  confirmed success nor a confirmed shortfall.
- **The next calculation is clear:** coupled-cluster energies on (part of) the plane, so the
  correction is computed rather than assumed constant across the valley.
- **Spin contamination is negligible:** UHF ⟨S²⟩ = 0.756 at the minimum and 0.758 at the saddle
  (pure doublet 0.75).

## 9. Baselines

**CH₄⁺, the operator dictionary without a language model**, scored exactly like Claude's proposals.
The candidates are 3 first-order and 8 second-order symmetry-allowed rotation–tunneling
operators; two duplicate the rotor and are rejected by the compiler.

| Scoring | Reference | Dictionary, greedy 1 → 4 terms | Claude's proposals |
|---|---|---|---|
| Against experiment (mean leave-one-position-out rms, cm⁻¹) | 0.91 | 0.83 → 0.78 → **0.68** → 0.68 | 0.72 (Coriolis term), 0.73 (four couplings) |
| Against first-principles levels (mean predicted rms, N⁺ > 5) | 0.70 | 0.44 → 0.31 → 0.15 → **0.15** | 0.35 (four couplings), 0.53 (Coriolis); 0.0002 when the frame-rotation operator from the prompt was written out |

- **The plain dictionary matches or beats Claude in both modes.** Claude's proposals were
  physically reasoned and certified, but on these data they are not better than a systematic
  search.
- **Claude only wins outright by copying the generating operator,** which is circular.
- **The strongest point in Claude's favour is the harder split:** the Coriolis term is the one
  proposal that improves leave-one-N⁺-group-out prediction for CH₄⁺ (1.70 vs 2.43 cm⁻¹). The
  dictionary models have not been run through that split yet.

**Hidden-truth synthetic molecule** (water-like asymmetric top; hidden terms `sym(JxJxJyJy)`,
`sym(JyJyJyJy)`, `sym(JxJxJxJxJzJz)`; noise 10⁻⁵ cm⁻¹; fit J ≤ 6, score J = 7–9):

| Model | Predicted rms (J = 7–9), cm⁻¹ |
|---|---|
| Reference (rigid rotor plan) | 0.76 |
| True model | 3.9×10⁻⁵ |
| Dictionary greedy, 4 steps | 6.0×10⁻⁵: recovers all three hidden terms plus one spurious one; its first pick was wrong |
| **Claude** (Opus, 4 rounds, $0.58 reported usage; run on the WSL machine) | **6.5×10⁻³**: a 25-term Watson-style expansion up to 8th order plus Jz¹⁰ (the test was biased; see below) |

**Claude scored about 100× worse than the dictionary, but this test was biased against it.** The
transcript (`synthetic_llm_transcript.json`) shows four rounds:

| Round | Terms | Fit rms (J ≤ 6) | Predicted rms (J = 7–9) | Claude's reasoning |
|---|---|---|---|---|
| 1 | 17 | 3.1×10⁻⁵ | 0.0136 | "Water is a light, strongly K-dependent rotor": all quartic and sextic A-reduction terms plus two octic |
| 2 | 25 | 5.7×10⁻⁶ | 0.0065 | read round 1 as truncation, completed the octic shell |
| 3 | 35 | 4.0×10⁻⁶ | 0.030 | added the decic shell, flagging the overfitting risk |
| 4 | 23 | 5.5×10⁻⁶ | 0.0129 | diagnosed overfitting from coefficient growth and pruned |

The reasoning is competent: C2v acts on J as D2, it checked the axis assignment, and it recognized
overfitting. But the test design favoured the dictionary in three ways:
1. **The molecule was named.** "H₂O" with a real geometry invited Claude to use what it knows about
   real water (very large centrifugal distortion). The synthetic target is a rigid rotor plus three
   small hidden terms.
2. **The hidden terms came from the dictionary's own operator basis** (`enumerate_terms`), so the
   greedy search could recover them exactly. Claude wrote the standard Watson A-reduced form, which
   is equivalent to the unreduced operators only to a given order.
3. **The noise level (10⁻⁵ cm⁻¹) was not stated.** Round 1 already fitted at the noise level.
   Claude assumed levels exact to 6 decimals and read the misfit as truncation, which drove the
   long expansions.

The budgets were also unequal: about 80 trial evaluations for the search against 4 proposals for
Claude, which saw only the total score.

**Conclusion from this test: inconclusive about Claude, and informative about test design.** A fair
version would use an anonymous asymmetric top (made-up constants, no formula), a truth generated
outside both candidate sets (for example a reduced Hamiltonian with random constants, or a hidden
resonance), a stated noise level, and equal evaluation budgets, with Claude allowed to request
trial fits. On the CH₄⁺ data, where these biases do not apply, the systematic dictionary search
still matches or beats Claude's proposals, and Claude's one clear win (the first-principles levels)
was circular.
