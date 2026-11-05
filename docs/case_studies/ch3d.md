# Case study: CH₃D ground state, built dynamically from the JPL catalogue

Reproduce with `python scripts/case_ch3d.py --data DIR --fetch`. Derived numbers are in
`ch3d_results.json`. The data are JPL catalogue tag 17003 (B. J. Drouin et al., JQSRT 110, 2077
(2009); C. Bray et al. (2017); infrared ground-state combination differences from Chackerian &
Guelachvili (1980, 1986) and Ulenikov et al. (1999); see d017003.cat).

CH₄ itself has no permanent dipole and no rotational catalogue entry. CH₃D, the C3v member of the
methane family, provides real rotational data.

## Data

- 203 catalogue lines give 230 (J, K) levels with J ≤ 23, via E′ = E″ + ν. Every level reached by
  more than one line agrees within 9×10⁻⁵ cm⁻¹ (combination-difference certificate).
- **Provenance: model output, not measurements.** The JPL file flags none of its 203 lines as
  measured (every TAG is positive), and E″ is the catalogue's computed lower-state energy. Every
  level here is therefore output of JPL's SPCAT Hamiltonian, so this study is a **model-reduction
  test**: how well a dynamically built model reproduces and extrapolates a published effective
  Hamiltonian. It is not a test against nature. The combination-difference agreement (9×10⁻⁵ cm⁻¹)
  sits at the 10⁻⁴ print rounding and cannot distinguish measured from predicted lines. E″ is
  printed to 10⁻⁴ cm⁻¹, so σ ≥ 2.9×10⁻⁵ cm⁻¹ (rounding). `levels_from_cat(..., measured_only=True)`
  now refuses this file, and `case_ch3d.py --measured-only` stops with that message.
- Assignment is by quantum number: the K of each model level comes from ⟨J_z²⟩ and is matched
  exactly to the catalogue (J, K). An earlier energy-ordered assignment swapped K = 0 and K = 3 at
  J = 22–23, where the rigid-rotor offset equals the K spacing. Matching by label removes that
  failure mode.

## Model construction

- **Start:** rotation |J k⟩ with group C3v; the two quadratic dictionary operators, (J_x² + J_y²)
  and J_z².
- **Growth:** `autobuild` proposed every C3v-invariant, time-reversal-even operator of degree 4 and
  6 (counts certified against characters). It accepted 8 on evidence and rejected the K±3
  operators (J_z J_x³ type) as removable by a contact transformation, as the usual reduction does.
- **Final model:** 10 parameters (2 + 3 quartic + 5 sextic). The Fisher rank equals the
  post-redundancy bound.
- **Rotational constants recovered:** B = 3.88016 cm⁻¹ and A = 2 × 2.62544 = 5.25087 cm⁻¹.
  The catalogue gives 116325 MHz (3.8802) and 157412 MHz (5.2507).

| | n | χ²/dof | rms / cm⁻¹ |
|---|---|---|---|
| Fit, J ≤ 12 | 101 | 0.46 | 1.9×10⁻⁵ |
| Held out, J = 13–23 (extrapolation) | 129 | — | 1.2×10⁻³ (max 3.9×10⁻³) |

The extrapolation error grows with J and is largest at J = 22–23, K = 0–3. Two explanations were
tested by forcing the terms into the model and refitting (`scripts/review_followups.py`, notes in
`review_followups.md`); neither holds:

| Model | χ²/dof | Held-out rms / cm⁻¹ | max \|residual\|, J = 22–23, K = 0, 3 |
|---|---|---|---|
| autobuild model (10 terms) | 0.463 | 1.15×10⁻³ | 3.9×10⁻³ |
| + the three ΔK = 3 operators autobuild rejected (fits seeded away from 0) | 0.463 | 4.9×10⁻³ | 2.7×10⁻² |
| + best single octic term (of the 10 C3v-allowed ones) | 0.41–0.46 | 1.07–1.57×10⁻³ | 3.1–4.0×10⁻³ |

- **ΔK = 3 operators (the method review's hypothesis):** they couple K = 0 and K = 3, but they are
  removable by a contact transformation, as autobuild's filter said. Forced in, they leave the fit
  unchanged and make extrapolation 4× worse. Their first derivative is exactly zero at zero
  coefficient (pure-K states), so they must be fitted from nonzero seeds; both signs give the same
  result, as expected for a gauge direction.
- **Octic terms (this study's earlier explanation):** no single octic term removes the J = 22–23
  misfit; the best reduces it from 3.9 to 3.1×10⁻³ cm⁻¹ and several make extrapolation much worse.
- **Most likely cause:** the "data" at J = 22–23 are JPL's own predictions, outside the range of the
  lines JPL fitted. The residual compares two extrapolations; it does not measure an error against
  nature.
- **Coverage:** the Fisher predictive σ of the held-out levels is calibrated: 61% within 1σ, 99%
  within 2σ, rms z = 0.96, against 68% / 95% / 1 if calibrated.

## What the run exposed in the library (fixed)

- **Vanishing operators.** A K±3 operator is identically zero at J = 1. The relative matrix
  certificate divided round-off by round-off and rejected a valid term. Operators that vanish at a
  given J are now skipped there.
- **SPCAT signs.** A negative TAG marks a measured frequency, not cm⁻¹ units (those are flagged
  by a negative ERR). The reader handled the H₂O₂ catalogue wrongly until this was fixed. The
  combination-difference certificate caught it as a 9×10⁵ cm⁻¹ inconsistency.
