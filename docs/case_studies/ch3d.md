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
- These are catalogue (SPCAT-model) energies built on the measured lines and the infrared
  combination differences, not raw measurements. E″ is printed to 10⁻⁴ cm⁻¹, so σ ≥ 2.9×10⁻⁵ cm⁻¹
  (rounding).
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

The extrapolation error grows with J and is largest at J = 22–23, K = 0–3. That is where octic
terms, not offered by the dictionary here, start to matter.

## What the run exposed in the library (fixed)

- **Vanishing operators.** A K±3 operator is identically zero at J = 1. The relative matrix
  certificate divided round-off by round-off and rejected a valid term. Operators that vanish at a
  given J are now skipped there.
- **SPCAT signs.** A negative TAG marks a measured frequency, not cm⁻¹ units (those are flagged
  by a negative ERR). The reader handled the H₂O₂ catalogue wrongly until this was fixed. The
  combination-difference certificate caught it as a 9×10⁵ cm⁻¹ inconsistency.
