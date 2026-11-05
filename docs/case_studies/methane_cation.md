# Case study: methane cation, Jahn-Teller tunneling between six C2v wells

Reproduce with `python scripts/case_methane_cation.py`. Derived numbers are in
`methane_cation_results.json`.

**Source.** M. Grütter, H. J. Wörner, F. Merkt, J. Chem. Phys. 131, 024309 (2009), from the
open-access copy in the ETH Research Collection (hdl:20.500.11850/182792). Numbers taken:
- Table III: Δ = 123.6(5) cm⁻¹, Δ′ = 243.2(5) cm⁻¹, σ = −1.7(4) cm⁻¹;
- the J = 0 tunneling pattern of the four-well isomer;
- the CH₄⁺ ground-state tunneling splitting of about 16 cm⁻¹ quoted there from Wörner et al.,
  J. Chem. Phys. 126, 144305 (2007).

The 2007 CH₄⁺/CD₄⁺ paper itself and the related PRL are closed access. They were not used.

## Model, built from symmetry

`WellSpace` holds six localized minima. Each well's adiabatic electronic function is a vector of
the T₂ carrier space: the six (0, 1, ±1)/√2-type directions. The group permutes the wells with
signs, ρ(g) v_n = s v_m, and the signs encode the geometric phase. Nothing else is put in by hand.

## CH₄⁺ (Td)

- **Well representation:** T1 + T2 by characters. This is the geometric-phase result; without
  the signs it would be A1 + E + F2.
- **Operators:** the dictionary has exactly 2 well operators, the well energy and the
  nearest-well hop, certified by the character count. `sym()` of a single hop generates the full
  signed tunneling matrix with octahedral connectivity: each well couples to four others, not to
  the opposite one.
- **Splitting:** T1 and T2 are split by exactly 4|σ|. The published δ ≈ 16 cm⁻¹ gives
  |σ| = 4 cm⁻¹, which is the "|σ| ≈ 4 cm⁻¹" the paper uses as an upper bound for CH₂D₂⁺.

## CH₂D₂⁺ (the C2v subgroup that keeps the D pair)

The group is `Group.subgroup` of Td; the parent's T₂ matrices are restricted to it.
- **Wells:** they fall into the orbits {1}, {6} and {2–5}, the paper's three isomers.
- **Operators:** the dictionary has 8 well operators: 3 orbit energies and 5 hop classes. The
  paper's model keeps the single nearest-well σ.
- **J = 0 levels:** with the published Δ, Δ′ and σ, the four-well isomer gives

| Level | Model / cm⁻¹ | Published pattern |
|---|---|---|
| A1 | 120.2 | Δ + 2σ |
| B1 + B2 | 123.6 (×2) | Δ |
| A2 | 127.0 | Δ − 2σ |

The irrep labels come out of the projection and agree with the paper. With the phase convention
set by the well vectors, the published labels require σ < 0, again in agreement. Reversing the
sign swaps A1 and A2.

## Rovibronic PFI-ZEKE spectra of CH₄⁺, ¹³CH₄⁺ and CD₄⁺

**Data.** Wörner, Qian & Merkt, J. Chem. Phys. 126, 144305 (2007), Tables VII–IX, read from the
PDF supplied by the user with `scripts/case_methane_cation_rovibronic.py --pdf …`. The measured
line tables are not stored in the repository; only derived results are, in
`methane_cation_rovibronic_results.json`. Neutral lower levels use B₀(CH₄) = 5.2410 cm⁻¹ and
B(CD₄) = 2.63273 cm⁻¹ (the value the paper uses). Lines with an ambiguous N⁺ (e.g. "2/3") are
excluded, as in the paper.

**Model, built rather than chosen.**
- **Spaces:** rotation |N k⟩ ⊗ the six T₂ geometric-phase wells; group Td.
- **Localized rotor:** the compiler finds each well's unique axis. For well 1, x is unique and
  Jx Jy is forbidden, so the symmetry-allowed quadratic rotor is
  A J_x² + ½(B+C)(J_y² + J_z²) ± ½(B−C){J_y, J_z}. This is the paper's Table IV form, which
  `sym()` carries to all six wells.
- **Tunneling:** nearest-well, δ = 4|σ|.
- **Fit:** the same 5 constants as the paper (IE, A, B, C, δ).
- **Assignment:** each line goes to one level of its nuclear-spin class (A ↔ A1/A2, F ↔ F1/F2,
  E ↔ E), nearest at the published constants; after the fit it is re-assigned once and the result
  kept only if the rms improves. A ≥ B ≥ C is enforced.

| | Lines | rms (paper) / cm⁻¹ | A | B | C | δ |
|---|---|---|---|---|---|---|
| CH₄⁺ | 21 | 0.94 (1.00) | 6.37(34) | 5.68(24) | 4.12(11) | 17.1(5) |
| ¹³CH₄⁺ | 20 | 0.94 (1.00) | 6.75(39) | 5.62(25) | 3.94(11) | 16.6(6) |
| CD₄⁺ | 19 | 0.27 (0.28) | 3.335(32) | 2.511(37) | 1.995(28) | 1.74(25) |

Paper (Table VI):
- CH₄⁺: A 6.40(50), B 5.55(40), C 4.03(50), δ 16.4(40).
- ¹³CH₄⁺: A 6.30(50), B 5.70(45), C 4.05(40), δ 16.2(40).
- CD₄⁺: A 3.35(30), B 2.53(25), C 1.98(20), δ 1.4(3).

All of our constants agree within the uncertainties, and the rms matches the published fits.
IE differs by the field-correction convention: we fit the uncorrected line positions.

**Findings.**
- **Two CD₄⁺ labels.** Two lines in Table IX (J″ = 2 E at 102 207.3 and 102 211.2 cm⁻¹) are
  labelled N⁺ = 2, but their energies need upper levels at 26.2 and 30.1 cm⁻¹. That is the
  N⁺ = 3 E region for these constants (model: 27.2 and 30.3); N⁺ = 2 lies near 6B̄ ≈ 15 cm⁻¹. The
  script keeps published labels unless they miss by more than 3 cm⁻¹ and a neighbouring N⁺ fits
  three times better. Exactly these two lines are relabelled N⁺ 2 → 3, and the report says so.
  This most likely reflects a typo in the published table: the paper's own small residuals for
  them are impossible with N⁺ = 2.
- **No tunneling–rotation coupling.** Every such operator in the dictionary (hop × J²) was tried.
  BIC favours one for CD₄⁺ (ΔBIC = −13), but leave-one-line-position-out prediction does not
  improve (0.597 → 0.597 cm⁻¹). The gain comes from lines listed several times at the same
  position, so the extension is rejected. For CH₄⁺ the best extension predicts slightly worse.
  The data support exactly the paper's 5-constant model.
- **The geometric phase is required by the measured lines** (the PRL 97, 173003 claim, tested
  directly). Without the phase (wells permuted without signs, `WellSpace(..., geometric_phase=False)`)
  the vibronic ground state spans A1 + E + F2, not F1 + F2.
  - For CH₄⁺ and ¹³CH₄⁺ that model cannot be fitted at all: it has no A1 level at N⁺ = 1,
    which the observed J″ = 0 → N⁺ = 1 A1 line (101 777.88 cm⁻¹; ¹³CH₄⁺ 101 782.26 cm⁻¹) requires.
  - For CD₄⁺, where δ is small, it fits with rms 0.90 against 0.27 cm⁻¹ with the phase.
- **Sign of σ.** With the well-vector phase convention, the band origin (J″ = 1 → N⁺ = 0 F₂) fixes
  F₂ as the lower vibronic component, so σ > 0 here.

## Limits

- **Electron spin not resolved.** Spin-rotation is below the PFI-ZEKE resolution (0.5–1 cm⁻¹); the
  levels are labelled by N⁺, as in the paper.
- **CH₂D₂⁺ uses published constants only.** Δ, Δ′ and σ are the 2009 paper's fitted constants, not
  raw data. CH₄⁺, ¹³CH₄⁺ and CD₄⁺ are fitted to the measured lines.
