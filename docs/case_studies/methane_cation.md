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

## Limits

- **No rotation.** The rotational-tunneling structure, where each well has its own principal
  axes, is not built. Fitting the PFI-ZEKE line positions (shown only as figures) would need it.
- **Published constants only.** Δ, Δ′ and σ are the paper's fitted constants, not raw data. This
  case study checks that the dynamically built, symmetry-generated model reproduces the published
  structure; it is not an independent fit.
