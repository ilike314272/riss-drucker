# Case study: H₂CO ground state from ExoMol AYTY (first real data)

Reproduce with `python scripts/case_h2co.py --data DIR --fetch --jcut 38 --order 8` (about 1 min
to parse 10.3 M states). The raw data is not in the repository. Derived numbers are in
`h2co_ayty_results.json`.

**Data.** ExoMol AYTY, version 20230430, dataset DOI 10.5281/zenodo.5716868. The states
carry `SourceType` (Ma = MARVEL, Ca = calculated) and `Ecal` (calculated energy). The
MARVEL energies come from Al-Derzi et al., *JQSRT* 266, 107563 (2021); the line list from
Al-Refaie et al. (2015). Check the license before redistributing the raw files.

| File | Bytes | SHA-256 | Retrieved (UTC) |
|---|---|---|---|
| `1H2-12C-16O__AYTY.def.json` | 9169 | `9a995832568a8fab…` | 2026-11-04T22:22:21 |
| `1H2-12C-16O__AYTY.states.bz2` | 157226592 | `c4b745f11aa586d6…` | 2026-11-04T22:22:29 |
| `1H2-12C-16O__AYTY.pf` | 75000 | `334d19b2bd2e46d2…` | 2026-11-04T22:22:30 |

The ground vibrational state with J ≤ 40 has 1,681 levels, all Σ(2J+1) of them.
544 are MARVEL-empirical (J ≤ 38), with median uncertainty about 2×10⁻⁴ cm⁻¹.

## Fit: Watson A-reduction, Iʳ, octic, J ≤ 38

Order continuation (quartic → sextic → octic, each started from the previous solution).
A cold octic start failed to converge. Its parameters span about 13 decades
(A ≈ 9, L_J ≈ 10⁻¹³ cm⁻¹), and a robust fit seeded from it then called half the data
outliers. That is why `fit_by_order` exists and why `fit_robust` now warns.

Robust χ²/dof = 0.48, outlier fraction 0.037,
certified = True, rank 24/24, warnings: none.

| Parameter | Value / MHz | σ / MHz |
|---|---|---|
| A | 281970.485 | 0.039 |
| B | 38836.0526 | 0.024 |
| C | 34002.2001 | 0.024 |
| DeltaJ | 0.0753375424 | 4e-05 |
| DeltaJK | 1.29102106 | 0.00082 |
| DeltaK | 19.4152717 | 0.005 |
| deltaJ | 0.0104450066 | 3.7e-05 |
| deltaK | 1.02933464 | 0.012 |
| PhiJ | 1.10495884e-07 | 9.5e-08 |
| PhiJK | 2.27405199e-05 | 1.7e-05 |
| PhiKJ | -0.000127774652 | 7.2e-05 |
| PhiK | 0.00419572691 | 0.00011 |
| phiJ | 3.63318615e-08 | 5.6e-08 |
| phiJK | 6.4201956e-06 | 2.2e-05 |
| phiK | 0.00216291624 | 0.0013 |
| LJ | -5.77670336e-13 | 6.2e-11 |
| LJJK | 1.04388622e-08 | 2.2e-08 |
| LJK | -1.90547926e-06 | 2.1e-07 |
| LKKJ | 5.86618165e-06 | 8.8e-07 |
| LK | -1.75149975e-06 | 8.4e-07 |
| lJ | 2.31101564e-11 | 3.2e-11 |
| lJK | -6.19488904e-09 | 1.2e-08 |
| lKJ | 1.23876755e-06 | 1.5e-06 |
| lK | -8.5394767e-05 | 9.5e-06 |

A, B and C agree with the observed ground-state values quoted in the literature
(281971 / 38836 / 34002 MHz). Comparing with the microwave/THz fits (Müller & Lewen,
*J. Mol. Spectrosc.* 331 (2017); CDMS) is a gold-set item.

## The Ka = 9 stack, and what constrains high Ka

Our MARVEL-only fit flags the Ka = 9 stack: mean residual −0.005 cm⁻¹ (sextic) or
−0.003 cm⁻¹ (octic), rms/σ 6–7, while Ka = 10 and 11 fit. Adding octic terms does not
remove it, so it is not simply a missing K-term. (An earlier version of this note
blamed one; that was wrong.)

An independent check: the CDMS Hamiltonian (Müller & Lewen 2017; the gold entry
`gold/cdms-h2co-v3`, round-trip validated against SPCAT) evaluated at the MARVEL levels:

| Ka | 0–7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|
| mean MARVEL − CDMS / cm⁻¹ | −0.0002 … −0.0001 | +0.0008 | −0.0017 | +0.0115 | +0.0225 |
| rms / σ_MARVEL | 0.3–0.9 | 60 | 1.8 | 11 | 20 |

- For Ka ≤ 7, MARVEL and the microwave/THz-based Hamiltonian agree within MARVEL's
  uncertainties, which cross-validates both.
- For Ka ≥ 8 they disagree by 10⁻³–2×10⁻² cm⁻¹, and against CDMS the Ka = 9 stack is not
  the outlier.
- H₂CO has only µ_a, so its 407 microwave lines (Ka ≤ 15) are ΔKa = 0 transitions. They fix
  the structure *within* each Ka stack but not the offsets *between* stacks, which come from
  infrared data only.

**Conclusion:** the high-Ka stack offsets are the weakly determined part of the H₂CO ground
state, and no single source is the truth there. Predictive windows for high-Ka levels
should reflect that, e.g. through a per-Ka model-error term. Resolving it needs the IR
transitions (the MARVEL H₂CO supplement, or the IR combination differences of Müller
et al. 2000).

## Calculated-vs-empirical error prior (README 6.4)

5111 MARVEL-tagged states in all vibrational states: E_MARVEL − E_calc has mean
-0.0522 cm⁻¹ and robust σ 0.0947 cm⁻¹.
Smooth model: σ(J) = √(a² + (b·J(J+1))²) with a = 0.0665 cm⁻¹ and
b = 1.68e-04 cm⁻¹.

| Vibrational state | n | Mean / cm⁻¹ | Robust σ / cm⁻¹ |
|---|---|---|---|
| v1..v6 = 000000 | 544 | -0.0162 | 0.0226 |
| v1..v6 = 010000 | 509 | -0.1411 | 0.0829 |
| v1..v6 = 000001 | 443 | -0.1271 | 0.1088 |
| v1..v6 = 100000 | 443 | -0.0170 | 0.0734 |
| v1..v6 = 000010 | 386 | +0.0697 | 0.0700 |
| v1..v6 = 001000 | 384 | -0.0064 | 0.0416 |
| v1..v6 = 000100 | 376 | -0.0104 | 0.0773 |
| v1..v6 = 020000 | 373 | -0.1490 | 0.1233 |
| v1..v6 = 020001 | 248 | -0.1934 | 0.0639 |
| v1..v6 = 001001 | 238 | -0.0389 | 0.0820 |

The errors are band-dependent and biased (e.g. ν₂ and ν₆ about −0.13 to −0.15 cm⁻¹),
so the M5 model-error prior should be per band, not one global σ.
