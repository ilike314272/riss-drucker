# Case study: H₂CO ground state from ExoMol AYTY (first real data)

Reproduce with `python scripts/case_h2co.py --data DIR --fetch` (about 1 min to parse
10.3 M states). The raw data is not in the repository. Derived numbers are in
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
496 of the J ≤ 30 levels are MARVEL-empirical, with median uncertainty about 2×10⁻⁴ cm⁻¹.

## Fit: Watson A-reduction, Iʳ, sextic, J ≤ 30

Plain fit: χ²/dof = 4.28. With the outlier mixture:
χ²/dof = 0.65, outlier fraction 0.050,
certified = True, rank 15/15 (no null directions).

| Parameter | Value / MHz | σ / MHz |
|---|---|---|
| A | 281970.556 | 0.031 |
| B | 38836.0416 | 0.011 |
| C | 34002.199 | 0.011 |
| DeltaJ | 0.0753155004 | 1.9e-05 |
| DeltaJK | 1.28972829 | 0.0003 |
| DeltaK | 19.4348767 | 0.0011 |
| deltaJ | 0.0104492942 | 1.5e-05 |
| deltaK | 1.02846546 | 0.004 |
| PhiJ | 8.73238619e-08 | 2.3e-08 |
| PhiJK | 3.55007121e-05 | 4.6e-06 |
| PhiKJ | -0.000144435007 | 1.8e-05 |
| PhiK | 0.00461378834 | 1.9e-05 |
| phiJ | 4.46652929e-08 | 1.4e-08 |
| phiJK | 1.52865919e-05 | 4.1e-06 |
| phiK | 0.00170806124 | 0.00033 |

A, B and C agree with the observed ground-state values quoted in the literature
(281971 / 38836 / 34002 MHz). A precise comparison with the microwave/THz fits
(e.g. Müller & Lewen, *J. Mol. Spectrosc.* 331 (2017); CDMS) is a gold-set item.

**Flagged levels.** All flagged levels are in the Ka = 9 stack (J = 9–19), plus the
Ka = 8 pair at J = 9. Their residuals are −0.0064 → −0.0048 cm⁻¹, about 10× their
MARVEL uncertainties, and drift slowly with J. Only the highest Ka is affected, which
points to a missing octic K-term (L_K·Ka⁸ ≈ 4×10⁷·L_K at Ka = 9) rather than bad data.
The library stops at sextic order. Adding octic terms is the next model step, and
re-testing this stack is the check. The robust fit down-weights these levels and
reports them; it does not drop them.

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
