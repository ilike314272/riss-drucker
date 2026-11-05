# Data sources (survey, M4)

What exists for the first molecules, with the best standard dataset for each
role. Surveyed 2026-11-04 by web search. **Fetched so far:** ExoMol H₂CO AYTY
(`.def.json`, `.states.bz2`, `.pf`) and the four H₂S MARVEL files from Mendeley Data
(SHA-256 checked). See `case_studies/`. The H₂CO MARVEL supplement is not in the arXiv
source and ScienceDirect refuses scripted downloads (403); it has to be downloaded by hand.
`discovery.ucl.ac.uk` answers 403 to scripted requests; use arXiv or the publisher
for those papers. Check every license before redistributing anything, including
private Kaggle datasets.

Items marked *(verify)* were taken from search snippets and have not been
checked against the primary record.

## Recommended standard examples

| Role | Dataset | Content | Access |
|---|---|---|---|
| Asymmetric rotor + data store (M2/M4), **first choice** | MARVEL H₂¹²C¹⁶O: Al-Derzi, Furtenbacher, Tennyson, Yurchenko, Császár et al., *JQSRT* **266**, 107563 (2021); arXiv:2102.07423 | 16,596 non-redundant transitions from 43 sources (16,403 validated) → 5,029 empirical levels with uncertainties; new pure-rotation measurements in the ground, ν₃, ν₄, ν₆ states; used to refine ExoMol AYTY | Supplementary material (transitions, levels, extended line list); arXiv; UCL Discovery |
| Second asymmetric rotor | MARVEL H₂³²S: Chubb et al., *JQSRT* **218**, 178–186 (2018); arXiv:1812.10503 | 44,325 transitions from 33 sources → 3,969 ortho + 3,467 para levels | Mendeley Data 10.17632/mx6skzhxp4.1; open access via UCL Discovery |
| Computed line lists (ExoMol) | H₂CO AYTY; H₂S AYT2; H₂O₂ APTY (Al-Refaie et al., ExoMol XV); CH₄ MM | `.states`/`.trans`/`.def.json`; AYTY ≈ 10⁷ states, 10¹⁰ transitions, so filter `.trans` | `www.exomol.com/db/<mol>/<iso>/<dataset>/` |
| Torsion baseline (M1/M8) | H₂O₂: Hunt, Leacock, Peters & Hecht, *J. Chem. Phys.* (1965), far-IR internal rotation *(verify citation details)* | Torsional levels ≈ 11, 254, 371, 569, 776 cm⁻¹; trans/cis barriers ≈ 386 / 2460 cm⁻¹ | Paper tables, so it is a gold-set item. Needs `n_fold=1` with V₁, V₂, V₃ |
| Td polyads (M8) | MARVEL ¹²CH₄: Kefala, Boudon, Yurchenko & Tennyson, *JQSRT* **316**, 108897 (2024) | 82,173 transitions from 96 sources → 23,292 levels up to 9,900 cm⁻¹, eight polyads | Supplementary material; UCL Discovery |
| **Methane cation (Phase B target)** | Wörner, Qian & Merkt, *J. Chem. Phys.* **126**, 144305 (2007) *(verify DOI 10.1063/1.2712840)*; Signorell & Merkt, *Faraday Discuss.* **115**, 205–228 (2000); Wörner et al. PRL (2006) | PFI-ZEKE rovibronic levels of CH₄⁺, ¹³CH₄⁺, CD₄⁺; effective tunnelling-rotational Hamiltonian; ground state a tunnelling doublet (T₂/T₁, ~16 cm⁻¹ apart); C₂ᵥ equilibrium, pseudorotation over six wells; adiabatic IE 101753.0(15) cm⁻¹ | No database: these are gold-set papers. Their supplements are the numbers to extract |
| Intensities / broadening only | HITRAN, HITEMP, GEISA | **ehamil_derived** positions: never independent fit input | `hitran.org` (free account) |
| JPL / CDMS catalogue levels (CH3D 17003, H₂O₂ 34004, H₂CO 30501) | JPL and CDMS catalogues | **ehamil_derived**: E″ is computed and unflagged lines are predictions. Measured lines (negative TAG): CH3D 0 of 203, H₂O₂ 301 of 38,357, H₂CO 407 of 7,016. Use `levels_from_measured` (network inversion, one catalogue anchor per component) for measurement-based term values | `spec.jpl.nasa.gov`, `cdms.astro.uni-koeln.de` |

No Kaggle dataset of ExoMol or HITRAN line lists turned up. Plan: publish our
own filtered, license-checked subsets as private Kaggle datasets with
`scripts/kaggle_datasets.py`.

## Why H₂CO first

It is a near-prolate asymmetric top (κ ≈ −0.96) that the M2 Watson model
already covers, with a recent, curated MARVEL set and a MARVEL-refined ExoMol
list. That gives both empirical levels (fit input) and computed levels (the
calculated-vs-empirical error prior, README 6.4) for the same states. H₂S
(nearly oblate-symmetric, with ortho/para) is the cross-check.

## Path to CH₄⁺ (not this stage)

CH₄⁺ needs Td/C₂ᵥ permutation-inversion groups, degenerate-mode chains (M8),
multi-surface Jahn-Teller models (Phase B) and the double group for the
unpaired electron (M8, Kramers class). The Merkt-group papers above are the
experimental anchor; neutral CH₄ MARVEL levels (M8) come first for the
operator library on Td.

## Methane cation (Jahn-Teller tunneling)

- Grütter, Wörner, Merkt, J. Chem. Phys. 131, 024309 (2009). Open-access copy in the ETH Research
  Collection (hdl:20.500.11850/182792), fetched through the repository's DSpace API. Numbers used:
  Table III constants and the quoted CH₄⁺ tunneling splitting. No text was copied.
- Wörner et al., J. Chem. Phys. 126, 144305 (2007) and the PRL 97, 173003 (2006) are closed access
  (Unpaywall: closed). They were not used, and the library does not use unauthorized copies.
