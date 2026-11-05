# Case study: H₂³²S MARVEL, independent network inversion

Reproduce with `python scripts/case_h2s.py --data DIR --fetch` (the four files come from
Mendeley Data through its public API; SHA-256 checked against the published hashes).
Data: Chubb et al., *JQSRT* 218, 178–186 (2018); Mendeley Data 10.17632/mx6skzhxp4.1.
Derived numbers are in `h2s_marvel_results.json`.

## Network inversion vs the published MARVEL levels

| Spin | Lines | Deleted by MARVEL | Flagged by us | Levels reproduced | Median \|Δ\| / cm⁻¹ | Max \|Δ\| / cm⁻¹ | Within MARVEL σ | Median σ ours / MARVEL |
|---|---|---|---|---|---|---|---|---|
| para | 18851 | 74 | 36 | 3467/3467 | 2.4e-05 | 1.2e-04 | 98.5% | 7.3e-04 / 6.8e-04 |
| ortho | 25474 | 179 | 33 | 3969/3969 | 4.8e-05 | 1.0e-04 | 98.1% | 7.1e-04 / 6.6e-04 |

Every published level is reproduced from the raw transitions, with uncertainties
consistent with MARVEL's.

**What real data taught the solver** (all fixed, with regression tests):
1. **Precision dynamic range.** Claimed uncertainties span 10⁻⁶ … 5×10⁻² cm⁻¹ (weights
   span 10⁹). A plain weighted solve, *and* Huber reweighting in units of each line's own σ,
   let a few mis-assigned lines with tiny σ drag thousands of levels (first attempt: levels
   off by ~100 cm⁻¹, 91% of lines "outliers"). The fix: detect outliers with capped
   precision σ_eff = max(σ, median σ), then solve the clean set at full precision and
   re-judge every line, un-flagging lines that fit.
2. **MARVEL file convention.** A negative wavenumber marks a line deleted by the MARVEL
   analysis (tags such as `…_pt`, `…incorrected`). Such lines are now read as `excluded`.
   Before this fix two levels came out at negative energies.

## Ground-state fit (Watson A, IIIʳ) to our inverted levels

81 confirmed levels with J ≤ 8. χ²/dof by order:
4: 4.38e+05, 6: 5.46e+03, 8: 273;
robust 0.99 with outlier fraction 0.19.

This is a physics result about the model: the Watson power series converges slowly for
this light, floppy molecule (H₂S, like H₂O). Even at octic order the plain fit is far from
the data's precision, and the robust fit has to down-weight about 19% of the levels. At
J ≤ 12 the outlier fraction reaches 0.5, and `fit_robust` now flags that as "more likely a
wrong model or a bad starting point than bad data" instead of reporting χ²/dof ≈ 0.2. The
next model step for this class is a resummed (Padé/Euler-series) rotational Hamiltonian or
a variational treatment.
