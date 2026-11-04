# Gold set (README 6.3)

Hand-extracted records from ~20 papers. They define the record schema in
practice and are the evaluation set for the automated extractor (M6).

Rules:
- Numbers and citations only. Never copy paper text, figures or tables verbatim
  beyond the numbers themselves.
- One directory per paper: `gold/<first-author><year>-<short>/records.jsonl`, plus
  `NOTES.md` (table/page of every number, conventions as stated by the paper,
  and anything ambiguous).
- Every constants record carries `convention` (reduction, representation,
  name_map with signs, units) and `fit_statistics`, and must pass
  `riss_drucker.data.roundtrip.validate_constants` against that paper's own
  lines or levels before it counts as gold.
- `source`: `measured` for lines and levels the paper measured; `marvel`
  for MARVEL outputs; `ehamil_derived` for values computed from the paper's
  Hamiltonian.

## Entries

| Entry | Content | Validation |
|---|---|---|
| `cdms-h2co-v3/` | H₂CO ground-state constants, Watson S-reduction through octic order (Müller & Lewen 2017; CDMS 030501 v3), from the published SPFIT `.var`. Built by `scripts/gold_cdms_h2co.py` | Round trip against SPCAT's own `.egy` levels (J ≤ 25 = KNMAX): rms 2.9×10⁻⁷ cm⁻¹ (print precision); a sign-flipped d₁ is quarantined. Decic p₅ and hyperfine terms are unmapped and listed |

Planned next: H₂O₂ far-IR torsion levels; CH₄⁺ ground vibronic levels (Wörner, Qian & Merkt 2007).
