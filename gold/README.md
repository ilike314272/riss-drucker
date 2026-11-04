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

Status: **empty.** The candidate papers are listed in `docs/data_sources.md`;
the papers could not be retrieved from the build environment (network policy).
First entries planned: H₂CO MARVEL subset (ground-state rotational levels),
H₂S MARVEL subset, H₂O₂ far-IR torsion levels, CH₄⁺ ground vibronic levels
(Wörner, Qian & Merkt 2007).
