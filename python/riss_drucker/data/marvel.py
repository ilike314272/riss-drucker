"""MARVEL file readers (README 6.1).

Transitions (input) lines:  wavenumber  unc [unc2 ...]  QN'(n_qn)  QN''(n_qn)  tag
  (a negative wavenumber marks a line deleted by the MARVEL analysis: read as
  excluded=True with |wavenumber|, and skipped by network.invert)
Energies (output) lines:    QN(n_qn)  energy  unc  n_transitions
The number of quantum-number columns depends on the molecule (e.g. H2CO:
v1..v6 J Ka Kc = 9), so it is a parameter. Quantum numbers are kept as strings.
If several uncertainty columns are present, the last one is used (MARVEL's
updated uncertainty).

Checked against real files (H2(32)S, Chubb et al. 2018, Mendeley Data
10.17632/mx6skzhxp4.1): transitions "nu unc v1 v2 v3 J Ka Kc op v1 v2 v3 J Ka Kc op tag"
(n_qn = 7) and energies with a header line. Column layouts differ between
MARVEL papers; check n_qn for each new file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarvelTransition:
    wavenumber: float
    uncertainty: float
    upper: tuple
    lower: tuple
    tag: str
    excluded: bool = False  # MARVEL convention: a negative wavenumber marks a deleted line


def read_transitions(path, n_qn: int, *, comment: str = "#") -> list[MarvelTransition]:
    out = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split(comment, 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        n_unc = len(tok) - 1 - 2 * n_qn - 1
        if n_unc < 1:
            raise ValueError(f"{path}:{ln}: expected >= {2 + 2 * n_qn + 1} columns, got {len(tok)}")
        unc = float(tok[n_unc])  # last uncertainty column
        qu = tuple(tok[1 + n_unc: 1 + n_unc + n_qn])
        ql = tuple(tok[1 + n_unc + n_qn: 1 + n_unc + 2 * n_qn])
        nu = float(tok[0])
        out.append(MarvelTransition(abs(nu), unc, qu, ql, tok[-1], excluded=nu < 0))
    return out


def read_energies(path, n_qn: int, *, comment: str = "#") -> list[dict]:
    out = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split(comment, 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        if len(tok) < n_qn + 2:
            raise ValueError(f"{path}:{ln}: expected >= {n_qn + 2} columns")
        try:
            float(tok[n_qn])
        except ValueError:
            continue  # header line (e.g. "v1 v2 v3 J Ka Kc op Energy Unc NumTrans")
        out.append({"qn": tuple(tok[:n_qn]), "energy": float(tok[n_qn]), "uncertainty": float(tok[n_qn + 1]),
                    "n_transitions": int(tok[n_qn + 2]) if len(tok) > n_qn + 2 else None})
    return out


def write_transitions(path, transitions, *, fmt: str = "{:.6f}") -> None:
    lines = [" ".join([fmt.format(-t.wavenumber if t.excluded else t.wavenumber), f"{t.uncertainty:.2e}",
                       *t.upper, *t.lower, t.tag])
             for t in transitions]
    Path(path).write_text("\n".join(lines) + "\n")
