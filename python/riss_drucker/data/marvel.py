"""MARVEL file readers (README 6.1).

Transitions (input) lines:  wavenumber  unc [unc2 ...]  QN'(n_qn)  QN''(n_qn)  tag
Energies (output) lines:    QN(n_qn)  energy  unc  n_transitions
The number of quantum-number columns depends on the molecule (e.g. H2CO:
v1..v6 J Ka Kc = 9), so it is a parameter. Quantum numbers are kept as strings.
If several uncertainty columns are present, the last one is used (MARVEL's
updated uncertainty).

NOTE: tested on files written in this layout; check the column order of a real
supplementary file (it varies between MARVEL papers) and pass `layout` if needed.
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
        out.append(MarvelTransition(float(tok[0]), unc, qu, ql, tok[-1]))
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
        out.append({"qn": tuple(tok[:n_qn]), "energy": float(tok[n_qn]), "uncertainty": float(tok[n_qn + 1]),
                    "n_transitions": int(tok[n_qn + 2]) if len(tok) > n_qn + 2 else None})
    return out


def write_transitions(path, transitions, *, fmt: str = "{:.6f}") -> None:
    lines = [" ".join([fmt.format(t.wavenumber), f"{t.uncertainty:.2e}", *t.upper, *t.lower, t.tag])
             for t in transitions]
    Path(path).write_text("\n".join(lines) + "\n")
