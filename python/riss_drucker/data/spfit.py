"""Pickett SPFIT/SPCAT files (.par/.var parameters, .egy energies), as published
by CDMS/JPL, mapped onto the library's Watson asymmetric-rotor parameters.

Checked against CDMS H2CO v3 (Mueller & Lewen 2017): with these maps our
S-reduction octic Hamiltonian reproduces SPCAT's .egy levels to the print
precision (5e-7 cm^-1) for J <= 40. Note: SPCAT truncates the K basis at KNMAX
(option line); for J > KNMAX its energies depend on that truncation. Our basis
is complete, so compare only J <= KNMAX, or impose the same truncation.

Parameter coding (ground state, Watson; value in MHz):
  1099 A-(B+C)/2   199 (B+C)/2   40099 (B-C)/4   [A-reduction style 10000/20000/30000 = A/B/C]
  2099 -DK  1199 -DJK  299 -DJ    40199 d1   50099 d2   (S reduction)
  3099 HK   2199 HKJ   1299 HJK   399 HJ   40299 h1  50199 h2  60099 h3
  4099 LK   3199 LKKJ  2299 LJK   1399 LJJK  499 LJ   40399 l1  50299 l2  60199 l3  70099 l4
Only the S-reduction map is verified here (CDMS H2CO); the A-reduction IDs are
not mapped until a verified example exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Watson S reduction, representation from IAX (|IAX| = 1 -> z = a, Ir).
# Each SPFIT id maps to {library parameter: coefficient}: paper value = sum coeff * ours.
S_REDUCTION_MAP = {
    1099: {"A": 1.0, "B": -0.5, "C": -0.5},
    199: {"B": 0.5, "C": 0.5},
    40099: {"B": 0.25, "C": -0.25},
    10000: {"A": 1.0}, 20000: {"B": 1.0}, 30000: {"C": 1.0},
    299: {"DJ": -1.0}, 1199: {"DJK": -1.0}, 2099: {"DK": -1.0},
    40199: {"d1": 1.0}, 50099: {"d2": 1.0},
    399: {"HJ": 1.0}, 1299: {"HJK": 1.0}, 2199: {"HKJ": 1.0}, 3099: {"HK": 1.0},
    40299: {"h1": 1.0}, 50199: {"h2": 1.0}, 60099: {"h3": 1.0},
    499: {"LJ": 1.0}, 1399: {"LJJK": 1.0}, 2299: {"LJK": 1.0}, 3199: {"LKKJ": 1.0}, 4099: {"LK": 1.0},
    40399: {"l1": 1.0}, 50299: {"l2": 1.0}, 60199: {"l3": 1.0}, 70099: {"l4": 1.0},
}


@dataclass
class SpfitParameter:
    id: int
    value: float
    error: float
    label: str


@dataclass
class SpfitFile:
    title: str
    reduction: str        # "A" or "S"
    iax: int              # SPFIT axis option (1: z = a, Ir)
    knmax: int            # K truncation of the basis used by SPCAT
    parameters: list


def read_var(path) -> SpfitFile:
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    title = lines[0].strip()
    opt = lines[2].split()
    reduction = "S" if opt[0].lower().startswith("s") else "A"
    knmax = int(opt[4]) if len(opt) > 4 else -1
    iax = int(opt[6]) if len(opt) > 6 else 1
    params = []
    for line in lines[3:]:
        m = re.match(r"\s*(-?\d+)\s+(\S+)\s+(\S+)\s*(?:/(.*))?$", line)
        if not m:
            break
        try:
            val = float(m.group(2).replace("D", "E"))
            err = float(m.group(3).replace("D", "E"))
        except ValueError:
            break
        params.append(SpfitParameter(int(m.group(1)), val, err, (m.group(4) or "").strip()))
    return SpfitFile(title, reduction, iax, knmax, params)


def read_egy(path) -> dict:
    """Levels "J(Ka,Kc)" -> energy (cm^-1) from an SPCAT .egy file (single state)."""
    out = {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if ":" not in line:
            continue
        head, qn = line.split(":", 1)
        q = qn.split()
        t = head.split()
        out[f"{q[0]}({q[1]},{q[2]})"] = float(t[2])
    return out


def constants_record(var: SpfitFile, *, citation: str, doi: str | None, url: str, sha256: str,
                     formula: str, extra_provenance: dict | None = None) -> dict:
    """Record (schemas/record.schema.json) of the rotational constants, with the
    SPFIT ids as names and a linear name_map onto the library parameters.
    Parameters without a verified mapping (e.g. decic, hyperfine) are listed in
    convention.unmapped and left out of the round trip."""
    if var.reduction != "S" or abs(var.iax) != 1:
        raise NotImplementedError("only the S reduction with IAX = 1 (Ir) is verified")
    consts, name_map, unmapped = {}, {}, []
    for p in var.parameters:
        key = str(p.id)
        if p.id in S_REDUCTION_MAP:
            consts[key] = {"value": p.value, "uncertainty": p.error, "fixed": False}
            name_map[key] = {"linear": S_REDUCTION_MAP[p.id]}
        else:
            unmapped.append(f"{p.id} {p.label} = {p.value:.6g}")
    return {
        "schema_version": 1, "kind": "constants", "molecule": {"formula": formula}, "source": "measured",
        "provenance": {"citation": citation, "doi": doi, "file": url, "sha256": sha256,
                       "extraction_method": "parser", "extraction_confidence": 1.0, **(extra_provenance or {})},
        "units": {"energy": "MHz"}, "uncertainty_definition": "1sigma", "constants": consts,
        "convention": {"model": "asym_rotor", "reduction": "S", "representation": "Ir",
                       "operator_convention": "SPFIT Watson S (Pickett); docs/conventions.md#9",
                       "name_map": name_map, "unmapped": unmapped, "knmax": var.knmax},
    }
