"""File boundary for ab initio point sets (JSON).

{"schema": "riss_drucker.pointset/1", "molecule": "H2O2", "method": "CCSD(T)", "basis": "cc-pVTZ",
 "program": "pyscf 2.x", "coordinate": "HOOH dihedral / deg", "units": {"energy": "hartree"},
 "points": [{"coordinate": 0.0, "energy": -151.3, "geometry": [["O", x, y, z], ...]}, ...],
 "provenance": {...}}
"""
from __future__ import annotations

import json
from pathlib import Path

from ..io_utils import atomic_write_json

SCHEMA = "riss_drucker.pointset/1"
HARTREE_TO_CM = 219474.6313705


def write(path, molecule: str, method: str, basis: str, points: list, *, program: str = "", coordinate: str = "",
          provenance: dict | None = None) -> Path:
    return atomic_write_json(path, {"schema": SCHEMA, "molecule": molecule, "method": method, "basis": basis,
                                    "program": program, "coordinate": coordinate, "units": {"energy": "hartree"},
                                    "points": points, "provenance": provenance or {}})


def read(path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    return d
