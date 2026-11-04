"""Record helpers: unit and uncertainty conversion, validation against
schemas/record.schema.json, and the rule that effective-Hamiltonian-derived
data are not independent measurements (README 6.1, 13)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

C_CM_PER_S = 29979245800.0
MHZ_PER_CM = C_CM_PER_S / 1e6  # 1 cm^-1 = 29979.2458 MHz
SCHEMA_VERSION = 1


def to_cm(value: float, unit: str) -> float:
    if unit == "cm-1":
        return float(value)
    if unit == "MHz":
        return float(value) / MHZ_PER_CM
    if unit == "GHz":
        return float(value) * 1e3 / MHZ_PER_CM
    raise ValueError(f"unknown energy unit {unit!r}")


def to_one_sigma(unc: float | None, definition: str) -> float | None:
    if unc is None:
        return None
    if definition == "1sigma":
        return float(unc)
    if definition == "2sigma":
        return float(unc) / 2.0
    raise ValueError("uncertainty definition must be 1sigma or 2sigma to be used in a fit")


def normalized(record: dict) -> dict:
    """Copy with energies/uncertainties in cm^-1 and 1 sigma (constants too)."""
    r = copy.deepcopy(record)
    unit = r["units"]["energy"]
    definition = r.get("uncertainty_definition", "unknown")
    if "value" in r:
        r["value"] = to_cm(r["value"], unit)
        if r.get("uncertainty") is not None:
            r["uncertainty"] = to_cm(to_one_sigma(r["uncertainty"], definition), unit)
    if r.get("kind") == "constants":
        for c in r["constants"].values():
            c["value"] = to_cm(c["value"], unit)
            if c.get("uncertainty") is not None:
                c["uncertainty"] = to_cm(to_one_sigma(c["uncertainty"], definition), unit)
    r["units"]["energy"] = "cm-1"
    if definition in ("1sigma", "2sigma"):
        r["uncertainty_definition"] = "1sigma"
    return r


def _schema_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "schemas" / "record.schema.json", here.parents[1] / "schemas" / "record.schema.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError("record.schema.json not found")


def validate(record: dict) -> None:
    import jsonschema

    jsonschema.validate(record, json.loads(_schema_path().read_text()))


def usable_as_fit_input(record: dict, *, allow_ehamil_derived: bool = False) -> bool:
    if record.get("validation", {}).get("status") == "quarantined":
        return False
    if record["source"] == "ehamil_derived" and not allow_ehamil_derived:
        return False
    return True
