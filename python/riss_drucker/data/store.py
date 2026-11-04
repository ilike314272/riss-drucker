"""A small append-only record store (JSON lines) with a quarantine.

    store = RecordStore(path)
    store.add(record)                  # schema-validated, provenance required
    store.quarantine(record, reason)
    store.fit_observations(model, formula="H2CO")   # refuses ehamil_derived by default
"""
from __future__ import annotations

import json
from pathlib import Path

from ..io_utils import atomic_write_bytes
from .records import normalized, usable_as_fit_input, validate


class RecordStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_file = self.path / "records.jsonl"
        self.quarantine_file = self.path / "quarantine.jsonl"

    def _append(self, file: Path, rec: dict) -> None:
        old = file.read_bytes() if file.exists() else b""
        atomic_write_bytes(file, old + (json.dumps(rec, sort_keys=True) + "\n").encode())

    def add(self, record: dict, *, check_schema: bool = True) -> None:
        if check_schema:
            validate(record)
        if record.get("validation", {}).get("status") == "quarantined":
            self._append(self.quarantine_file, record)
        else:
            self._append(self.records_file, record)

    def quarantine(self, record: dict, reason: str) -> None:
        rec = {**record, "validation": {"status": "quarantined", "detail": reason}}
        self._append(self.quarantine_file, rec)

    def records(self, *, formula: str | None = None, kind: str | None = None, quarantined: bool = False):
        file = self.quarantine_file if quarantined else self.records_file
        if not file.exists():
            return []
        out = [json.loads(l) for l in file.read_text().splitlines() if l.strip()]
        return [r for r in out if (formula is None or r["molecule"]["formula"] == formula)
                and (kind is None or r["kind"] == kind)]

    def fit_observations(self, model, *, formula: str, allow_ehamil_derived: bool = False):
        """Level/transition records -> fit Observations (labels must be J(Ka,Kc))."""
        from ..fit import Observation
        from ..rotor import label_index

        idx = label_index(model)
        obs, refused = [], 0
        for r in self.records(formula=formula):
            if r["kind"] not in ("level", "transition"):
                continue
            if not usable_as_fit_input(r, allow_ehamil_derived=allow_ehamil_derived):
                refused += 1
                continue
            n = normalized(r)
            if n.get("uncertainty") in (None, 0):
                refused += 1
                continue
            if r["kind"] == "level":
                b, i = idx[n["assignment"]["label"]]
                obs.append(Observation(b, i, n["value"], n["uncertainty"], label=n["assignment"]["label"]))
            else:
                (ub, ui), (lb, li) = idx[n["upper"]["label"]], idx[n["lower"]["label"]]
                obs.append(Observation(ub, ui, n["value"], n["uncertainty"],
                                       label=f"{n['upper']['label']}<-{n['lower']['label']}",
                                       lower_block=lb, lower_index=li))
        self.last_refused = refused
        return obs
