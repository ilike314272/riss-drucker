"""ExoMol readers (README 6.1): .def.json, .states, .trans (plain or .bz2).

The column layout of .states files is defined per dataset in its .def.json; the
reader takes field names from there (key "states_file_fields", found anywhere in
the JSON) or from an explicit `fields` list. Always present: id, E, gtot, J.
.trans files are streamed with filters (intensity cutoff, J range, band), since
they can hold billions of transitions.

NOTE: this container could not reach exomol.com; the readers are tested on
files written in the documented format, not on downloaded files. Check the
def.json key layout against a real file before relying on automatic fields.
"""
from __future__ import annotations

import bz2
import io
import json
import math
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

C2 = 1.438776877           # cm K
C_CM_PER_S = 29979245800.0


def _open(path) -> io.TextIOBase:
    path = Path(path)
    if path.suffix == ".bz2":
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="ascii")
    return open(path, encoding="ascii")


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def read_def(path) -> dict:
    d = json.loads(Path(path).read_text())
    fields = _find_key(d, "states_file_fields")
    names = [f["name"] if isinstance(f, dict) else str(f) for f in fields] if fields else None
    return {"raw": d, "states_fields": names}


DEFAULT_FIELDS = ["i", "E", "g_tot", "J"]


def read_states(path, fields: Sequence[str] | None = None, *, j_max: float | None = None,
                e_max: float | None = None) -> dict:
    """Return {field: array} (numeric where possible) for states passing the filters."""
    names = list(fields) if fields else None
    cols: dict[str, list] = {}
    with _open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if names is None:
                names = DEFAULT_FIELDS + [f"qn{k}" for k in range(len(tok) - 4)]
                cols = {n: [] for n in names}
            if not cols:
                cols = {n: [] for n in names}
            if len(tok) < len(names):
                raise ValueError(f"{path}: line has {len(tok)} columns, def lists {len(names)}: {line[:80]}")
            E, J = float(tok[1]), float(tok[3])
            if (j_max is not None and J > j_max) or (e_max is not None and E > e_max):
                continue
            for n, t in zip(names, tok):
                cols[n].append(t)
    out = {}
    for n, v in cols.items():
        try:
            arr = np.array(v, dtype=float)
            if n in ("i", "g_tot") or (arr.size and np.all(arr == np.round(arr)) and n == "i"):
                arr = arr.astype(np.int64)
            out[n] = arr
        except ValueError:
            out[n] = np.array(v, dtype=object)
    if "i" in out:
        out["i"] = out["i"].astype(np.int64)
    return out


def iter_trans(path, states: dict, *, T: float | None = None, Q: float | None = None,
               s_min: float | None = None, nu_range: tuple[float, float] | None = None,
               keep: Callable[[int, int], bool] | None = None) -> Iterator[dict]:
    """Stream transitions; optionally compute intensities S(T) [cm/molecule] and
    filter on S >= s_min, a wavenumber range, or a predicate on (upper, lower) ids."""
    index = {int(i): k for k, i in enumerate(states["i"])}
    E, g = states["E"], states["g_tot"]
    with _open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 3:
                continue
            u, l, A = int(tok[0]), int(tok[1]), float(tok[2])
            ku, kl = index.get(u), index.get(l)
            if ku is None or kl is None:
                continue  # filtered-out state
            nu = float(tok[3]) if len(tok) > 3 else float(E[ku] - E[kl])
            if nu <= 0:
                continue  # unresolved within the file's energy precision
            if nu_range and not (nu_range[0] <= nu <= nu_range[1]):
                continue
            if keep and not keep(u, l):
                continue
            rec = {"upper": u, "lower": l, "A": A, "nu": nu}
            if T is not None:
                if Q is None:
                    raise ValueError("intensities need the partition function Q")
                rec["S"] = intensity(A, g[ku], E[kl], nu, T, Q)
                if s_min is not None and rec["S"] < s_min:
                    continue
            yield rec


def intensity(A: float, g_up: float, E_low: float, nu: float, T: float, Q: float) -> float:
    """Absorption intensity S (cm/molecule) from an Einstein A coefficient."""
    return (g_up * A / (8 * math.pi * C_CM_PER_S * nu * nu) * math.exp(-C2 * E_low / T)
            * (1 - math.exp(-C2 * nu / T)) / Q)


def partition_function(states: dict, T: float) -> float:
    return float(np.sum(states["g_tot"] * np.exp(-C2 * states["E"] / T)))
