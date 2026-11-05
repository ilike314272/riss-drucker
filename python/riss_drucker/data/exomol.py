"""ExoMol readers (README 6.1): .def.json, .states, .trans (plain or .bz2).

The column layout of .states files is defined per dataset in its .def.json; the
reader takes field names from there (key "states_file_fields", found anywhere in
the JSON) or from an explicit `fields` list. Always present: id, E, gtot, J.
.trans files are streamed with filters (intensity cutoff, J range, band), since
they can hold billions of transitions.

Checked against a real file (H2CO AYTY, version 20230430): def.json lists
fields under dataset/states/states_file_fields with names "ID", "E", "gtot",
"J", "unc", "tau", ..., and recent lists carry "Auxiliary:SourceType"
(Ma = MARVEL, Ca = calculated, EH = effective Hamiltonian, IE = isotopologue
extrapolation) and "Auxiliary:Ecal" (calculated energy). The first four
columns are always exposed under the canonical names i, E, g_tot, J as well.
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
CANONICAL = ("i", "E", "g_tot", "J")


def read_states(path, fields: Sequence[str] | None = None, *, j_max: float | None = None,
                e_max: float | None = None, keep: Callable[[list[str]], bool] | None = None) -> dict:
    """Return {field: array} for states passing the filters.

    Columns keep their def.json names; the first four are also available as
    i, E, g_tot, J. `keep(tokens)` is a fast row predicate on the raw string
    tokens (e.g. select one vibrational state) applied before conversion.
    """
    names = list(fields) if fields else None
    rows: list[list[str]] = []
    with _open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if names is None:
                names = DEFAULT_FIELDS + [f"qn{k}" for k in range(len(tok) - 4)]
            if len(tok) < len(names):
                raise ValueError(f"{path}: line has {len(tok)} columns, def lists {len(names)}: {line[:80]}")
            if j_max is not None and float(tok[3]) > j_max:
                continue
            if e_max is not None and float(tok[1]) > e_max:
                continue
            if keep is not None and not keep(tok):
                continue
            rows.append(tok[: len(names)])
    out = {}
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    for n, v in zip(names, cols):
        try:
            out[n] = np.array(v, dtype=float)
        except ValueError:
            out[n] = np.array(v, dtype=object)
    for canon, n in zip(CANONICAL, names[:4]):
        out[canon] = out[n]
    out["i"] = np.asarray(out["i"], dtype=float).astype(np.int64)
    out["g_tot"] = np.asarray(out["g_tot"], dtype=float)
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
