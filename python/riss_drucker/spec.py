"""ModelSpec: the discrete structure of a model, canonically hashed (README 5.1).

The hash is the cache key. Any change to a field is a change of model
structure: a different hash, a recompiled operator library, and a model
comparison. Never change structure inside a fit loop.

    spec = ModelSpec.asym_rotor(j_max=20, reduction="A", axis="Ir", order=4)
    spec.hash()          # sha256 hex of the canonical JSON
    model = spec.compile()           # or riss_drucker.cache.compile_cached(spec)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from . import _core

# Bump when the definition of any operator changes (sign, normalization,
# basis phase). It is part of every hash, so stale caches are never reused.
OPERATOR_LIBRARY_VERSION = "1"
CONVENTIONS = "docs/conventions.md (v1)"

_KINDS = ("asym_rotor", "torsion_rotor")


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if obj.is_integer():
            return int(obj)
        return repr(obj)  # exact round-trip text, platform independent
    raise TypeError(f"ModelSpec fields must be JSON-like, got {type(obj).__name__}")


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    structure: dict = field(default_factory=dict)
    group_chain: tuple = ()
    # Reserved for later phases (README 1): extra electronic states (Phase B).
    electronic_states: tuple = ("ground",)
    conventions: str = CONVENTIONS

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}; known: {_KINDS}")

    # -- constructors ---------------------------------------------------------
    @classmethod
    def asym_rotor(cls, *, j_max: int, j_min: int = 0, reduction: str = "A", axis: str = "Ir",
                   order: int = 4, extra_operators=()) -> "ModelSpec":
        return cls("asym_rotor", {"j_min": int(j_min), "j_max": int(j_max), "reduction": reduction,
                                  "axis": axis, "order": int(order),
                                  "extra_operators": sorted(extra_operators)}, group_chain=("D2",))

    @classmethod
    def torsion_rotor(cls, *, n_fold: int = 3, v_orders=(1,), m_max: int = 30, j_max: int = 0,
                      kp_coupling: bool = False) -> "ModelSpec":
        return cls("torsion_rotor", {"n_fold": int(n_fold), "v_orders": sorted(set(int(v) for v in v_orders)),
                                     "m_max": int(m_max), "j_max": int(j_max), "kp_coupling": bool(kp_coupling)},
                   group_chain=(f"C{int(n_fold)}v(M)",))

    # -- canonical form and hash ---------------------------------------------
    def canonical(self) -> dict:
        return _canonical({
            "kind": self.kind, "structure": self.structure, "group_chain": list(self.group_chain),
            "electronic_states": list(self.electronic_states), "conventions": self.conventions,
            "operator_library_version": OPERATOR_LIBRARY_VERSION,
        })

    def canonical_json(self) -> str:
        return json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_canonical(cls, d: dict) -> "ModelSpec":
        if d.get("operator_library_version") != OPERATOR_LIBRARY_VERSION:
            raise ValueError("spec was written by a different operator library version")
        return cls(d["kind"], dict(d["structure"]), tuple(d["group_chain"]), tuple(d["electronic_states"]),
                   d["conventions"])

    # -- compile -----------------------------------------------------------------
    def compile(self):
        if len(self.electronic_states) != 1:
            # TODO(riss-drucker-v1): multi-surface models (Phase B).
            raise NotImplementedError("multiple electronic states are a later phase")
        s = self.structure
        if self.kind == "asym_rotor":
            return _core.compile_asym_rotor(j_min=s["j_min"], j_max=s["j_max"], reduction=s["reduction"],
                                            axis=s["axis"], order=s["order"],
                                            extra_operators=list(s["extra_operators"]))
        return _core.compile_torsion_rotor(n_fold=s["n_fold"], v_orders=list(s["v_orders"]),
                                           m_max=s["m_max"], j_max=s["j_max"], kp_coupling=s["kp_coupling"])
