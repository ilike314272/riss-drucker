"""Round-trip validation of extracted constants (README 6.3, mandatory).

Compile the record's model, plug in its constants (after unit and sign
conversion through the record's convention metadata), and reproduce the
paper's own levels or lines. If they do not reproduce, the extraction or the
convention is wrong and the record is quarantined, not ingested. This catches
OCR errors, sign flips and unit mistakes.

convention.name_map maps paper names to library parameters:
    {"DJ": "DeltaJ"}  or  {"DJ": {"param": "DeltaJ", "scale": -1.0}}
Unmapped library parameters are set to 0 and listed in the report.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .records import normalized


def _theta(record: dict, param_names) -> tuple[dict, list[str]]:
    rec = normalized(record)
    name_map = rec.get("convention", {}).get("name_map", {})
    theta = {p: 0.0 for p in param_names}
    unmapped = []
    for paper_name, c in rec["constants"].items():
        target = name_map.get(paper_name, paper_name)
        scale = 1.0
        if isinstance(target, dict):
            scale = float(target.get("scale", 1.0))
            target = target["param"]
        if target not in theta:
            unmapped.append(paper_name)
            continue
        theta[target] = scale * c["value"]
    def target_of(n):
        t = name_map.get(n, n)
        return t["param"] if isinstance(t, dict) else t

    mapped = {target_of(n) for n in rec["constants"]}
    missing = [p for p in param_names if p not in mapped]
    return theta, unmapped + [f"(zero) {m}" for m in missing]


def validate_constants(record: dict, references: list[dict], *, model=None, spec=None,
                       rms_factor: float = 2.0, outlier_sigma: float = 5.0) -> dict:
    """Return a copy of `record` with validation = validated | quarantined."""
    from ..rotor import label_index
    from ..solve import solve
    from ..spec import ModelSpec

    rec = copy.deepcopy(record)
    if rec.get("kind") != "constants":
        raise ValueError("round-trip validation applies to constants records")
    refs = [normalized(r) for r in references]
    if model is None:
        if spec is None:
            if "model_spec" not in rec:
                raise ValueError("record has no model_spec; pass spec= or model=")
            spec = ModelSpec.from_canonical(rec["model_spec"])
        model = spec.compile()
    theta, notes = _theta(rec, list(model.param_names))
    idx = label_index(model)
    need = {}
    rows = []
    for r in refs:
        if r["kind"] == "level":
            parts = [(idx[r["assignment"]["label"]], +1)]
        elif r["kind"] == "transition":
            parts = [(idx[r["upper"]["label"]], +1), (idx[r["lower"]["label"]], -1)]
        else:
            continue
        for (blk, i), _ in parts:
            need[blk] = max(need.get(blk, 0), i + 1)
        rows.append((r, parts))
    if not rows:
        raise ValueError("no level or transition references to compare with")
    keys = [b["key"] for b in model.blocks]
    sub_idx = sorted(keys.index(k) for k in need)
    sub = model.subset(sub_idx)
    sol = solve(sub, theta, nlowest=max(need.values()), method="dense")
    E = {b["key"]: r["eigenvalues"] for b, r in zip(sol.blocks, sol.results)}
    resid, sig = [], []
    for r, parts in rows:
        calc = sum(sgn * E[blk][i] for (blk, i), sgn in parts)
        resid.append(r["value"] - calc)
        sig.append(r.get("uncertainty") or float("nan"))
    resid, sig = np.asarray(resid), np.asarray(sig)
    rms = float(np.sqrt(np.mean(resid**2)))
    paper_rms = (normalized(rec).get("fit_statistics") or {}).get("rms")
    if paper_rms is not None and rec["units"]["energy"] != "cm-1":
        from .records import to_cm
        paper_rms = to_cm(paper_rms, rec["units"]["energy"])
    scale = paper_rms if paper_rms else float(np.nanmedian(sig))
    reasons = []
    if not sol.certified:
        reasons.append("solve not certified")
    if not np.isfinite(scale) or scale <= 0:
        reasons.append("no paper rms or reference uncertainties to compare against")
    else:
        if rms > rms_factor * scale:
            reasons.append(f"rms {rms:.3e} > {rms_factor} x reference scale {scale:.3e}")
        # Individual references may sit a few times the scale away (model limits),
        # never far beyond it: a sign or unit error produces large outliers.
        ref = max(scale, float(np.nanmedian(sig)) if np.isfinite(sig).any() else scale)
        worst = float(np.max(np.abs(resid)) / ref)
        if worst > outlier_sigma:
            reasons.append(f"worst residual {worst:.1f} x reference scale")
    status = "quarantined" if reasons else "validated"
    rec["validation"] = {"status": status, "round_trip_rms": rms,
                         "detail": "; ".join(reasons + ([f"notes: {', '.join(notes)}"] if notes else [])) or "ok"}
    return rec
