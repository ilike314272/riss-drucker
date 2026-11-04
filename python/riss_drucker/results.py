"""Result records in the format of schemas/result.schema.json (README 9)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .fit import FitResult
from .spec import ModelSpec

SCHEMA_VERSION = 1


def _num(x):
    return None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else float(x)


def fit_result_record(fit: FitResult, spec: ModelSpec, *, model=None, run_record: dict | None = None,
                      final_solution=None, windows: list | None = None, coverage: dict | None = None,
                      assignments=None, near_crossing_resampling: list | None = None) -> dict:
    certs = []
    if final_solution is not None:
        for b, r in zip(final_solution.blocks, final_solution.results):
            certs += [{"block": b["key"], **c} for c in r["certificates"]]
    else:
        certs = [{**c} for c in fit.failed_certificates]
    rec = {
        "schema_version": SCHEMA_VERSION,
        "model": {"spec": spec.canonical(), "spec_hash": spec.hash()},
        "parameters": {
            "names": fit.param_names, "values": [fit.theta[n] for n in fit.param_names],
            "sigma": [_num(fit.sigma[n]) for n in fit.param_names],
            "covariance": np.asarray(fit.covariance).tolist(), "free": fit.free, "fixed": fit.fixed,
            "gauge_null_directions": fit.null_directions, "gauge_constraints": fit.gauge_constraints,
            "units": "cm^-1", "time_index": None,
        },
        "fit_statistics": {
            "chi2": fit.chi2, "dof": fit.dof, "reduced_chi2": _num(fit.reduced_chi2), "rms": fit.rms,
            "n_obs": fit.n_obs, "rank": fit.rank, "singular_values": fit.singular_values,
            "iterations": fit.iterations, "converged": fit.converged,
        },
        "levels": [{k: v for k, v in lv.items() if k != "true_energy"} for lv in fit.levels],
        "certificates": certs,
        "certified": bool(fit.certified),
    }
    if windows is not None:
        by = {(w["block"], w["index"]): w for w in windows}
        for lv in rec["levels"]:
            w = by.get((lv["block"], lv["index"]))
            if w:
                lv.update(near_crossing=w["near_crossing"], window=list(w["window"]),
                          extrapolated=w["extrapolated"], beyond_weyl_fence=w["beyond_weyl_fence"])
        rec["windows"] = [{**w, "window": list(w["window"])} for w in windows]
    if near_crossing_resampling is not None:
        rec["near_crossing_resampling"] = [{**r, "quantiles": {str(k): v for k, v in r["quantiles"].items()}}
                                           for r in near_crossing_resampling]
    if coverage is not None:
        rec["coverage"] = coverage
    if assignments is not None:
        rec["assignments"] = {"lines": [{**r, "credible_set": [str(c) for c in r["credible_set"]]}
                                        for r in assignments.lines],
                              "expected_misassignments": assignments.expected_misassignments,
                              "settings": assignments.settings}
    if model is not None:
        rec["model"]["compile_certificates"] = list(model.compile_certificates)
    if run_record is not None:
        rec["run_record"] = run_record
    return rec


def schema_path() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent / "schemas" / "result.schema.json",
                 here.parents[2] / "schemas" / "result.schema.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError("result.schema.json not found")


def validate_result(record: dict) -> None:
    """Validate against the schema (needs the optional `jsonschema` package)."""
    import jsonschema

    jsonschema.validate(record, json.loads(schema_path().read_text()))
