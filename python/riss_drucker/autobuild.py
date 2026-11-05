"""Automatic construction of effective Hamiltonians from data (README 7).

    spec0 (spaces, group, zeroth-order terms)
      -> compile (certified) -> fit -> evidence + diagnostics
      -> proposers (operator dictionary / amortized network / language model)
      -> filters: grammar + compile certificates, linear independence,
                  contact-transformation redundancy
      -> trial fits scored by Laplace evidence -> accept the best, repeat.

Nothing a proposer says is trusted: every candidate is compiled, certified and
fitted before it is kept. Results carry the Fisher-rank-vs-redundancy certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import algebra
from .build import BuildSpec, BuiltModel, GrammarError, compile_build
from .fit import FitResult, Observation, fit
from .selection import laplace_log_evidence
from .solve import solve


@dataclass
class LevelDatum:
    J: float
    irrep: str
    index: int
    energy: float
    sigma: float
    polyad: int | None = None
    label: str = ""


def observations_for(bm: BuiltModel, data: list[LevelDatum]) -> list[Observation]:
    keys = {}
    for b, blk in zip(bm.model.blocks, bm.blocks):
        keys[(blk.two_j, blk.irrep, blk.polyad)] = b["key"]
    out = []
    for d in data:
        k = keys.get((round(2 * d.J), d.irrep, d.polyad))
        if k is None:
            raise GrammarError(f"no block for J={d.J} {d.irrep} P={d.polyad}")
        out.append(Observation(k, int(d.index), float(d.energy), float(d.sigma), label=d.label))
    return out


def hf_jacobian(bm: BuiltModel, theta, obs: list[Observation]) -> np.ndarray:
    """Hellmann-Feynman derivatives dE/dtheta_k for each observed level."""
    sol = solve(bm.model, theta, window=(-1e12, 1e12), method="dense", keep_vectors=True)
    key_to_b = {b["key"]: i for i, b in enumerate(bm.model.blocks)}
    cache = {}
    rows = []
    for o in obs:
        b = key_to_b[o.block]
        if b not in cache:
            cache[b] = np.asarray(bm.model.jacobian(b, np.asarray(sol.results[b]["eigenvectors"])))
        rows.append(cache[b][o.index])
    return np.array(rows)


def residual_diagnostics(bm: BuiltModel, data: list[LevelDatum], fr: FitResult) -> dict:
    res = np.array([l["residual"] for l in fr.levels])
    sig = np.array([d.sigma for d in data])
    groups = {}
    for d, r in zip(data, res):
        groups.setdefault((d.irrep, d.polyad), []).append((d.J, r))
    by_block = []
    for (irr, P), v in sorted(groups.items(), key=lambda kv: str(kv[0])):
        Jv = np.array([x for x, _ in v])
        rv = np.array([r for _, r in v])
        X = Jv * (Jv + 1)
        slope = float(np.polyfit(X, rv, 1)[0]) if len(set(X)) > 1 else 0.0
        by_block.append({"irrep": irr, "polyad": P, "n": len(v), "mean": float(rv.mean()),
                         "rms": float(np.sqrt(np.mean(rv**2))), "slope_vs_J(J+1)": slope})
    worst = np.argsort(-np.abs(res / sig))[:8]
    return {"chi2_per_dof": fr.reduced_chi2, "rms": fr.rms, "by_block": by_block,
            "worst": [{"J": data[i].J, "irrep": data[i].irrep, "index": data[i].index, "polyad": data[i].polyad,
                       "residual": float(res[i]), "z": float(res[i] / sig[i])} for i in worst]}


def coupling_diagnostic(bm: BuiltModel, theta, labels: dict, *, max_states: int = 200) -> dict:
    """Which candidate quantum numbers are good: for each label operator L (grammar text),
    the spread <L^2> - <L>^2 of each eigenstate relative to the spread of L's own
    eigenvalues. Near 0: L is a good label (e.g. K^2 -> near-prolate, Sigma -> Hund (a),
    N^2 -> Hund (b), polyad -> resonance-free)."""
    out = {}
    vecs = bm.eig(theta)
    for name, txt in labels.items():
        Lb = bm.operator_blocks(txt)
        L2b = bm.operator_blocks(f"({txt}) ({txt})")
        rel, n = [], 0
        for (w, V), L, L2 in zip(vecs, Lb, L2b):
            evals = np.linalg.eigvalsh((L + L.conj().T) / 2)
            spread = max(np.ptp(evals), 1e-12)
            for j in range(min(V.shape[1], max_states)):
                v = V[:, j]
                m = float(np.real(v @ L @ v))
                m2 = float(np.real(v @ L2 @ v))
                rel.append(math.sqrt(max(m2 - m * m, 0.0)) / spread)
                n += 1
        rel = np.array(rel)
        out[name] = {"median_relative_spread": float(np.median(rel)) if n else float("nan"),
                     "fraction_good": float(np.mean(rel < 0.05)) if n else float("nan"), "n_states": n}
    return out


@dataclass
class BuildStep:
    term: str
    source: str
    accepted: bool
    delta_log_evidence: float
    chi2_before: float
    chi2_after: float
    reason: str


@dataclass
class BuildResult:
    spec: BuildSpec
    model: BuiltModel
    fit: FitResult
    log_evidence: float
    steps: list
    certificates: list
    diagnostics: dict
    redundancy: dict | None = None


class DictionaryProposer:
    """Every symmetry-allowed operator of the given multidegrees not yet in the model
    (counts certified against characters)."""

    def __init__(self, degrees_fn, name: str = "dictionary"):
        self.degrees_fn, self.name = degrees_fn, name

    def propose(self, context: dict) -> list[str]:
        spec = context["spec"]
        degs = self.degrees_fn(spec) if callable(self.degrees_fn) else self.degrees_fn
        return algebra.enumerate_terms(spec, degs, exclude=list(spec.terms), prefix="d").terms


def _rename(term: str, used: set, base: str) -> str:
    name, expr = term.split(":", 1)
    name = name.strip()
    if name in used:
        k = 1
        while f"{base}{k}" in used:
            k += 1
        name = f"{base}{k}"
    used.add(name)
    return f"{name}:{expr}"


def autobuild(spec0: BuildSpec, data: list[LevelDatum], theta0: dict, proposers: list, *, max_steps: int = 8,
              prior_sigma: float = 1e3, min_gain: float = 3.0, trial_top: int = 12, start_scale: float = 1e-2,
              redundancy_degrees=None, model_error: float = 0.0, verbose: bool = False,
              fit_kw: dict | None = None, new_term_sigma: float | None = None) -> BuildResult:
    """prior_sigma: prior width of the starting parameters; new_term_sigma: prior width of
    every proposed term (its Occam factor in the evidence; default prior_sigma)."""
    fit_kw = dict(fit_kw or {})
    nts = prior_sigma if new_term_sigma is None else new_term_sigma
    base_names = set(theta0)

    def ps(fres):
        return {n: (prior_sigma if n in base_names else nts) for n in fres.free}
    spec = spec0
    theta = dict(theta0)
    bm = compile_build(spec)
    obs = observations_for(bm, data)
    fr = fit(bm.model, obs, theta, model_error=model_error, method="dense", **fit_kw)
    ev = laplace_log_evidence(fr, ps(fr))
    steps, rejected = [], set()
    for _ in range(max_steps):
        diag = residual_diagnostics(bm, data, fr)
        ctx = {"spec": spec, "model": bm, "fit": fr, "diagnostics": diag, "terms": list(spec.terms), "data": data,
               "theta": dict(fr.theta)}
        cands = []
        for p in proposers:
            try:
                for t in p.propose(ctx):
                    cands.append((t, getattr(p, "name", type(p).__name__)))
            except Exception as exc:  # a proposer failing is reported, not fatal
                steps.append(BuildStep("", getattr(p, "name", "?"), False, 0.0, fr.chi2, fr.chi2,
                                       f"proposer error: {exc}"))
        used = set(spec.terms[i].split(":", 1)[0].strip() for i in range(len(spec.terms)))
        scored = []
        J0 = None
        for text, src in cands:
            expr = text.split(":", 1)[1].strip() if ":" in text else text
            if expr in rejected:
                continue
            term = _rename(text, set(used), "x")
            try:
                bmc = compile_build(spec.with_terms(list(spec.terms) + [term]))
            except Exception as exc:
                rejected.add(expr)
                steps.append(BuildStep(term, src, False, 0.0, fr.chi2, fr.chi2, f"rejected by compiler: {exc}"))
                continue
            if redundancy_degrees is not None:
                red = algebra.redundancy(bmc, {**fr.theta, term.split(":")[0].strip(): 0.0}, redundancy_degrees)
                red0 = algebra.redundancy(bm, fr.theta, redundancy_degrees)
                if red["identifiable_max"] <= red0["identifiable_max"]:
                    rejected.add(expr)
                    steps.append(BuildStep(term, src, False, 0.0, fr.chi2, fr.chi2,
                                           "removable by a contact transformation (no new identifiable direction)"))
                    continue
            name = term.split(":")[0].strip()
            th_c = {**fr.theta, name: 0.0}
            obs_c = observations_for(bmc, data)
            Jac = hf_jacobian(bmc, th_c, obs_c)
            sig = np.sqrt(np.array([o.sigma for o in obs_c]) ** 2 + model_error**2)
            r = np.array([l["residual"] for l in fr.levels]) / sig
            col = Jac[:, bmc.param_names.index(name)] / sig
            others = np.delete(Jac, bmc.param_names.index(name), axis=1) / sig[:, None]
            q, _ = np.linalg.qr(others)
            col_perp = col - q @ (q.T @ col)
            nrm = float(np.linalg.norm(col_perp))
            score = float((col_perp @ r) ** 2 / nrm**2) if nrm > 1e-12 else 0.0
            scored.append((score, nrm, term, src, bmc))
        if not scored:
            break
        # score test first; terms with no first-order derivative (resonances) still get trial fits
        scored.sort(key=lambda x: -x[0])
        trial = scored[:trial_top] + [s for s in scored[trial_top:] if s[1] < 1e-9][:trial_top]
        best = None
        for score, nrm, term, src, bmc in trial:
            name = term.split(":")[0].strip()
            obs_c = observations_for(bmc, data)
            results = []
            starts = [0.0] if nrm > 1e-9 else [start_scale * nts, -start_scale * nts]
            for st in starts:
                try:
                    f2 = fit(bmc.model, obs_c, {**fr.theta, name: st}, model_error=model_error,
                             method="dense", **fit_kw)
                    results.append(f2)
                except Exception:
                    continue
            if not results:
                continue
            f2 = min(results, key=lambda f: f.chi2)
            ev2 = laplace_log_evidence(f2, ps(f2))
            if verbose:
                print(f"  {term:50s} score {score:9.2f}  dlogZ {ev2 - ev:9.2f}")
            if best is None or ev2 > best[0]:
                best = (ev2, term, src, bmc, f2)
        if best is None or best[0] - ev < min_gain:
            if best is not None:
                steps.append(BuildStep(best[1], best[2], False, best[0] - ev, fr.chi2, best[4].chi2,
                                       f"best candidate below evidence threshold {min_gain}"))
            break
        ev2, term, src, bmc, f2 = best
        steps.append(BuildStep(term, src, True, ev2 - ev, fr.chi2, f2.chi2, "accepted on evidence"))
        spec, bm, fr, ev = bmc.spec, bmc, f2, ev2
    certs = list(bm.certificates)
    red = None
    if redundancy_degrees is not None:
        red = algebra.redundancy(bm, fr.theta, redundancy_degrees)
        certs.append(algebra.fisher_rank_certificate(fr, red))
    return BuildResult(spec, bm, fr, ev, steps, certs, residual_diagnostics(bm, data, fr), red)
