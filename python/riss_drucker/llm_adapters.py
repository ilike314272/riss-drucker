"""Molecule-independent adapters for riss_drucker.llm_loop.

LevelsAdapter: any molecule (riss_drucker.molecule.Molecule) and any list of target levels
{J, irrep, E} - from a variational ab initio calculation, a catalogue or experiment. The language
model proposes a BuildSpec (spaces, group, terms); the library compiles and certifies it, fits its
coefficients to the target levels with J <= jfit and scores it by the rms of the predicted levels
with J > jfit (extrapolation, so extra terms must earn their place). The molecule's certified
zeroth-order plan (`Molecule.plan`) is the reference.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .llm_loop import strip_private

ANSWER = """Answer with exactly one JSON object (in a ```json block):
{"spec": {"spaces": [...], "terms": ["<param>: <expr>", ...]},   (the group is the molecule's, from its geometry)
 "initial": {"<param>": <starting value>, ...},
 "rationale": "<short physical justification>"}
Spaces as in the reference spec ({"kind": "rot"}, {"kind": "vib", ...}, ...). Coefficients are fitted to the
target levels with J <= jfit; the score is the rms of the predicted levels with J > jfit (cm-1)."""


def _group_name(spec_dict):
    g = spec_dict["group"]
    return g if isinstance(g, str) else g.get("name", "custom")


class LevelsAdapter:
    system = ("You are an expert molecular spectroscopist building effective Hamiltonians. Reason about symmetry, "
              "then answer only with the requested JSON object.")

    def __init__(self, molecule, target: list, *, jfit: float, plan_kw: dict | None = None):
        from .build import BuildSpec
        self.mol, self.jfit = molecule, jfit
        self.J = sorted({float(t["J"]) for t in target})
        self.target = defaultdict(list)
        for t in target:
            self.target[(float(t["J"]), t["irrep"])].append(float(t["E"]))
        for k in self.target:
            self.target[k].sort()
        kw = dict(plan_kw or {})
        kw.setdefault("J", [j if j % 1 else int(j) for j in self.J])
        self.plan = molecule.plan(**kw)
        self._group = self.plan["spec"].group
        self.BuildSpec = BuildSpec

    # -- adapter interface
    def reference(self):
        d = self.plan["spec"].to_dict()
        return {"spec": {"group": getattr(self._group, "name", "molecule"), "spaces": d["spaces"], "terms": d["terms"]},
                "initial": self.plan["theta"], "rationale": "certified zeroth-order plan (Molecule.plan)"}

    def describe(self, prop):
        sp = prop.get("spec", {})
        lines = [f"group {_group_name(sp) if isinstance(sp.get('group'), (str, dict)) else 'plan group'}",
                 *[f"space: {s.get('kind')}" for s in sp.get("spaces", [])],
                 *[f"term {t}" for t in sp.get("terms", [])]]
        if prop.get("rationale"):
            lines.append("reasoning: " + str(prop["rationale"])[:600])
        return "\n".join(lines)

    def prompt(self, history):
        from .llm import GRAMMAR
        an = self.mol.analyse()
        ref = self.reference()
        counts = defaultdict(int)
        for (J, ir), v in self.target.items():
            counts[f"J={J:g} {ir}"] += len(v)
        sample = sorted(((J, ir, e) for (J, ir), v in self.target.items() for e in v), key=lambda t: t[2])[:30]
        payload = {"task": "Build an effective Hamiltonian that reproduces the target levels of this molecule; it is "
                           f"fitted to J <= {self.jfit:g} and scored on J > {self.jfit:g}.",
                   "molecule": {k: an[k] for k in an if k != "character_table"},
                   "character_table": an.get("character_table"),
                   "reference_spec": {"spaces": ref["spec"]["spaces"], "terms": ref["spec"]["terms"]},
                   "target_level_counts": dict(counts), "lowest_target_levels": [
                       {"J": J, "irrep": ir, "E": round(e, 6)} for J, ir, e in sample],
                   "grammar": GRAMMAR, "previous_attempts_and_feedback": history, "answer_format": ANSWER}
        return json.dumps(payload, indent=1, default=str)

    def _compile(self, prop):
        """The group is the molecule's (from its geometry); the proposal chooses spaces and terms."""
        from .build import BuildSpec, _space_from_dict, compile_build
        sp = dict(prop["spec"])
        spec = BuildSpec([_space_from_dict(s) for s in sp["spaces"]], self._group, list(sp["terms"]),
                         [j if j % 1 else int(j) for j in self.J])
        return spec, compile_build(spec)

    def _levels(self, bm, theta):
        out = defaultdict(list)
        for b, (w, _) in zip(bm.blocks, bm.eig(theta)):
            out[(b.two_j / 2, b.irrep)].extend(w.tolist())
        return {k: sorted(v) for k, v in out.items()}

    def evaluate(self, prop, log=print):
        from scipy.optimize import least_squares
        spec, bm = self._compile(prop)
        params = [t.split(":")[0].strip() for t in spec.terms]
        keys = sorted(self.target)
        lv = self._levels(bm, {p: 0.0 for p in params})
        missing = {f"J={J:g} {ir}": (len(lv.get((J, ir), [])), len(self.target[(J, ir)])) for J, ir in keys
                   if len(lv.get((J, ir), [])) < len(self.target[(J, ir)])}
        if missing:
            raise ValueError(f"model has fewer levels than the target in (model, target): {dict(list(missing.items())[:8])}")
        init = dict(self.plan["theta"])
        init.update(prop.get("initial") or {})
        x0 = np.array([float(init.get(p, 0.0)) for p in params])

        def resid(x, sel):
            lv = self._levels(bm, dict(zip(params, x)))
            r = []
            for k in keys:
                if sel(k[0]):
                    t = self.target[k]
                    r.extend(np.array(lv[k][:len(t)]) - np.array(t))
            return np.array(r)
        fit = lambda J: J <= self.jfit
        pred = lambda J: J > self.jfit
        # common energy zero: the lowest target level is 0; fit an offset as well
        def resid_off(x, sel):
            r = resid(x[:-1], sel)
            return r + x[-1]
        sol = least_squares(resid_off, np.r_[x0, 0.0], args=(fit,), x_scale=np.r_[np.maximum(np.abs(x0), 1e-3), 1.0])
        x, off = sol.x[:-1], sol.x[-1]
        rf = resid(x, fit) + off
        rp = resid(x, pred) + off
        sc = {"fit_rms_cm": float(np.sqrt(np.mean(rf ** 2))), "predict_rms_cm": float(np.sqrt(np.mean(rp ** 2))) if len(rp) else float("nan"),
              "n_fit": int(len(rf)), "n_predict": int(len(rp)), "coefficients": dict(zip(params, map(float, x))),
              "n_terms": len(params)}
        sc["score"] = sc["predict_rms_cm"]
        sc["summary"] = (f"fit rms {sc['fit_rms_cm']:.4g} cm-1 ({sc['n_fit']} levels J<={self.jfit:g}); predicted "
                         f"J>{self.jfit:g} rms {sc['predict_rms_cm']:.4g} ({sc['n_predict']} levels); {len(params)} terms")
        return {"spec": spec, "bm": bm, "params": params}, sc

    def feedback(self, sc):
        return {k: v for k, v in strip_private(sc).items() if k not in ("summary",)}

    def relation(self, cand, sc, ref_c, ref_s):
        a, b = set(cand["spec"].terms), set(ref_c["spec"].terms)
        if a == b:
            return "equivalent to the reference model"
        extra = sorted(a - b)
        return f"reference model + {len(extra)} extra term(s)" if b <= a else "a different model from the reference"
