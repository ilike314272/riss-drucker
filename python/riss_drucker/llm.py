"""Language-model proposer for effective-Hamiltonian terms (README 7).

The model sees what a spectroscopist would: the spaces and group (with its
character table), the grammar, the current terms and fitted values, residual
diagnostics, and the symmetry-allowed operator dictionary. It answers with term
lines "<name>: <expr>". Every line is then parsed, compiled, certified and fitted
by autobuild; nothing is trusted on the model's word. The completion function is
pluggable (any callable str -> str); `anthropic_complete` is an adapter for the
Claude Messages API.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from . import algebra
from .build import BuildSpec, make_group

GRAMMAR = """Term syntax: "<param>: <expr>", expr built from atoms with * (or space), +, -, numbers, i, I, (...),
sym(...) = average over the symmetry group, and a trailing "+ h.c." for the Hermitian completion.
Atoms: a<mode><comp>[+] (ladder), q<mode><comp>, p<mode><comp>, n<mode>; Jx Jy Jz J2 (molecule-fixed);
Sx Sy Sz S2; Nx Ny Nz N2 (N = J - S); e[<i>,<j>] = |i><j| electronic, Pe[<state>]; cos<k>[<c>], sin<k>[<c>],
p[<c>], q[<c>] for a large-amplitude coordinate c (k may be half-integer on 4pi coordinates).
Every term must be Hermitian, time-reversal even (J, S, p are odd; multiply odd products by i) and invariant
under the group; otherwise the compiler rejects it."""


class LLMProposer:
    name = "llm"

    def __init__(self, complete: Callable[[str], str], *, max_terms: int = 8, dictionary_degrees=None,
                 dictionary_sample: int = 40):
        self.complete, self.max_terms = complete, max_terms
        self.dictionary_degrees, self.dictionary_sample = dictionary_degrees, dictionary_sample

    def prompt(self, context: dict) -> str:
        spec: BuildSpec = context["spec"]
        g = make_group(spec)
        fr = context.get("fit")
        dictionary = []
        if self.dictionary_degrees is not None:
            degs = self.dictionary_degrees(spec) if callable(self.dictionary_degrees) else self.dictionary_degrees
            dictionary = algebra.enumerate_terms(spec, degs, exclude=list(spec.terms), prefix="d").terms
        payload = {
            "task": "Propose additional effective-Hamiltonian terms that would explain the residuals.",
            "grammar": GRAMMAR,
            "spaces": spec.to_dict()["spaces"],
            "group": g.character_table(),
            "J_values": spec.J,
            "current_terms": spec.terms,
            "fitted_values": (fr.theta if fr is not None else {}),
            "reduced_chi2": (fr.reduced_chi2 if fr is not None else None),
            "diagnostics": context.get("diagnostics", {}),
            "symmetry_allowed_dictionary_sample": dictionary[:self.dictionary_sample],
            "answer_format": f"At most {self.max_terms} lines, each '<new_param_name>: <expr>', nothing else.",
        }
        return json.dumps(payload, indent=1, default=str)

    def propose(self, context: dict) -> list[str]:
        text = self.complete(self.prompt(context))
        out = []
        for line in text.splitlines():
            line = line.strip().strip("`").strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", line):
                out.append(line)
        return out[:self.max_terms]


def anthropic_complete(model: str = "claude-opus-5-5", *, max_tokens: int = 16000, effort: str = "high",
                       fallback: bool = True, system: str | None = None) -> Callable[[str], str]:
    """Completion function backed by the Claude Messages API (anthropic SDK; credentials
    from the environment). Adaptive thinking is on by default on this model; effort
    sets its depth. With fallback=True a policy decline is re-run server-side on a
    fallback model; a final refusal raises RuntimeError."""
    import anthropic
    client = anthropic.Anthropic()
    system = system or ("You are an expert molecular spectroscopist building effective Hamiltonians. "
                        "Answer only with term lines in the given grammar.")

    def complete(prompt: str) -> str:
        kw = dict(model=model, max_tokens=max_tokens, system=system, output_config={"effort": effort},
                  messages=[{"role": "user", "content": prompt}])
        if fallback:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        else:
            resp = client.messages.create(**kw)
        u = getattr(resp, "usage", None)
        complete.last_usage = {"input_tokens": getattr(u, "input_tokens", None),
                               "output_tokens": getattr(u, "output_tokens", None)}
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {getattr(resp, 'stop_details', None)}")
        return "".join(b.text for b in resp.content if b.type == "text")
    complete.last_usage = {}
    return complete
