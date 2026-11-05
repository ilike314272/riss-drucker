#!/usr/bin/env python3
"""Language-model-in-the-loop construction of the CH4+ effective Hamiltonian from the ab initio surface.

  python scripts/llm_build_ch4plus.py --abinitio DIR --pdf PAPER.pdf --llm claude-cli [--rounds 4]
  python scripts/llm_build_ch4plus.py --abinitio DIR --pdf PAPER.pdf --llm api
  python scripts/llm_build_ch4plus.py --abinitio DIR --pdf PAPER.pdf --llm replay --replay transcript.json
  python scripts/llm_build_ch4plus.py --abinitio DIR --dry-run          # print the first prompt only

Division of labour (nothing is trusted on the language model's word):
* Claude proposes the *structure*: spaces, group, terms in the builder grammar, and a role per term
  (rotor / tunneling / extra). It sees the ab initio facts (stationary points, symmetry, barrier,
  splitting), the grammar, the Td character table and the feedback from every earlier attempt.
  It does not see the hand-built reference model.
* The library compiles and certifies the proposal (hermiticity, time reversal, symmetry, counts,
  representation); a failure is fed back verbatim.
* Constants come from the surface, never from the model: rotor coefficients are calibrated so the
  untunneled levels equal the ab initio rigid rotor (A, B, C of the C2v minimum) replicated over
  the wells; the tunneling coefficient is set so the nearest-well matrix element equals the ab
  initio sigma (1D reaction-path tunneling); extra terms start at 0.
* The spectrum is predicted with only the IE offset, then a Bayesian update (ab initio prior) and
  leave-one-position-out cross-validation score the model against the measured PFI-ZEKE lines of
  Woerner, Qian & Merkt, JCP 126, 144305 (2007) (read from the user's PDF).
* The hand-built reference model of case_methane_cation_rovibronic goes through the identical
  pipeline. The final model is the best by cross-validated rms; the report says whether it is the
  language model's and whether it is equivalent to the reference.

Backends: claude-cli runs `claude -p` (Claude Code print mode, no tools) on your own login, so
calls draw on your subscription; api uses the Anthropic SDK with ANTHROPIC_API_KEY (API billing);
replay re-runs a saved transcript exactly (reproducible, used by the tests).

Limits: a usage, rate or credit limit stops the loop cleanly (no retry); overload, timeout and
network errors are retried with backoff (--retries, --retry-wait). --max-budget-usd and
--max-total-tokens stop the loop before a cap is passed (the budget is also passed to each
`claude -p` call). On any stop, including Ctrl-C, the transcript and the best model so far are
written, and `--resume TRANSCRIPT` continues later: saved answers are re-evaluated without new
calls and count against the caps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ISOTOPES = (("CH4+", "H"), ("CD4+", "D"))

WELL_GRAMMAR = """Well space atoms: w[<i>,<j>] = |well i><well j| (1-based well indices); w[i,i] projects on well i.
sym(X) averages X over the group, so sym(w[1,1] Jx Jx) is one well's operator copied to every well with the
group's signs. A WellSpace is {"kind": "well", "name": str, "irrep": str, "vectors": [[...], ...],
"geometric_phase": bool}: one vector per well = the adiabatic electronic function of that well in the carrier
basis (x, y, z) of `irrep`; the group permutes the wells with signs rho(g) v_n = s v_m. With
geometric_phase=false the signs are dropped (wells permuted only). Rotation space: {"kind": "rot"} (|N k>,
molecule-fixed Jx Jy Jz). The rotational quantum number here is N+ (electron spin not resolved)."""

ANSWER_FORMAT = """Answer with exactly one JSON object (in a ```json block), nothing else outside it:
{"spec": {"group": "<group>", "spaces": [...], "terms": ["<param>: <expr>", ...], "J": [0, 1, 2, 3, 4, 5]},
 "roles": {"<param>": "rotor" | "tunneling" | "extra", ...},
 "rationale": "<short physical justification>"}
Roles: "rotor" terms are calibrated together so that, without tunneling, every well is the ab initio rigid
rotor (A, B, C of the C2v minimum); exactly one "tunneling" term is calibrated so its nearest-well matrix
element equals the ab initio sigma; "extra" terms start at 0 and are fitted with a zero-mean prior."""

SYSTEM = ("You are an expert molecular spectroscopist building effective Hamiltonians from ab initio results. "
          "Reason carefully about symmetry, then answer only with the requested JSON object.")


# ----------------------------------------------------------------------------- completion backends
class LimitReached(Exception):
    """Usage, rate or credit limit: stop the loop, keep what was done, resume later with --resume."""


class TransientError(Exception):
    """Overload, timeout or network failure: worth retrying after a pause."""


LIMIT_PAT = re.compile(r"usage limit|limit reached|limit will reset|rate.?limit|quota|credit balance|"
                       r"too many requests|\b429\b|budget", re.I)
TRANSIENT_PAT = re.compile(r"overloaded|\b5\d\d\b|timed? ?out|connection|temporar|ECONNRESET|network", re.I)


def classify(msg: str) -> Exception:
    if LIMIT_PAT.search(msg):
        return LimitReached(msg)
    if TRANSIENT_PAT.search(msg):
        return TransientError(msg)
    return RuntimeError(msg)


def claude_cli_complete(model=None, timeout=1800, budget_left=None):
    """Claude Code print mode, tools disabled, one turn; uses the logged-in subscription unless
    ANTHROPIC_API_KEY is set (then Claude Code bills the API key). `budget_left()` (USD) is passed as
    --max-budget-usd so a single call cannot overrun the cap. After each call `complete.last_usage`
    holds the reported cost and tokens (for a subscription the cost is a usage measure, not a bill)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("note: ANTHROPIC_API_KEY is set, so `claude -p` will bill the API key, not your subscription",
              file=sys.stderr)

    def complete(prompt: str) -> str:
        cmd = ["claude", "-p", "--tools", "", "--max-turns", "1", "--output-format", "json",
               "--append-system-prompt", SYSTEM]
        if model:
            cmd += ["--model", model]
        if budget_left is not None and budget_left() is not None:
            cmd += ["--max-budget-usd", f"{max(budget_left(), 0.01):.2f}"]
        try:
            out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TransientError(f"claude -p timed out after {timeout} s")
        except FileNotFoundError:
            raise RuntimeError("`claude` not found: install Claude Code (curl -fsSL https://claude.ai/install.sh | bash)")
        try:
            data = json.loads(out.stdout)
        except (json.JSONDecodeError, ValueError):
            data = {}
        complete.last_usage = {"cost_usd": data.get("total_cost_usd"),
                               "input_tokens": (data.get("usage") or {}).get("input_tokens"),
                               "output_tokens": (data.get("usage") or {}).get("output_tokens")}
        if out.returncode or data.get("is_error") or "result" not in data:
            msg = f"claude -p failed (exit {out.returncode}): {data.get('result') or data.get('subtype') or ''} " \
                  f"{out.stderr[-2000:]} {out.stdout[-500:] if not data else ''}"
            raise classify(msg)
        return data["result"]
    complete.last_usage = {}
    return complete


def api_complete(model):
    from riss_drucker.llm import anthropic_complete
    inner = anthropic_complete(model, system=SYSTEM)

    def complete(prompt: str) -> str:
        try:
            text = inner(prompt)
        except Exception as e:                 # anthropic.RateLimitError (429), overloaded (529), low credit, ...
            status = getattr(e, "status_code", None)
            msg = f"{type(e).__name__} (status {status}): {e}"
            if status == 429 or "credit" in str(e).lower():
                raise LimitReached(msg)
            if status in (500, 502, 503, 504, 529) or "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
                raise TransientError(msg)
            raise classify(msg)
        complete.last_usage = dict(inner.last_usage)
        return text
    complete.last_usage = {}
    return complete


def with_retries(complete, prompt, retries=3, wait=(30, 90, 270), log=print):
    """Retry transient failures with backoff; limits and other errors propagate at once."""
    for i in range(retries + 1):
        try:
            return complete(prompt)
        except TransientError as e:
            if i == retries:
                raise RuntimeError(f"gave up after {retries} retries: {e}")
            log(f"  transient failure ({str(e)[:200]}); retry {i + 1}/{retries} in {wait[min(i, len(wait) - 1)]} s")
            time.sleep(wait[min(i, len(wait) - 1)])


def resumed(saved: list, live):
    """Serve the saved answers of an interrupted run first (no calls, no usage), then the live backend."""
    saved = list(saved)

    def complete(prompt: str) -> str:
        if saved:
            complete.last_usage = {"resumed": True}
            return saved.pop(0)
        if live is None:
            raise RuntimeError("no saved answers left and no live backend")
        text = live(prompt)
        complete.last_usage = getattr(live, "last_usage", {})
        return text
    complete.last_usage = {}
    complete.pending = lambda: len(saved)
    return complete


def replay_complete(path):
    answers = [r["response"] for r in json.loads(Path(path).read_text())["rounds"]]

    def complete(prompt: str) -> str:
        if not answers:
            raise RuntimeError("replay transcript exhausted")
        return answers.pop(0)
    return complete


# ----------------------------------------------------------------------------- ab initio facts
def abinitio_facts(abd: Path) -> dict:
    import case_methane_cation_abinitio as A
    from riss_drucker.molecule import Molecule
    mn, ts = A.load(abd, "min"), A.load(abd, "ts")
    facts = {"basis": mn.get("meta", {}).get("basis")}
    m_min = Molecule(["C", "H", "H", "H", "H"], np.array(mn["xyz"]), charge=1, multiplicity=2)
    m_ts = Molecule(["C", "H", "H", "H", "H"], np.array(ts["xyz"]), charge=1, multiplicity=2)
    pg_min, pg_ts = m_min.schoenflies(), m_ts.schoenflies()
    order = {"C1": 1, "Cs": 2, "C2": 2, "C2v": 4, "C3v": 6, "D2d": 8, "Td": 24}
    facts["structure"] = {
        "parent": "CH4 Td; removing one electron from the t2 HOMO gives a 2T2 state, Jahn-Teller active in e + t2",
        "minimum_point_group": pg_min, "saddle_point_group": pg_ts,
        "n_equivalent_minima": order["Td"] // order.get(pg_min, 1) if pg_min in order else None,
        "minimum_CH_bonds_A": sorted(round(float(np.linalg.norm(np.array(mn["xyz"])[i] - np.array(mn["xyz"])[0])), 4)
                                     for i in range(1, 5)),
        "minimum_xyz_A": np.round(np.array(mn["xyz"]), 5).tolist(),
        "saddle_imaginary_frequency_cm": round(float(min(ts["frequencies"])), 1),
        "saddle_connects": ts.get("neighbour", "least-motion neighbouring minimum"),
    }
    per = {}
    for name, iso in ISOTOPES:
        prof = A.path_profile(abd, iso)
        t = A.tunneling(prof)
        per[name] = {"rigid_rotor_C2v_minimum_cm": A.rotational_constants(mn["xyz"], iso),
                     "constant_about_C2_axis": c2_axis_constant(mn["xyz"], iso),
                     "barrier_electronic_cm": round(float(prof["V"].max()), 1),
                     "barrier_above_tunneling_ground_level_cm": round(t["barrier_above_ground"], 1),
                     "sigma_abs_cm": t["sigma_abs"],
                     "sigma_definition": "half the splitting of the lowest pair of the 1D double well between two "
                                         "neighbouring minima (reaction-path tunneling with ZPE_perp)"}
    facts["isotopologues"] = per
    return facts


def c2_axis_constant(xyz, iso) -> str | None:
    """Which rotational constant (A, B or C) belongs to the principal axis along the C2 axis of the
    minimum (geometry only)."""
    from riss_drucker.molecule import Molecule
    m = Molecule(["C"] + [iso] * 4, np.array(xyz))
    rc = m.rotational_constants()
    for O in m.symmetry_operations():
        O = np.asarray(O)
        if np.linalg.det(O) > 0 and abs(np.trace(O) + 1) < 1e-6:
            k = int(np.argmax(np.abs(np.linalg.eigh(O)[1][:, -1])))
            order = sorted(rc, key=lambda a: -rc[a])
            return "ABC"[order.index("xyz"[k])]
    return None


def observed_classes(tables) -> dict:
    out = {}
    for name, _ in ISOTOPES:
        rows = [r for r in tables[name] if r["N"] and "/" not in r["N"]]
        out[name] = sorted({f"J''={r['J']} {r['G']} -> N+={r['N']} {r['Gp']}" for r in rows})
    return out


# ----------------------------------------------------------------------------- candidate model
class Candidate:
    """A compiled proposal with constants calibrated from the ab initio surface."""

    def __init__(self, spec_dict: dict, roles: dict):
        from riss_drucker import build as B
        from riss_drucker.build import Context, Operators, Parser, runtime_spaces
        d = dict(spec_dict)
        d.setdefault("J", list(range(6)))
        d.setdefault("max_order", 10)
        d.setdefault("name", "llm")
        self.spec = B.BuildSpec.from_dict(d)
        self.bm = B.compile_build(self.spec)
        self.params = [t.split(":")[0].strip() for t in self.spec.terms]
        unknown = set(roles) - set(self.params)
        if unknown:
            raise ValueError(f"roles given for unknown parameters {sorted(unknown)}")
        self.roles = {p: roles.get(p, "extra") for p in self.params}
        self.rotor = [p for p in self.params if self.roles[p] == "rotor"]
        tun = [p for p in self.params if self.roles[p] == "tunneling"]
        if len(tun) != 1:
            raise ValueError(f"exactly one term must have role 'tunneling' (got {tun})")
        if not self.rotor:
            raise ValueError("no term has role 'rotor'")
        self.tun = tun[0]
        self.extra = [p for p in self.params if self.roles[p] == "extra"]
        g = self.bm.group
        ctx = Context(g, runtime_spaces(self.spec, g))
        self._ctx, self._P, self._Ops = ctx, Parser(ctx), Operators
        self.exprs = {t.split(":")[0].strip(): t.split(":", 1)[1] for t in self.spec.terms}
        expr = self.exprs[self.tun]
        M = Operators(ctx, 0).matrix(ctx.canon(Parser(ctx).parse_expr(expr))).toarray()
        self.hop = float(np.max(np.abs(M)))
        if self.hop < 1e-12:
            raise ValueError(f"tunneling term {self.tun} has no matrix elements at N=0")
        self.dims = {ir: g.irreps[ir][0].shape[0] if ir in g.irreps else 1 for ir in {b.irrep for b in self.bm.blocks}}
        self.n_wells = next((len(s.vectors) for s in self.spec.spaces if s.kind == "well"), 1)

    def _well(self):
        ws = [s for s in self.spec.spaces if s.kind == "well"]
        return ws[0] if len(ws) == 1 and [s.kind for s in self.spec.spaces] == ["rot", "well"] else None

    def well_c2_axis(self):
        """Axis (molecule frame) of the proper C2 rotation that keeps well 1 (up to sign), or None."""
        w = self._well()
        if w is None:
            return None
        g = self.bm.group
        v = np.asarray(w.vectors[0], float)
        v = v / np.linalg.norm(v)
        for el, rho in zip(g.elements, g.irreps[w.irrep]):
            R = np.asarray(el.R, float)
            if el.parity == 1 and abs(np.trace(R) + 1) < 1e-8 and abs(abs(v @ (np.asarray(rho).real @ v)) - 1) < 1e-8:
                wv, V = np.linalg.eigh((R + R.T) / 2)
                return V[:, int(np.argmax(wv))]
        return None

    def well_quadratic_form(self, coeffs: dict):
        """Rotor terms restricted to well 1 written as sum_ab Q_ab J_a J_b (fit over N = 1, 2): returns
        the principal constants and axes of Q (molecule frame)."""
        nw = self.n_wells
        pairs = [(a, b) for i, a in enumerate("xyz") for b in "xyz"[i:]]
        rows, rhs = [], []
        for tj in (2, 4):
            ops = self._Ops(self._ctx, tj)
            idx = [k * nw for k in range(tj + 1)]

            def mat(expr):
                return ops.matrix(self._ctx.canon(self._P.parse_expr(expr))).toarray()[np.ix_(idx, idx)]
            H = sum(c * mat(self.exprs[p]) for p, c in coeffs.items() if p in self.rotor)
            basis = [mat(f"J{a} J{b}") if a == b else mat(f"J{a} J{b} + J{b} J{a}") for a, b in pairs]
            rows.append(np.array([B.ravel() for B in basis]).T)
            rhs.append(H.ravel())
        A_ = np.vstack(rows)
        q = np.linalg.lstsq(np.vstack([A_.real, A_.imag]), np.r_[np.concatenate(rhs).real, np.concatenate(rhs).imag],
                            rcond=None)[0]
        Q = np.zeros((3, 3))
        for (a, b), val in zip(pairs, q):
            i, j = "xyz".index(a), "xyz".index(b)
            Q[i, j] = Q[j, i] = val
        return np.linalg.eigh(Q)

    def theta(self, coeffs: dict) -> dict:
        return {p: float(coeffs.get(p, 0.0)) for p in self.params}

    def level_multiset(self, th, nmax=3):
        out = {}
        for b, (w, _) in zip(self.bm.blocks, self.bm.eig(th)):
            N = b.two_j // 2
            if N <= nmax:
                out.setdefault(N, []).extend(np.repeat(w, self.dims[b.irrep]).tolist())
        return {N: np.sort(v) for N, v in out.items()}

    def calibrate(self, A, Bc, C, nmax=3, starts=40, seed=0):
        """Rotor coefficients such that the untunneled levels equal the rigid rotor (A, B, C) replicated
        over the wells for N <= nmax (sorted multisets: the proposal's frame and normalization do not
        matter). The rotor levels do not fix which principal axis lies along which well-frame axis, nor
        signs of cross terms; these branches differ once tunneling is on, so every distinct branch is
        returned. Each comes with the linear map d(coefficients)/d(A, B, C), so the update can be
        parametrized by the physical constants whatever the operator normalization."""
        from scipy.optimize import least_squares
        abc0 = np.array([A, Bc, C], float)

        def target(abc):
            return {N: np.sort(np.repeat(rigid_rotor(*abc, N), self.n_wells)) for N in range(nmax + 1)}

        def res(x, tg):
            lev = self.level_multiset(self.theta(dict(zip(self.rotor, x))), nmax)
            if set(lev) != set(tg) or any(len(lev[N]) != len(tg[N]) for N in tg):
                raise ValueError("proposal's state count per N differs from rigid rotor x wells "
                                 f"({ {N: len(v) for N, v in lev.items()} } vs { {N: len(v) for N, v in tg.items()} })")
            return np.concatenate([lev[N] - tg[N] for N in sorted(tg)])
        tg0 = target(abc0)
        rng = np.random.default_rng(seed)
        sols = []
        scale = 2 * float(abc0.max())
        for _ in range(starts):
            sol = least_squares(res, rng.uniform(-6 * scale, 6 * scale, len(self.rotor)), args=(tg0,),
                                xtol=1e-14, ftol=1e-14, gtol=1e-14)
            sols.append(sol)
            for signs in itertools.product((1, -1), repeat=len(self.rotor)):
                x = sol.x * np.array(signs)
                if np.sqrt(np.mean(res(x, tg0) ** 2)) < 1e-6:
                    sols.append(type("S", (), {"x": x, "fun": res(x, tg0)})())
        good = [s.x for s in sols if np.sqrt(np.mean(np.square(s.fun))) < 1e-6]
        best_rms = float(min(np.sqrt(np.mean(np.square(s.fun))) for s in sols))
        if not good:
            return best_rms, []
        branches = []
        for x in good:
            c = dict(zip(self.rotor, x))
            c[self.tun] = 1.0                      # unit tunneling: distinguishes the branches
            lv = self.level_multiset(self.theta(c), 5)
            if any(all(np.allclose(lv[N], b["_lv"][N], atol=1e-7) for N in lv) for b in branches):
                continue
            L = []
            for i in range(3):                     # follow the same branch: start at x, tiny step in A, B or C
                h = 1e-4 * abc0[i]
                d = abc0.copy()
                d[i] += h
                xi = least_squares(res, x, args=(target(d),), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
                L.append((xi - x) / h)
            branches.append({"coeffs": dict(zip(self.rotor, x)), "L": np.array(L), "_lv": lv})
        for b in branches:
            b.pop("_lv")
        return best_rms, branches


def rigid_rotor(A, Bc, C, N):
    k = np.arange(-N, N + 1, dtype=float)
    jp = np.diag(np.sqrt(N * (N + 1) - k[:-1] * (k[:-1] + 1)), -1)
    Jx, Jy, Jz = (jp + jp.T) / 2, (jp - jp.T) / 2j, np.diag(k)
    H = A * Jz @ Jz + Bc * Jx @ Jx + C * (Jy @ Jy).real
    return np.linalg.eigvalsh((H + H.conj().T).real / 2 if np.iscomplexobj(H) else H)


# ----------------------------------------------------------------------------- evaluation vs measured lines
def levels(cand: Candidate, coeffs: dict):
    out = {}
    for b, (w, _) in zip(cand.bm.blocks, cand.bm.eig(cand.theta(coeffs))):
        for k, e in enumerate(w):
            out[(b.two_j // 2, b.irrep, k)] = float(e)
    e0 = min(e for (N, _, _), e in out.items() if N == 0)
    return {k: v - e0 for k, v in out.items()}


def assign(cand, IE, coeffs, rows, Bn):
    import case_methane_cation_rovibronic as R
    lev = levels(cand, coeffs)
    out = []
    for r in rows:
        El = Bn * r["J"] * (r["J"] + 1)
        dist = lambda k: abs(IE + lev[k] - El - r["nu"])
        cands = [k for k in lev if k[0] == int(r["N"]) and k[1] in R.allowed(r["Gp"], r["G"])]
        if not cands:
            raise ValueError(f"model has no level of class {r['Gp']} at N+={r['N']} "
                             f"(needed by the measured line J''={r['J']} {r['G']} at {r['nu']} cm-1)")
        best = min(cands, key=dist)
        if dist(best) > R.RELABEL_TOL:
            alt = [k for k in lev if abs(k[0] - int(r["N"])) == 1 and k[1] in R.allowed(r["Gp"], r["G"])]
            if alt and dist(min(alt, key=dist)) < dist(best) / 3:
                best = min(alt, key=dist)
        out.append(best)
    return out


def calc(cand, IE, coeffs, rows, Bn, asg):
    lev = levels(cand, coeffs)
    return np.array([IE + lev[k] - Bn * r["J"] * (r["J"] + 1) for r, k in zip(rows, asg)])


def evaluate(cand: Candidate, name: str, facts: dict, rows, Bn, paper_IE, cv=True) -> dict:
    """Ab initio prediction (IE offset only), Bayesian update with re-assignment, LOO-position CV.
    Parameters of the update are physical: IE, A, B, C, sigma (+ extra-term coefficients)."""
    from scipy.optimize import least_squares
    iso = facts["isotopologues"][name]
    rc = iso["rigid_rotor_C2v_minimum_cm"]
    abc0 = np.array([rc["A"], rc["B"], rc["C"]])
    sig0 = iso["sigma_abs_cm"]
    cal_rms, branches = cand.calibrate(*abc0)
    if not branches:
        raise ValueError(f"rotor terms cannot reproduce the ab initio rigid rotor (calibration rms {cal_rms:.3g} cm-1)")
    # geometry fixes which constant lies along each well's C2 axis; only branches that respect it are physical
    axis, label = cand.well_c2_axis(), iso.get("constant_about_C2_axis")
    n_all = len(branches)
    if axis is not None and label is not None:
        want = abc0["ABC".index(label)]
        keep = []
        for br in branches:
            vals, vecs = cand.well_quadratic_form(br["coeffs"])
            k = int(np.argmax(np.abs(vecs.T @ axis)))
            if abs(abs(vecs[:, k] @ axis) - 1) < 1e-6 and abs(vals[k] - want) < 1e-4 * want:
                keep.append(br)
        if not keep:
            raise ValueError(f"no calibration of the rotor terms puts {label} = {want:.4f} cm-1 on the well's C2 axis "
                             "(the C2v geometry requires it)")
        branches = keep
    obs = np.array([r["nu"] for r in rows])

    def coeffs_of(br, p):
        """p = [A, B, C, sigma, extras...] -> term coefficients on branch br."""
        c = {k: v + float(br["L"][:, i] @ (p[:3] - abc0)) for i, (k, v) in enumerate(br["coeffs"].items())}
        c[cand.tun] = p[3] / cand.hop
        c.update(dict(zip(cand.extra, p[4:])))
        return c
    p0 = np.r_[abc0, sig0, np.zeros(len(cand.extra))]
    best = None
    for bi, br in enumerate(branches):
        for sgn in (1.0, -1.0):                    # sign of sigma (which vibronic component is lower)
            p = p0.copy()
            p[3] *= sgn
            c = coeffs_of(br, p)
            asg = assign(cand, paper_IE, c, rows, Bn)
            IE = paper_IE + float(np.mean(obs - calc(cand, paper_IE, c, rows, Bn, asg)))
            asg = assign(cand, IE, c, rows, Bn)
            IE += float(np.mean(obs - calc(cand, IE, c, rows, Bn, asg)))
            rms = float(np.sqrt(np.mean((obs - calc(cand, IE, c, rows, Bn, asg)) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, IE, bi, p)
    rms0, IE0, bi, pstart = best
    br = branches[bi]
    x0 = np.r_[IE0, pstart]
    rot_scale = max(abs(v) for v in br["coeffs"].values())
    sd = np.r_[0.05 * abc0, 0.7 * sig0 + 0.3, np.full(len(cand.extra), 0.1 * rot_scale)]

    def update(rows_fit, x_start):
        o = np.array([r["nu"] for r in rows_fit])
        x, asg, seen = x_start.copy(), None, []
        for _ in range(8):
            a = assign(cand, x[0], coeffs_of(br, x[1:]), rows_fit, Bn)
            key = repr(a)
            if key in seen:
                break
            seen.append(key)
            asg = a

            def r_(q, asg=asg):
                return np.r_[(o - calc(cand, q[0], coeffs_of(br, q[1:]), rows_fit, Bn, asg)) / 0.5,
                             (q[1:] - x0[1:]) / sd]
            x = least_squares(r_, x, x_scale=np.r_[1.0, sd], diff_step=1e-6).x
        return x, asg
    xu, asg = update(rows, x0)
    cu = coeffs_of(br, xu[1:])
    rmsu = float(np.sqrt(np.mean((obs - calc(cand, xu[0], cu, rows, Bn, asg)) ** 2)))
    pn = ["A", "B", "C", "sigma"] + cand.extra
    out = {"calibration_rms_cm": cal_rms, "n_calibration_branches": n_all,
           "n_branches_consistent_with_geometry": len(branches), "branch_selected": bi,
           "ab_initio": dict(zip(pn, map(float, pstart))), "prediction_rms_IE_only_cm": rms0, "IE_offset": IE0,
           "posterior": {"IE": float(xu[0]), **dict(zip(pn, map(float, xu[1:])))}, "updated_rms_cm": rmsu,
           "shift_in_prior_sd": {k: float((xu[i + 1] - x0[i + 1]) / sd[i]) for i, k in enumerate(pn)},
           "delta_posterior_cm": float(4 * abs(xu[4])), "n_lines": len(rows),
           "ab_initio_coefficients": coeffs_of(br, pstart), "posterior_coefficients": cu}
    if cv:
        pos = sorted({round(r["nu"], 2) for r in rows})
        err = []
        for p in pos:
            fit = [r for r in rows if round(r["nu"], 2) != p]
            test = [r for r in rows if round(r["nu"], 2) == p]
            xf, _ = update(fit, xu)
            cf = coeffs_of(br, xf[1:])
            err += list(np.array([r["nu"] for r in test]) - calc(cand, xf[0], cf, test, Bn, assign(cand, xf[0], cf, test, Bn)))
        out["cv_rms_cm"] = float(np.sqrt(np.mean(np.square(err))))
    out["_plot"] = {"IE": float(xu[0]), "coeffs": cu, "IE0": IE0, "coeffs0": coeffs_of(br, pstart), "obs": obs.tolist()}
    return out


def score_candidate(cand, facts, tables, cv=True):
    import case_methane_cation_rovibronic as R
    res = {}
    for name, _ in ISOTOPES:
        rows = [r for r in tables[name] if r["N"] and "/" not in r["N"]]
        res[name] = evaluate(cand, name, facts, rows, R.NEUTRAL_B[name], R.PAPER[name]["IE"], cv=cv)
    res["score_cv_rms_mean"] = float(np.mean([res[n].get("cv_rms_cm", res[n]["updated_rms_cm"]) for n, _ in ISOTOPES]))
    return res


def reference_proposal() -> dict:
    import case_methane_cation_rovibronic as R
    m = R.Model(5)
    return {"spec": {k: v for k, v in m.spec.to_dict().items() if k in ("group", "spaces", "terms", "J")},
            "roles": {"rA": "rotor", "rBC": "rotor", "rD": "rotor", "sig": "tunneling"},
            "rationale": "hand-built reference (case_methane_cation_rovibronic)"}


def equivalent(c1: Candidate, k1: dict, c2: Candidate, k2: dict) -> bool:
    l1, l2 = c1.level_multiset(c1.theta(k1), 5), c2.level_multiset(c2.theta(k2), 5)
    return set(l1) == set(l2) and all(len(l1[N]) == len(l2[N]) and np.allclose(l1[N], l2[N], atol=1e-6) for N in l1)


# ----------------------------------------------------------------------------- the loop
def build_prompt(facts, classes, group_table, history) -> str:
    from riss_drucker.llm import GRAMMAR
    payload = {
        "task": ("Build the effective Hamiltonian for the vibronic ground state of CH4+ (rotation x tunneling "
                 "between the equivalent Jahn-Teller minima) whose constants will be taken from the ab initio "
                 "surface below, and which must reproduce the measured PFI-ZEKE line classes listed. Choose the "
                 "spaces (including the well vectors), the group and the terms."),
        "ab_initio": facts,
        "measured_line_classes (nuclear-spin symmetry conserved: A<->A1/A2, E<->E, F<->F1/F2)": classes,
        "grammar": GRAMMAR + "\n" + WELL_GRAMMAR,
        "group_Td_character_table": group_table,
        "previous_attempts_and_feedback": history,
        "answer_format": ANSWER_FORMAT,
    }
    return json.dumps(payload, indent=1, default=str)


def parse_answer(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        blob = m.group(1)
    elif "{" in text and "}" in text:
        blob = text[text.index("{"):text.rindex("}") + 1]
    else:
        raise ValueError("no JSON object found in the answer (expected one ```json block)")
    d = json.loads(blob)
    if "spec" not in d or "roles" not in d:
        raise ValueError("answer lacks 'spec' or 'roles'")
    return d


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abinitio", required=True)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--llm", choices=["claude-cli", "api", "replay"], default="claude-cli")
    ap.add_argument("--model", default=None, help="claude-cli: model alias (e.g. opus); api: model id")
    ap.add_argument("--replay", default=None, help="transcript JSON from an earlier run (--llm replay)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation (faster)")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    ap.add_argument("--out", default="ch4plus_llm_model.json")
    ap.add_argument("--transcript", default="ch4plus_llm_transcript.json")
    ap.add_argument("--figures", default=None)
    ap.add_argument("--resume", default=None, help="transcript of an interrupted run: its answers are reused "
                                                   "(no calls), then the loop continues live up to --rounds in total")
    ap.add_argument("--max-budget-usd", type=float, default=None,
                    help="stop before the reported cost exceeds this (claude-cli: total_cost_usd per call, which for "
                         "a subscription measures usage rather than billing; also passed to each call)")
    ap.add_argument("--max-total-tokens", type=int, default=None, help="stop once input+output tokens exceed this")
    ap.add_argument("--retries", type=int, default=3, help="retries for overload/timeout/network errors")
    ap.add_argument("--retry-wait", type=float, default=30.0, help="first retry pause in s (then x3, x9)")
    a = ap.parse_args()

    from riss_drucker import symmetry as S
    facts = abinitio_facts(Path(a.abinitio))
    group_table = S.get_group("Td").character_table()
    tables = classes = None
    if a.pdf:
        import case_methane_cation_rovibronic as R
        tables = R.read_tables(Path(a.pdf))
        classes = observed_classes(tables)
    if a.dry_run:
        print(build_prompt(facts, classes, group_table, []))
        return 0
    if tables is None:
        ap.error("--pdf is required (the proposals are scored against the measured lines)")
    usage = {"cost_usd": 0.0, "tokens": 0, "calls": 0}

    def budget_left():
        return None if a.max_budget_usd is None else a.max_budget_usd - usage["cost_usd"]
    live = {"claude-cli": lambda: claude_cli_complete(a.model, budget_left=budget_left),
            "api": lambda: api_complete(a.model or "claude-opus-5-5"),
            "replay": lambda: replay_complete(a.replay)}[a.llm]()
    saved_rounds = []
    if a.resume:
        saved_rounds = [r for r in json.loads(Path(a.resume).read_text())["rounds"] if "response" in r]
        for r in saved_rounds:                     # earlier usage counts against the caps
            u = r.get("usage") or {}
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            usage["tokens"] += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        print(f"resuming: {len(saved_rounds)} saved answers will be re-evaluated without calls", file=sys.stderr)
    complete = resumed([r["response"] for r in saved_rounds], live)

    cv = not a.no_cv
    ref = reference_proposal()
    ref_cand = Candidate(ref["spec"], ref["roles"])
    ref_score = score_candidate(ref_cand, facts, tables, cv)
    print(f"reference model: score {ref_score['score_cv_rms_mean']:.3f} cm-1", file=sys.stderr)

    history, rounds, best = [], [], ("reference", ref_cand, ref_score, ref)
    stop_reason = "completed all rounds"

    def save_transcript():
        Path(a.transcript).write_text(json.dumps({"backend": a.llm, "model": a.model, "rounds": rounds,
                                                  "usage_total": usage, "stop_reason": stop_reason},
                                                 indent=1, default=str))
    for k in range(a.rounds):
        live_call = complete.pending() == 0            # saved answers cost nothing; caps only gate new calls
        if live_call and a.max_budget_usd is not None and usage["cost_usd"] >= a.max_budget_usd:
            stop_reason = f"budget cap reached ({usage['cost_usd']:.2f} >= {a.max_budget_usd} USD)"
            break
        if live_call and a.max_total_tokens is not None and usage["tokens"] >= a.max_total_tokens:
            stop_reason = f"token cap reached ({usage['tokens']} >= {a.max_total_tokens})"
            break
        prompt = build_prompt(facts, classes, group_table, history)
        t0 = time.time()
        try:
            text = with_retries(complete, prompt, a.retries, wait=(a.retry_wait, 3 * a.retry_wait, 9 * a.retry_wait),
                                log=lambda m: print(m, file=sys.stderr))
        except LimitReached as e:
            stop_reason = f"usage/rate limit: {str(e)[:500]}"
            break
        except KeyboardInterrupt:
            stop_reason = "interrupted by user"
            break
        except Exception as e:
            stop_reason = f"language-model call failed: {type(e).__name__}: {str(e)[:500]}"
            break
        u = dict(getattr(complete, "last_usage", {}) or {})
        if not u.get("resumed"):
            usage["calls"] += 1
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            usage["tokens"] += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        rec = {"round": k + 1, "response": text, "seconds": round(time.time() - t0, 1), "usage": u}
        try:
            prop = parse_answer(text)
            cand = Candidate(prop["spec"], prop["roles"])
            sc = score_candidate(cand, facts, tables, cv)
            same = all(equivalent(cand, sc[n]["ab_initio_coefficients"], ref_cand, ref_score[n]["ab_initio_coefficients"])
                       for n, _ in ISOTOPES)
            fb = {"status": "compiled, certified, calibrated and scored", "equivalent_to_a_known_model": same,
                  **{n: {kk: sc[n][kk] for kk in ("n_calibration_branches", "n_branches_consistent_with_geometry",
                                                   "prediction_rms_IE_only_cm",
                                                   "updated_rms_cm", "cv_rms_cm", "delta_posterior_cm", "posterior",
                                                   "shift_in_prior_sd") if kk in sc[n]}
                     for n, _ in ISOTOPES},
                  "score_cv_rms_mean": sc["score_cv_rms_mean"],
                  "reference_score_cv_rms_mean": ref_score["score_cv_rms_mean"]}
            rec.update(proposal=prop, score=strip_private(sc), equivalent_to_reference=same)
            if sc["score_cv_rms_mean"] < best[2]["score_cv_rms_mean"] - 1e-4 or (
                    best[0] == "reference" and sc["score_cv_rms_mean"] <= ref_score["score_cv_rms_mean"] + 1e-4):
                best = (f"llm round {k + 1}", cand, sc, prop)
        except Exception as e:          # every failure goes back to the model verbatim
            fb = {"status": "rejected", "error": f"{type(e).__name__}: {e}"}
            rec["error"] = fb["error"]
        print(f"round {k + 1}: {fb.get('status')} {fb.get('error', '')} "
              f"{fb.get('score_cv_rms_mean', '')}", file=sys.stderr)
        history.append({"round": k + 1, "proposal": rec.get("proposal", text[:4000]), "feedback": fb})
        rounds.append(rec)
        save_transcript()
    save_transcript()
    if stop_reason != "completed all rounds":
        print(f"stopped early: {stop_reason}\n  {len(rounds)} round(s) saved in {a.transcript}; the best model so far "
              f"is written below. Continue later with:  --resume {a.transcript}", file=sys.stderr)

    label, cand, sc, prop = best
    final = {"selected": label, "stop_reason": stop_reason, "rounds_done": len(rounds), "usage_total": usage,
             "selection_rule": "lowest mean leave-one-position-out rms over CH4+ and CD4+ "
                                                  "(Bayesian update with ab initio prior); the language model's "
                                                  "proposal is preferred on ties with the reference",
             "spec": cand.spec.to_dict(), "roles": cand.roles, "rationale": prop.get("rationale"),
             "abinitio_basis": facts["basis"], "scores": strip_private(sc),
             "reference_scores": strip_private(ref_score),
             "levels_N_le_5": {n: [{"N": N, "irrep": ir, "k": kk, "E_cm": e}
                                   for (N, ir, kk), e in sorted(levels(cand, sc[n]["posterior_coefficients"]).items(),
                                                                key=lambda t: t[1])]
                               for n, _ in ISOTOPES}}
    Path(a.out).write_text(json.dumps(final, indent=1, default=str))
    print(json.dumps({"selected": label, "stop_reason": stop_reason, "usage_total": usage,
                      "score": sc["score_cv_rms_mean"],
                      "reference_score": ref_score["score_cv_rms_mean"],
                      **{n: {"prediction_rms_IE_only_cm": sc[n]["prediction_rms_IE_only_cm"],
                             "updated_rms_cm": sc[n]["updated_rms_cm"], "cv_rms_cm": sc[n].get("cv_rms_cm"),
                             "posterior": sc[n]["posterior"], "delta_cm": sc[n]["delta_posterior_cm"]}
                         for n, _ in ISOTOPES}}, indent=1))
    if a.figures:
        plot(Path(a.figures), cand, sc, tables)
    return 0


def plot(fd: Path, cand, sc, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import case_methane_cation_rovibronic as R
    fd.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, _) in zip(axes, ISOTOPES):
        rows = [r for r in tables[name] if r["N"] and "/" not in r["N"]]
        Bn = R.NEUTRAL_B[name]
        p = sc[name]["_plot"]
        obs = np.array(p["obs"])
        for IE, cc, lab, mk in ((p["IE0"], p["coeffs0"], "ab initio", "o"), (p["IE"], p["coeffs"], "updated", "s")):
            c = calc(cand, IE, cc, rows, Bn, assign(cand, IE, cc, rows, Bn))
            ax.plot(obs, obs - c, mk, mfc="none" if lab == "ab initio" else None,
                    label=f"{lab} (rms {np.sqrt(np.mean((obs - c) ** 2)):.2f})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("line position / cm$^{-1}$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("obs - calc / cm$^{-1}$")
    fig.tight_layout()
    fig.savefig(fd / "ch4plus_llm_model_residuals.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
