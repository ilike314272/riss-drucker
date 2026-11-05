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
from riss_drucker import llm_loop as LL  # noqa: E402
from riss_drucker.llm_loop import (Heartbeat, LimitReached, TransientError, classify, parse_answer,  # noqa: E402,F401
                                   replay_complete, resumed, say, strip_private, with_retries)


def claude_cli_complete(model=None, timeout=1800, budget_left=None):
    return LL.claude_cli_complete(SYSTEM, model, timeout, budget_left)


def api_complete(model):
    return LL.api_complete(SYSTEM, model)


def describe_proposal(prop: dict) -> str:
    sp = prop.get("spec", {})
    lines = [f"group {sp.get('group')}, N+ = {sp.get('J')}"]
    for s in sp.get("spaces", []):
        if s.get("kind") == "well":
            lines.append(f"space: {len(s.get('vectors', []))} wells carrying {s.get('irrep')} electronic functions, "
                         f"geometric phase {'ON' if s.get('geometric_phase', True) else 'OFF'}")
        else:
            lines.append(f"space: {s.get('kind')}")
    roles = prop.get("roles", {})
    for term in sp.get("terms", []):
        name = term.split(":")[0].strip()
        lines.append(f"term [{roles.get(name, 'extra'):9s}] {term}")
    r = (prop.get("rationale") or "").strip()
    if r:
        lines.append("Claude's reasoning: " + (r if len(r) < 900 else r[:900] + " ..."))
    return "\n".join(lines)


def describe_score(sc: dict) -> str:
    out = []
    for n, _ in ISOTOPES:
        s = sc[n]
        p = s["posterior"]
        out.append(f"{n}: ab initio prediction rms {s['prediction_rms_IE_only_cm']:.2f} cm-1 -> after update "
                   f"{s['updated_rms_cm']:.2f}" + (f", cross-validated {s['cv_rms_cm']:.2f}" if "cv_rms_cm" in s else "")
                   + f"; A B C = {p['A']:.3f} {p['B']:.3f} {p['C']:.3f}, delta = {s['delta_posterior_cm']:.2f} cm-1"
                   + f" (sigma moved {s['shift_in_prior_sd']['sigma']:+.2f} prior sd)"
                   + f" ({s['n_branches_consistent_with_geometry']} of {s['n_calibration_branches']} calibrations fit the geometry)")
    return "\n".join(out)


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


# ----------------------------------------------------------------------------- adapters
def _check_roles(prop):
    if "roles" not in prop:
        raise ValueError("answer lacks 'roles' (rotor / tunneling / extra for each term)")


def relation_to_reference(cand, k_c, ref_c, k_r):
    core = all(equivalent(cand, k_c[n], ref_c, k_r[n]) for n, _ in ISOTOPES)
    if core and not cand.extra:
        return "equivalent to the reference model"
    if core:
        return f"reference model + {len(cand.extra)} extra term(s) {cand.extra}"
    return "a different model from the reference"


class ExperimentAdapter:
    """Score proposals against the measured PFI-ZEKE lines (ab initio prior, Bayesian update, CV)."""
    system = SYSTEM

    def __init__(self, facts, tables, cv=True):
        from riss_drucker import symmetry as S
        self.facts, self.tables, self.cv = facts, tables, cv
        self.classes = observed_classes(tables)
        self.group_table = S.get_group("Td").character_table()

    def reference(self):
        return reference_proposal()

    def prompt(self, history):
        return build_prompt(self.facts, self.classes, self.group_table, history)

    def describe(self, prop):
        return describe_proposal(prop)

    def evaluate(self, prop, log=say):
        _check_roles(prop)
        log("compiling and certifying (hermiticity, time reversal, Td symmetry, level counts) ...", 2)
        cand = Candidate(prop["spec"], prop["roles"])
        log(f"certified: {len(cand.bm.blocks)} symmetry blocks; calibrating constants from the ab initio surface "
            "and scoring against the measured lines ...", 2)
        sc = score_candidate(cand, self.facts, self.tables, self.cv)
        sc["score"] = sc["score_cv_rms_mean"]
        sc["summary"] = describe_score(sc)
        return cand, sc

    def feedback(self, sc):
        fb = {n: {kk: sc[n][kk] for kk in ("n_calibration_branches", "n_branches_consistent_with_geometry",
                                           "prediction_rms_IE_only_cm", "updated_rms_cm", "cv_rms_cm",
                                           "delta_posterior_cm", "posterior", "shift_in_prior_sd") if kk in sc[n]}
              for n, _ in ISOTOPES}
        fb["note"] = ("delta here is 4|sigma|; with extra J-dependent tunneling couplings the N+ > 0 pattern can be "
                      "absorbed by them, so check whether sigma is still determined by the data "
                      "(shift_in_prior_sd of sigma near 0 means it is not)")
        return fb

    def relation(self, cand, sc, ref_c, ref_s):
        return relation_to_reference(cand, {n: sc[n]["ab_initio_coefficients"] for n, _ in ISOTOPES}, ref_c,
                                     {n: ref_s[n]["ab_initio_coefficients"] for n, _ in ISOTOPES})

    def final(self, label, cand, sc, prop, ref_s, run):
        return {"mode": "experiment", "selected": label, "stop_reason": run["stop_reason"],
                "rounds_done": len(run["rounds"]), "usage_total": run["usage"],
                "selection_rule": "lowest mean leave-one-position-out rms over CH4+ and CD4+ (Bayesian update with "
                                  "ab initio prior); the language model's proposal is preferred on ties with the reference",
                "spec": cand.spec.to_dict(), "roles": cand.roles, "rationale": prop.get("rationale"),
                "abinitio_basis": self.facts["basis"], "scores": strip_private(sc),
                "reference_scores": strip_private(ref_s) if ref_s else None,
                "levels_N_le_5": {n: [{"N": N, "irrep": ir, "k": kk, "E_cm": e}
                                      for (N, ir, kk), e in sorted(levels(cand, sc[n]["posterior_coefficients"]).items(),
                                                                   key=lambda t: t[1])] for n, _ in ISOTOPES}}


FP_TASK = ("Build the effective Hamiltonian for the vibronic ground state of CH4+ (rotation x tunneling between the "
           "equivalent Jahn-Teller minima). Its target is the set of FIRST-PRINCIPLES rovibronic levels computed from "
           "the ab initio surface (not experiment): they include the vibrationally averaged rotor of each well, the "
           "tunneling splitting from a variational reaction-plane calculation, and the rotation of the molecule-fixed "
           "frame accumulated along the tunneling path (a hop between neighbouring wells carries a rotation by "
           "theta about axis n, i.e. the operator exp(-i theta n.J); its expansion gives rotation-tunneling couplings "
           "of the form i*sym(w[1,j] (a.J)) and J-dependent tunneling). Your terms' coefficients are FITTED to the "
           "first-principles levels with N+ <= nfit and the model is SCORED by how well it then predicts the levels "
           "with N+ > nfit (rms, cm-1). The measured spectrum is used only for an independent validation report. "
           "Choose spaces (including well vectors), group and terms; fewer terms that extrapolate well are better.")


class FirstPrinciplesAdapter:
    """Score proposals by fitting them to first-principles levels (N+ <= nfit) and predicting N+ > nfit."""
    system = SYSTEM

    def __init__(self, facts, fp: dict, tables=None, nfit=5):
        from riss_drucker import symmetry as S
        from riss_drucker.abinitio.firstprinciples import levels_from_json
        self.facts, self.fp, self.tables, self.nfit = facts, fp, tables, nfit
        self.nmax = fp["nmax"]
        self.target = {n: levels_from_json(fp["isotopologues"][n]["levels"]) for n, _ in ISOTOPES}
        self.group_table = S.get_group("Td").character_table()

    def reference(self):
        return reference_proposal()

    def describe(self, prop):
        return describe_proposal(prop)

    def prompt(self, history):
        from riss_drucker.llm import GRAMMAR
        fp_facts = {}
        for n, _ in ISOTOPES:
            e = self.fp["isotopologues"][n]
            ta = e["tunneling"]
            fp_facts[n] = {
                "tunneling_delta_cm (= 4|sigma|)": round(ta["delta"], 4),
                "sign_of_sigma": "N+=0 ground level is F2 (Ham's theorem)",
                "vibrationally_averaged_rotor_tensor_cm (Q_ab J_a J_b, tetrahedral reference frame, well with vector L)":
                    np.round(np.array(ta["rotor_tensor_averaged_td"]), 5).tolist(),
                "averaged_A_B_C_cm": np.round(ta["rotational_constants_averaged"], 4).tolist(),
                "frame_rotation_per_hop": {"angle_deg": round(ta["frame_rotation"]["angle_deg"], 3),
                                           "axis (reference frame)": np.round(ta["frame_rotation"]["axis"], 4).tolist(),
                                           "from_well_vector_L": np.round(ta["well_vector_L"], 4).tolist(),
                                           "to_well_vector_R": np.round(ta["well_vector_R"], 4).tolist()},
                "first_principles_levels_N_le_2 (cm-1, relative to the N+=0 ground level)":
                    [{"N": N, "irrep": ir, "E": round(v, 4)} for (N, ir, k), v in
                     sorted(self.target[n].items(), key=lambda t: t[1]) if N <= 2]}
        payload = {"task": FP_TASK.replace("nfit", str(self.nfit)), "ab_initio": self.facts,
                   "first_principles": fp_facts, "levels_fitted_N_le": self.nfit, "levels_predicted_N": [self.nfit + 1, self.nmax],
                   "grammar": GRAMMAR + "\n" + WELL_GRAMMAR, "group_Td_character_table": self.group_table,
                   "previous_attempts_and_feedback": history, "answer_format": ANSWER_FORMAT}
        return json.dumps(payload, indent=1, default=str)

    def _fit(self, cand, name, log):
        from scipy.optimize import least_squares
        ta = self.fp["isotopologues"][name]["tunneling"]
        tgt = self.target[name]
        A, Bc, C = ta["rotational_constants_averaged"]
        lv0 = levels(cand, cand.theta({p: 0.0 for p in cand.params}))
        if set(lv0) != set(tgt):
            from collections import Counter
            cm = Counter((N, ir) for N, ir, _ in lv0)
            ct = Counter((N, ir) for N, ir, _ in tgt)
            diff = {f"N={N} {ir}": (cm.get((N, ir), 0), ct.get((N, ir), 0)) for N, ir in set(cm) | set(ct)
                    if cm.get((N, ir), 0) != ct.get((N, ir), 0)}
            raise ValueError(f"level structure differs from the first-principles levels (model, target counts): "
                             f"{dict(sorted(diff.items())[:8])}")
        keys_fit = sorted(k for k in tgt if k[0] <= self.nfit)
        keys_pred = sorted(k for k in tgt if k[0] > self.nfit)
        cal_rms, branches = cand.calibrate(A, Bc, C)
        if not branches:
            raise ValueError(f"rotor terms cannot reproduce the averaged rigid rotor (calibration rms {cal_rms:.3g})")
        names = cand.params

        def resid(x, keys):
            lv = levels(cand, dict(zip(names, x)))
            return np.array([lv[k] - tgt[k] for k in keys])
        best = None
        for br in branches:
            for s in (1.0, -1.0):
                x0 = np.array([br["coeffs"].get(p, 0.0) if p in cand.rotor else
                               (s * ta["sigma_abs"] / cand.hop if p == cand.tun else 0.0) for p in names])
                sol = least_squares(resid, x0, args=(keys_fit,), x_scale=np.maximum(np.abs(x0), 0.05))
                if best is None or sol.cost < best.cost:
                    best = sol
        x = best.x
        fit_rms = float(np.sqrt(np.mean(resid(x, keys_fit) ** 2)))
        pred_rms = float(np.sqrt(np.mean(resid(x, keys_pred) ** 2))) if keys_pred else float("nan")
        coeffs = dict(zip(names, map(float, x)))
        out = {"fit_rms_cm": fit_rms, "predict_rms_cm": pred_rms, "n_fit": len(keys_fit), "n_predict": len(keys_pred),
               "coefficients": coeffs, "delta_cm": 4 * abs(coeffs[cand.tun]) * cand.hop}
        if self.tables is not None:
            import case_methane_cation_rovibronic as R
            rows = [r for r in self.tables[name] if r["N"] and "/" not in r["N"]]
            Bn = R.NEUTRAL_B[name]
            e = self.fp["isotopologues"][name]
            IE = e["ionization_energy_abinitio_cm"] + e.get("validation", {}).get("IE_offset_cm", 0.0)
            obs = np.array([r["nu"] for r in rows])
            for _ in range(5):
                asg = assign(cand, IE, coeffs, rows, Bn)
                c = calc(cand, IE, coeffs, rows, Bn, asg)
                IE += float(np.mean(obs - c))
            c = calc(cand, IE, coeffs, rows, Bn, assign(cand, IE, coeffs, rows, Bn))
            out["validation_rms_IE_shifted_cm"] = float(np.sqrt(np.mean((obs - c) ** 2)))
            out["validation_IE_offset_cm"] = IE - e["ionization_energy_abinitio_cm"]
            out["_IE"] = IE
        # fields read by plot_ch4plus_sticks.py
        out.update({"IE_offset": out.get("_IE", self.fp["isotopologues"][name]["ionization_energy_abinitio_cm"]),
                    "ab_initio_coefficients": coeffs, "posterior_coefficients": coeffs,
                    "posterior": {"IE": out.get("_IE", 0.0)}, "prediction_rms_IE_only_cm": out.get("validation_rms_IE_shifted_cm", float("nan")),
                    "updated_rms_cm": out.get("validation_rms_IE_shifted_cm", float("nan")),
                    "delta_posterior_cm": out["delta_cm"]})
        return out

    def evaluate(self, prop, log=say):
        _check_roles(prop)
        d = dict(prop["spec"])
        d["J"] = list(range(self.nmax + 1))
        log("compiling and certifying (hermiticity, time reversal, Td symmetry, level counts) ...", 2)
        cand = Candidate(d, prop["roles"])
        log(f"certified: {len(cand.bm.blocks)} symmetry blocks; fitting to the first-principles levels N+ <= {self.nfit} "
            f"and predicting N+ = {self.nfit + 1}..{self.nmax} ...", 2)
        sc = {n: self._fit(cand, n, log) for n, _ in ISOTOPES}
        sc["score"] = float(np.mean([sc[n]["predict_rms_cm"] for n, _ in ISOTOPES]))
        sc["summary"] = "\n".join(
            f"{n}: fit rms {sc[n]['fit_rms_cm']:.4f} cm-1 ({sc[n]['n_fit']} levels N+<={self.nfit}), predicted N+>{self.nfit} "
            f"rms {sc[n]['predict_rms_cm']:.4f} ({sc[n]['n_predict']} levels); delta {sc[n]['delta_cm']:.3f}"
            + (f"; measured lines (IE shifted only) rms {sc[n]['validation_rms_IE_shifted_cm']:.2f}"
               if "validation_rms_IE_shifted_cm" in sc[n] else "") for n, _ in ISOTOPES)
        return cand, sc

    def feedback(self, sc):
        return {n: {k: v for k, v in sc[n].items() if not k.startswith("_") and k not in
                    ("ab_initio_coefficients", "posterior_coefficients", "posterior", "IE_offset")} for n, _ in ISOTOPES}

    def relation(self, cand, sc, ref_c, ref_s):
        return relation_to_reference(cand, {n: sc[n]["coefficients"] for n, _ in ISOTOPES}, ref_c,
                                     {n: ref_s[n]["coefficients"] for n, _ in ISOTOPES})

    def final(self, label, cand, sc, prop, ref_s, run):
        return {"mode": "firstprinciples", "selected": label, "stop_reason": run["stop_reason"],
                "rounds_done": len(run["rounds"]), "usage_total": run["usage"],
                "selection_rule": f"lowest mean rms of predicted first-principles levels N+ > {self.nfit} after fitting "
                                  f"N+ <= {self.nfit}; the measured spectrum is not used for selection",
                "spec": cand.spec.to_dict(), "roles": cand.roles, "rationale": prop.get("rationale"),
                "abinitio_basis": self.facts["basis"], "first_principles_file_basis": self.fp.get("basis"),
                "scores": strip_private(sc), "reference_scores": strip_private(ref_s) if ref_s else None,
                "levels": {n: [{"N": N, "irrep": ir, "k": kk, "E_cm": e}
                               for (N, ir, kk), e in sorted(levels(cand, sc[n]["coefficients"]).items(),
                                                            key=lambda t: t[1])] for n, _ in ISOTOPES}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abinitio", required=True)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--target", choices=["experiment", "firstprinciples"], default="experiment",
                    help="experiment: score against the measured lines; firstprinciples: fit to the first-principles "
                         "levels (--fp) and score by prediction of higher N+ (measured lines only validate)")
    ap.add_argument("--fp", default=None, help="first-principles levels from scripts/firstprinciples_ch4plus.py")
    ap.add_argument("--nfit", type=int, default=5, help="firstprinciples: fit levels with N+ <= nfit")
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

    facts = abinitio_facts(Path(a.abinitio))
    tables = None
    if a.pdf:
        import case_methane_cation_rovibronic as R
        tables = R.read_tables(Path(a.pdf))
    if a.target == "firstprinciples":
        if not a.fp:
            ap.error("--target firstprinciples needs --fp (from scripts/firstprinciples_ch4plus.py)")
        adapter = FirstPrinciplesAdapter(facts, json.loads(Path(a.fp).read_text()), tables, a.nfit)
    else:
        if tables is None and not a.dry_run:
            ap.error("--pdf is required with --target experiment (proposals are scored against the measured lines)")
        if tables is None:
            from riss_drucker import symmetry as S
            print(build_prompt(facts, None, S.get_group("Td").character_table(), []))
            return 0
        adapter = ExperimentAdapter(facts, tables, cv=not a.no_cv)
    if a.dry_run:
        print(adapter.prompt([]))
        return 0
    make_live = {"claude-cli": lambda bl: claude_cli_complete(a.model, budget_left=bl),
                 "api": lambda bl: api_complete(a.model or "claude-opus-5-5"),
                 "replay": lambda bl: replay_complete(a.replay)}[a.llm]
    run = LL.run_loop(adapter, make_live, rounds=a.rounds, transcript=a.transcript, max_budget_usd=a.max_budget_usd,
                      max_total_tokens=a.max_total_tokens, retries=a.retries, retry_wait=a.retry_wait,
                      resume=a.resume, backend=a.llm, model=a.model)
    label, cand, sc, prop = run["best"]
    ref_s = run["reference"][1]
    final = adapter.final(label, cand, sc, prop, ref_s, run)
    Path(a.out).write_text(json.dumps(final, indent=1, default=str))
    short = {"selected": label, "stop_reason": run["stop_reason"], "usage_total": run["usage"], "score": sc["score"],
             "reference_score": ref_s["score"] if ref_s else None}
    for n, _ in ISOTOPES:
        short[n] = {k: v for k, v in sc[n].items() if k in ("prediction_rms_IE_only_cm", "updated_rms_cm", "cv_rms_cm",
                                                             "posterior", "delta_posterior_cm", "fit_rms_cm",
                                                             "predict_rms_cm", "validation_rms_IE_shifted_cm", "delta_cm")}
    print(json.dumps(short, indent=1, default=str))
    if a.figures and a.target == "experiment":
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
