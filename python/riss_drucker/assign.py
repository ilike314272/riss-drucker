"""Line assignment with calibrated confidence (README 5.5).

For each predicted line i and observed peak j inside a chi^2 gate
(z^2 = (nu_j - nu_i)^2 / (sigma_j^2 + sigma_pred_i^2) < gate), the likelihood
ratio against "peak j is background" is

    LR_ij = N(nu_j - nu_i; 0, sigma^2) / rho_j  x  N(log I_j - log(s I_i); 0, s_l^2) / p_bg(log I_j)

with rho_j the local peak density (the Poisson background) and s a global
intensity scale. Marginal posteriors use a detection probability p_d:
    P(i -> j) = p_d LR_ij / ((1 - p_d) + p_d sum_j' LR_ij'),  P(i -> none) likewise.
The joint assignment is a monotone alignment (dynamic programming) within each
series: observed order follows predicted order, one peak per line. Conflicts
between series go to the higher posterior and are flagged.

Reported per line: assigned peak, its posterior, the credible (ambiguity) set,
the chance-coincidence probability P(>=1 random peak within +-delta) =
1 - exp(-2 delta rho), and Benjamini-Hochberg acceptance across the band.
The sum of (1 - posterior) over assigned lines is the expected number of
misassignments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Peak:
    position: float
    sigma: float
    intensity: float | None = None
    sigma_log_intensity: float = 0.3
    truth: object = None          # synthetic data only: id of the true line, or None


@dataclass
class PredictedLine:
    id: str
    position: float
    sigma: float                  # predictive sigma of the position (from Sigma_theta)
    intensity: float | None = None
    series: str = ""
    lower_label: str | None = None
    upper_label: str | None = None
    upper: tuple | None = None    # (block key, index)
    lower: tuple | None = None


def merge_blends(lines: Sequence[PredictedLine], resolution: float) -> list[PredictedLine]:
    """Merge lines closer than `resolution` (chained) into one blended line:
    intensity-weighted position, summed intensity, id 'a+b' (sorted members).
    Unresolved K-doublets are one peak in a spectrum and are assigned as one."""
    srt = sorted(lines, key=lambda l: l.position)
    groups: list[list[PredictedLine]] = []
    for l in srt:
        if groups and l.position - groups[-1][-1].position < resolution:
            groups[-1].append(l)
        else:
            groups.append([l])
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        w = np.array([max(x.intensity or 0.0, 0.0) for x in g])
        w = w / w.sum() if w.sum() > 0 else np.full(len(g), 1.0 / len(g))
        strongest = g[int(np.argmax(w))]
        out.append(PredictedLine(
            id="+".join(sorted(x.id for x in g)), position=float(sum(wi * x.position for wi, x in zip(w, g))),
            sigma=float(max(x.sigma for x in g)),
            intensity=float(sum(x.intensity or 0.0 for x in g)) if all(x.intensity for x in g) else None,
            series=strongest.series, lower_label=strongest.lower_label, upper_label=strongest.upper_label,
            upper=None, lower=None))
    return out


def chance_probability(delta: float, density: float) -> float:
    """P(>= 1 background peak within +-delta) for a Poisson density rho."""
    return 1.0 - math.exp(-2.0 * abs(delta) * density)


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> list[bool]:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    accept = np.zeros(m, bool)
    accept[order[:k]] = True
    return accept.tolist()


@dataclass
class AssignmentResult:
    lines: list[dict]
    expected_misassignments: float
    n_assigned: int
    n_accepted: int
    n_ambiguous: int
    intensity_scale: float | None
    settings: dict = field(default_factory=dict)


def _local_density(sorted_pos: np.ndarray, at: float, half_width: float) -> float:
    lo, hi = np.searchsorted(sorted_pos, [at - half_width, at + half_width])
    return max(int(hi - lo), 1) / (2.0 * half_width)


def assign(predicted: Sequence[PredictedLine], peaks: Sequence[Peak], *, gate: float = 10.83,
           p_detect: float = 0.9, use_intensity: bool = True, sigma_log_model: float = 0.3,
           credible: float = 0.95, fdr: float = 0.05, density_half_width: float | None = None) -> AssignmentResult:
    pred = list(predicted)
    pk = list(peaks)
    pos = np.array([p.position for p in pk])
    order = np.argsort(pos)
    spos = pos[order]
    if density_half_width is None:
        sp = np.diff(spos)
        density_half_width = 25.0 * float(np.median(sp[sp > 0])) if len(sp) else 1.0
    rho = np.array([_local_density(spos, x, density_half_width) for x in pos])

    use_int = use_intensity and all(p.intensity and p.intensity > 0 for p in pk) and \
        all(l.intensity and l.intensity > 0 for l in pred)
    logI = np.log([p.intensity for p in pk]) if use_int else None
    bg_mu, bg_sd = (float(logI.mean()), float(logI.std() + 1e-12)) if use_int else (0.0, 1.0)

    # Gated candidates (window search on sorted positions).
    max_ps = max((p.sigma for p in pk), default=0.0)
    cand: list[list[tuple[int, float]]] = []
    for ln in pred:
        half = math.sqrt(gate * (ln.sigma ** 2 + max_ps ** 2))
        lo, hi = np.searchsorted(spos, [ln.position - half, ln.position + half])
        c = []
        for j in order[lo:hi]:
            d = pk[j].position - ln.position
            if d * d < gate * (pk[j].sigma ** 2 + ln.sigma ** 2):
                c.append((int(j), d))
        cand.append(c)

    # Global intensity scale from mutually unique gated pairs (robust median).
    scale = None
    if use_int:
        owners: dict[int, list[int]] = {}
        for i, c in enumerate(cand):
            for j, _ in c:
                owners.setdefault(j, []).append(i)
        ratios = [math.log(pk[c[0][0]].intensity / pred[i].intensity) for i, c in enumerate(cand)
                  if len(c) == 1 and len(owners[c[0][0]]) == 1]
        if not ratios:
            ratios = [math.log(pk[j].intensity / pred[i].intensity) for i, c in enumerate(cand) for j, _ in c]
        scale = math.exp(float(np.median(ratios))) if ratios else 1.0

    def log_lr(i, j, d):
        s2 = pk[j].sigma ** 2 + pred[i].sigma ** 2
        ll = -0.5 * d * d / s2 - 0.5 * math.log(2 * math.pi * s2) - math.log(rho[j])
        if use_int:
            sl2 = pk[j].sigma_log_intensity ** 2 + sigma_log_model ** 2
            x = logI[j] - math.log(scale * pred[i].intensity)
            ll += -0.5 * x * x / sl2 - 0.5 * math.log(sl2)
            ll -= -0.5 * ((logI[j] - bg_mu) / bg_sd) ** 2 - math.log(bg_sd)
        return ll

    # Marginal posteriors per line.
    marg = []
    for i, c in enumerate(cand):
        lls = np.array([log_lr(i, j, d) for j, d in c])
        mx = max(lls.max() if len(lls) else -np.inf, 0.0)
        num = p_detect * np.exp(lls - mx)
        none = (1 - p_detect) * math.exp(-mx)
        z = none + num.sum()
        marg.append({"none": none / z, **{j: float(v / z) for (j, _), v in zip(c, num)}})

    # Monotone alignment per series.
    assigned: dict[int, int] = {}
    series: dict[str, list[int]] = {}
    for i, ln in enumerate(pred):
        series.setdefault(ln.series, []).append(i)
    for idx in series.values():
        idx = sorted(idx, key=lambda i: pred[i].position)
        peaks_s = sorted({j for i in idx for j, _ in cand[i]}, key=lambda j: pk[j].position)
        if not peaks_s:
            continue
        col = {j: b for b, j in enumerate(peaks_s)}
        na, nb = len(idx), len(peaks_s)
        score = np.full((na, nb), -np.inf)
        for a, i in enumerate(idx):
            for j, d in cand[i]:
                pij, pn = marg[i][j], marg[i]["none"]
                score[a, col[j]] = math.log(max(pij, 1e-300)) - math.log(max(pn, 1e-300))
        D = np.zeros((na + 1, nb + 1))
        for a in range(1, na + 1):
            for b in range(1, nb + 1):
                D[a, b] = max(D[a - 1, b], D[a, b - 1], D[a - 1, b - 1] + score[a - 1, b - 1])
        a, b = na, nb
        while a > 0 and b > 0:
            if D[a, b] == D[a - 1, b]:
                a -= 1
            elif D[a, b] == D[a, b - 1]:
                b -= 1
            else:
                assigned[idx[a - 1]] = peaks_s[b - 1]
                a, b = a - 1, b - 1

    # Cross-series conflicts: one peak, one line (highest posterior).
    conflict = set()
    by_peak: dict[int, list[int]] = {}
    for i, j in assigned.items():
        by_peak.setdefault(j, []).append(i)
    for j, owners in by_peak.items():
        if len(owners) > 1:
            best = max(owners, key=lambda i: marg[i][j])
            for i in owners:
                conflict.add(i)
                if i != best:
                    del assigned[i]

    rows, pvals, assigned_rows = [], [], []
    for i, ln in enumerate(pred):
        opts = sorted(marg[i].items(), key=lambda kv: -kv[1])
        cset, cum = [], 0.0
        for k, v in opts:
            cset.append(k)
            cum += v
            if cum >= credible:
                break
        j = assigned.get(i)
        row = {"line": ln.id, "series": ln.series, "predicted": ln.position, "sigma_pred": ln.sigma,
               "lower_label": ln.lower_label, "upper_label": ln.upper_label,
               "peak": j, "observed": pk[j].position if j is not None else None,
               "posterior": marg[i][j] if j is not None else marg[i]["none"],
               "posterior_none": marg[i]["none"], "n_candidates": len(cand[i]),
               "credible_set": cset, "ambiguous": len(cset) > 1, "conflict": i in conflict,
               "chance_probability": None, "bh_accepted": False}
        if j is not None:
            d = pk[j].position - ln.position
            row["chance_probability"] = chance_probability(d, rho[j])
            pvals.append(row["chance_probability"])
            assigned_rows.append(row)
        rows.append(row)
    for row, ok in zip(assigned_rows, benjamini_hochberg(pvals, fdr)):
        row["bh_accepted"] = bool(ok)
    exp_mis = float(sum(1.0 - r["posterior"] for r in assigned_rows))
    return AssignmentResult(rows, exp_mis, len(assigned_rows), sum(r["bh_accepted"] for r in assigned_rows),
                            sum(r["ambiguous"] for r in rows), scale,
                            {"gate": gate, "p_detect": p_detect, "fdr": fdr, "credible": credible,
                             "density_half_width": density_half_width, "use_intensity": use_int})


def predict_lines(model, theta, mu, T, *, covariance=None, min_relative_intensity: float = 0.0,
                  spin_weight=None, solution=None) -> tuple[list[PredictedLine], list]:
    """Predicted lines with positions, predictive sigma (Hellmann-Feynman + Sigma_theta)
    and relative intensities. Series = (Delta J, Ka'' -> Ka')."""
    from .rotor import asym_top_labels
    from .solve import solve
    from .spectra import add_intensities, line_strengths

    labels = asym_top_labels(model)
    if solution is None:
        solution = solve(model, theta, nlowest=max(b["dim"] for b in model.blocks), keep_vectors=True,
                         method="dense")
    lines = line_strengths(model, solution, mu)
    add_intensities(lines, solution, T, spin_weight=spin_weight, labels=labels)
    imax = max(l.intensity for l in lines) if lines else 1.0
    lines = [l for l in lines if l.intensity >= min_relative_intensity * imax]
    keys = [b["key"] for b in model.blocks]
    cov = None if covariance is None else np.asarray(covariance)
    jac = {}
    out = []
    for n, l in enumerate(lines):
        s = 0.0
        if cov is not None:
            g = []
            for key, idx in (l.upper, l.lower):
                bi = keys.index(key)
                if bi not in jac:
                    jac[bi] = model.jacobian(bi, np.asarray(solution.results[bi]["eigenvectors"]))
                g.append(jac[bi][idx])
            dg = g[0] - g[1]
            s = math.sqrt(max(dg @ cov @ dg, 0.0))
        ka = lambda lab: int(lab.split("(")[1].split(",")[0])
        dj = {-1: "P", 0: "Q", 1: "R"}[l.j_upper - l.j_lower]
        out.append(PredictedLine(id=f"{l.upper_label}<-{l.lower_label}", position=l.wavenumber, sigma=s,
                                 intensity=l.intensity, series=f"{dj}:Ka{ka(l.lower_label)}->{ka(l.upper_label)}",
                                 lower_label=l.lower_label, upper_label=l.upper_label, upper=l.upper, lower=l.lower))
    return out, lines


def line_observations(result: "AssignmentResult", predicted: Sequence[PredictedLine], peaks: Sequence[Peak], *,
                      accepted_only: bool = True, min_posterior: float = 0.95):
    """Confident, unblended assignments as transition observations for `fit`."""
    from .fit import Observation

    by_id = {p.id: p for p in predicted}
    out = []
    for r in result.lines:
        if r["peak"] is None or "+" in r["line"] or r["posterior"] < min_posterior:
            continue
        if accepted_only and not r["bh_accepted"]:
            continue
        p = by_id[r["line"]]
        pk = peaks[r["peak"]]
        out.append(Observation(block=p.upper[0], index=p.upper[1], energy=pk.position, sigma=pk.sigma,
                               label=p.id, lower_block=p.lower[0], lower_index=p.lower[1]))
    return out
