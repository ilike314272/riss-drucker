"""JPL / CDMS SPCAT catalogue (.cat) reader and level derivation.

Fixed format: FREQ(13) ERR(8) LGINT(8) DR(2) ELO(10) GUP(3) TAG(7) QNFMT(4) QN'(12) QN''(12),
frequencies in MHz (cm^-1 when ERR is negative), ELO in cm^-1. A negative TAG marks an
experimentally measured frequency (kept as `measured`). Quantum numbers are
2-character fields; values above 99 use a leading letter (A0 = 100, a0 = -10, ...).

`levels_from_cat` derives upper-state energies E' = E'' + nu and checks every level reached
by several transitions: the spread is a combination-difference consistency certificate.
"""
from __future__ import annotations

from dataclasses import dataclass

MHZ_PER_CM = 29979.2458


def _qn(s: str):
    s = s.strip()
    if not s:
        return None
    if s[0].isalpha():
        if s[0].isupper():
            return 100 + 10 * (ord(s[0]) - ord("A")) + int(s[1:] or 0)
        return -(10 + 10 * (ord(s[0]) - ord("a")) + int(s[1:] or 0))
    return int(s)


@dataclass
class CatLine:
    freq: float           # MHz (or cm^-1 if wavenumber=True)
    err: float
    lgint: float
    elo: float            # cm^-1
    gup: int
    tag: int
    qn_up: tuple
    qn_lo: tuple
    wavenumber: bool = False
    measured: bool = False

    @property
    def nu_cm(self) -> float:
        return self.freq if self.wavenumber else self.freq / MHZ_PER_CM

    @property
    def err_cm(self) -> float:
        return self.err if self.wavenumber else self.err / MHZ_PER_CM


def read_cat(path) -> list[CatLine]:
    out = []
    for raw in open(path, errors="replace"):
        line = raw.rstrip("\n")
        if len(line) < 55:
            continue
        try:
            freq, err, lgint = float(line[0:13]), float(line[13:21]), float(line[21:29])
            elo = float(line[31:41])
            gup = int(line[41:44] or 0)
            tag = int(line[44:51])
            qnfmt = int(line[51:55])
        except ValueError:
            continue
        nq = qnfmt % 10 or 6
        qu = tuple(_qn(line[55 + 2 * i:57 + 2 * i]) for i in range(nq))
        ql = tuple(_qn(line[67 + 2 * i:69 + 2 * i]) for i in range(nq))
        out.append(CatLine(freq, abs(err), lgint, elo, gup, abs(tag), qu, ql, wavenumber=err < 0, measured=tag < 0))
    return out


def levels_from_cat(lines: list[CatLine], *, max_spread: float = 1e-3, measured_only: bool = False) -> tuple[dict, dict]:
    """Energy levels {qn: (E, sigma)} from lower-state energies and E'' + nu.

    PROVENANCE: E'' is the catalogue's computed lower-state energy and unflagged lines are predictions,
    so these levels are model output ('ehamil_derived'), whatever the line flags. The report says how
    many lines were measured. measured_only=True keeps only measured lines (still anchored on the
    catalogue E''); use levels_from_measured for term values from measured frequencies alone.

    Returns (levels, report). A level reached several ways must agree within max_spread
    (cm^-1); the worst spread is reported and violations raise ValueError."""
    n_all = len(lines)
    n_meas = sum(1 for ln in lines if ln.measured)
    if measured_only:
        lines = [ln for ln in lines if ln.measured]
        if not lines:
            raise ValueError(f"no line of the {n_all} in this catalogue is flagged as measured: its levels are "
                             "model output only")
    est = {}
    for ln in lines:
        est.setdefault(ln.qn_lo, []).append((ln.elo, 1e-6))
        est.setdefault(ln.qn_up, []).append((ln.elo + ln.nu_cm, max(ln.err_cm, 1e-7)))
    levels, worst, worst_qn = {}, 0.0, None
    for qn, v in est.items():
        es = [e for e, _ in v]
        spread = max(es) - min(es)
        if spread > worst:
            worst, worst_qn = spread, qn
        e = sum(es) / len(es)
        sig = max(min(s for _, s in v), spread / 2)
        levels[qn] = (e, sig)
    if worst > max_spread:
        raise ValueError(f"combination-difference inconsistency {worst:.2e} cm^-1 at {worst_qn}")
    return levels, {"n_levels": len(levels), "n_lines": len(lines), "max_spread_cm": worst,
                    "worst_level": worst_qn, "n_measured_lines": n_meas, "n_catalogue_lines": n_all,
                    "provenance": "ehamil_derived (catalogue E'' and predicted lines)" if not measured_only else
                    "measured lines anchored on catalogue E''"}


def levels_from_measured(lines: list[CatLine], *, threshold: float = 5.0) -> tuple[dict, dict]:
    """Term values from MEASURED line frequencies only (network inversion, data.network.invert).

    Each connected component of the measured-line network is anchored at one catalogue lower-state
    energy (a single model number per component); every energy difference inside a component is then
    measured. Levels reached by one line only are 'unconfirmed'. Returns ({qn: (E, sigma)}, report)."""
    from .network import invert
    meas = [ln for ln in lines if ln.measured]
    if not meas:
        raise ValueError("no measured lines")
    tr = [{"wavenumber": ln.nu_cm, "uncertainty": max(ln.err_cm, 1e-7), "upper": ln.qn_up, "lower": ln.qn_lo}
          for ln in meas]
    anchors = {}
    for ln in meas:
        anchors.setdefault(ln.qn_lo, ln.elo)
    res = invert(tr, anchors=anchors, threshold=threshold)
    levels = {k: (v["energy"], max(v["uncertainty"], 1e-7) if v.get("uncertainty") is not None else float("nan"))
              for k, v in res.levels.items()}      # nan: uncertainty not computed (anchor / very large component)
    return levels, {"n_levels": len(levels), "n_measured_lines": len(meas), "n_components": len(res.components),
                    "component_sizes": sorted((c["size"] for c in res.components), reverse=True)[:10],
                    "n_unconfirmed": sum(1 for v in res.levels.values() if not v.get("confirmed", True)),
                    "outliers": len(res.outliers),
                    "provenance": "measured line frequencies; one catalogue E'' anchor per network component"}
