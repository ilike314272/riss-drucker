"""M4 data layer: records, ExoMol/MARVEL readers, network inversion, error
prior, round-trip validation, store.

All files here are SYNTHETIC fixtures written by the tests from the library's
own model in the documented formats; no real database content is used."""
import bz2
import json
import math

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker.data import error_prior, exomol, marvel, network, records, roundtrip, store
from riss_drucker.rotor import asym_top_labels

TRUE = {"A": 9.405, "B": 1.2953, "C": 1.1342, "DeltaJ": 2.4e-6, "DeltaJK": 9.7e-5, "DeltaK": 4.4e-3,
        "deltaJ": 3.4e-7, "deltaK": 2.7e-5}
CIT = {"citation": "synthetic test fixture", "extraction_method": "parser"}


@pytest.fixture(scope="module")
def rotor():
    m = rd.compile_asym_rotor(j_max=10, order=4)
    sol = rd.solve(m, TRUE, nlowest=21, keep_vectors=True, method="dense")
    labels = asym_top_labels(m)
    lines = rd.spectra.line_strengths(m, sol, (0.0, 1.0, 0.0))  # b-type: two components (ee/oo, eo/oe)
    lines += rd.spectra.line_strengths(m, sol, (1.0, 0.0, 0.0))  # add a-type: connects everything
    levels = {}
    for b, r in zip(sol.blocks, sol.results):
        for i, e in enumerate(r["eigenvalues"]):
            levels[labels[b["key"]][i]] = float(e)
    return m, levels, lines


# ---- records ----------------------------------------------------------------

def test_units_uncertainty_and_schema():
    assert records.to_cm(29979.2458, "MHz") == pytest.approx(1.0, rel=1e-12)
    assert records.to_one_sigma(0.02, "2sigma") == 0.01
    with pytest.raises(ValueError):
        records.to_one_sigma(0.02, "unknown")
    pytest.importorskip("jsonschema")
    lvl = {"schema_version": 1, "kind": "level", "molecule": {"formula": "H2CO"}, "source": "marvel",
           "provenance": CIT, "units": {"energy": "cm-1"}, "value": 1.0, "uncertainty": 1e-4,
           "uncertainty_definition": "1sigma", "assignment": {"J": 1, "label": "1(0,1)"}}
    records.validate(lvl)
    bad = {**lvl, "kind": "constants"}
    with pytest.raises(Exception):
        records.validate(bad)  # constants need convention + constants
    assert records.usable_as_fit_input(lvl)
    assert not records.usable_as_fit_input({**lvl, "source": "ehamil_derived"})
    assert records.usable_as_fit_input({**lvl, "source": "ehamil_derived"}, allow_ehamil_derived=True)


# ---- ExoMol ---------------------------------------------------------------------

def test_exomol_roundtrip(tmp_path, rotor):
    m, levels, lines = rotor
    labs = sorted(levels, key=lambda k: levels[k])
    ids = {lab: i + 1 for i, lab in enumerate(labs)}
    with open(tmp_path / "X.states", "w") as fh:
        for lab in labs:
            J = int(lab.split("(")[0])
            ka, kc = lab.split("(")[1].rstrip(")").split(",")
            src = "Ma" if J <= 5 else "Ca"
            fh.write(f"{ids[lab]:12d} {levels[lab]:12.6f} {2 * J + 1:6d} {J:7d} {1e-4:12.6f}  Infinity"
                     f" {ka:>2} {kc:>2} {src} {levels[lab] + 0.01:12.6f}\n")
    with bz2.open(tmp_path / "X.trans.bz2", "wt") as fh:
        for l in lines:
            fh.write(f"{ids[l.upper_label]:12d} {ids[l.lower_label]:12d} {l.strength * 1e-3:10.4e}\n")
    # Field names and nesting as in a real ExoMol def.json (checked against H2CO AYTY).
    (tmp_path / "X.def.json").write_text(json.dumps({"dataset": {"states": {"states_file_fields": [
        {"name": n} for n in ["ID", "E", "gtot", "J", "unc", "tau", "Herzberg:Ka", "Herzberg:Kb",
                              "Auxiliary:SourceType", "Auxiliary:Ecal"]]}}}))
    fields = exomol.read_def(tmp_path / "X.def.json")["states_fields"]
    st = exomol.read_states(tmp_path / "X.states", fields)
    assert len(st["i"]) == len(levels) and st["i"].dtype == np.int64
    assert np.all(st["ID"] == st["i"]) and np.all(st["gtot"] == st["g_tot"])
    ma = exomol.read_states(tmp_path / "X.states", fields, keep=lambda t: t[8] == "Ma")
    assert ma["J"].max() == 5 and np.allclose(ma["Auxiliary:Ecal"] - ma["E"], 0.01)
    np.testing.assert_allclose(np.sort(st["E"]), np.sort(list(levels.values())), atol=1e-6)
    low = exomol.read_states(tmp_path / "X.states", fields, j_max=4)
    assert low["J"].max() == 4
    T = 100.0
    Q = exomol.partition_function(st, T)
    tr = list(exomol.iter_trans(tmp_path / "X.trans.bz2", st, T=T, Q=Q))
    assert 0.9 * len(lines) <= len(tr) <= len(lines)  # nu = 0 at file precision is skipped
    t0 = tr[0]
    k = {int(i): n for n, i in enumerate(st["i"])}
    ref = exomol.intensity(t0["A"], st["g_tot"][k[t0["upper"]]], st["E"][k[t0["lower"]]], t0["nu"], T, Q)
    assert t0["S"] == pytest.approx(ref)
    smax = max(t["S"] for t in tr)
    strong = list(exomol.iter_trans(tmp_path / "X.trans.bz2", st, T=T, Q=Q, s_min=0.01 * smax))
    assert 0 < len(strong) < len(tr)
    filtered = list(exomol.iter_trans(tmp_path / "X.trans.bz2", low))  # states filtered -> trans dropped
    assert all(t["upper"] in set(low["i"].tolist()) for t in filtered)


# ---- MARVEL + network ---------------------------------------------------------------

def _marvel_transitions(lines, rng, sigma=1e-4):
    out = []
    for n, l in enumerate(lines):
        qn = lambda lab: (lab.split("(")[0], *lab.split("(")[1].rstrip(")").split(","))
        out.append(marvel.MarvelTransition(l.wavenumber + sigma * rng.standard_normal(), sigma,
                                           qn(l.upper_label), qn(l.lower_label), f"synth.{n}"))
    return out


def test_marvel_files_and_network_inversion(tmp_path, rotor):
    m, levels, lines = rotor
    rng = np.random.default_rng(0)
    trans = _marvel_transitions(lines, rng)
    marvel.write_transitions(tmp_path / "x.txt", trans)
    back = marvel.read_transitions(tmp_path / "x.txt", n_qn=3)
    assert len(back) == len(trans) and back[0].upper == trans[0].upper
    res = network.invert(back, anchors={("0", "0", "0"): 0.0})
    assert len(res.components) == 1 and res.components[0]["anchored"]
    assert not res.outliers
    # Calibration: per-level pulls over independent noise realizations (levels
    # within one network share errors through common paths to the anchor).
    keys = [k for k in res.levels if k != ("0", "0", "0")][:25]
    z = []
    for seed in range(120):
        r = network.invert(_marvel_transitions(lines, np.random.default_rng(100 + seed)),
                           anchors={("0", "0", "0"): 0.0})
        z.append([(r.levels[k]["energy"] - levels[f"{k[0]}({k[1]},{k[2]})"]) / r.levels[k]["uncertainty"]
                  for k in keys])
    z = np.array(z)
    # Levels share path errors, so the effective sample is ~ the number of
    # replications: tolerance ~ 3 / sqrt(2 * 120).
    assert np.all(np.abs(z.mean(axis=0)) < 0.45)
    assert 0.8 < z.std() < 1.2


def test_network_flags_outliers_and_components(rotor):
    m, levels, lines = rotor
    rng = np.random.default_rng(1)
    b_only = [l for l in lines if rd.spectra.band_type(l.lower_label, l.upper_label) == "b"]
    trans = _marvel_transitions(b_only, rng)
    bad = [3, 40, 77]
    for i in bad:
        trans[i].wavenumber += 30e-4 * (1 if i % 2 else -1)
    res = network.invert(trans, anchors={("0", "0", "0"): 0.0})
    # b-type rules connect ee<->oo and eo<->oe only: two components, one floating.
    assert len(res.components) == 2
    assert sum(c["anchored"] for c in res.components) == 1
    assert set(bad) <= set(res.outliers) and len(res.outliers) <= len(bad) + 2
    # A level reached by a single transition cannot be checked by any cycle.
    tri = [marvel.MarvelTransition(1.0, 1e-3, ("b",), ("a",), "t1"), marvel.MarvelTransition(2.0, 1e-3, ("c",), ("a",), "t2"),
           marvel.MarvelTransition(1.0, 1e-3, ("c",), ("b",), "t3"), marvel.MarvelTransition(5.0, 1e-3, ("d",), ("c",), "t4")]
    small = network.invert(tri, anchors={("a",): 0.0})
    assert small.levels[("d",)]["confirmed"] is False and small.levels[("b",)]["confirmed"] is True
    assert small.levels[("d",)]["energy"] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        network.invert([])


# ---- error prior --------------------------------------------------------------------

def test_error_prior_recovers_sigma_model():
    rng = np.random.default_rng(2)
    J = rng.integers(0, 40, 4000).astype(float)
    sig = np.sqrt(0.02**2 + (1e-4 * J * (J + 1)) ** 2)
    e_emp = rng.uniform(0, 5000, len(J))
    e_calc = e_emp - 0.01 + sig * rng.standard_normal(len(J))
    s = error_prior.summarize(e_emp, e_calc, J, band=np.where(J % 2 == 0, "even", "odd"))
    assert s["mean"] == pytest.approx(0.01, abs=0.005)
    mdl = s["sigma_model"]
    assert mdl["a"] == pytest.approx(0.02, rel=0.3) and mdl["b"] == pytest.approx(1e-4, rel=0.15)
    assert len(s["groups"]) > 4
    np.testing.assert_allclose(error_prior.sigma_at(mdl, [0, 30]), [mdl["a"], math.hypot(mdl["a"], mdl["b"] * 930)])


# ---- round trip + store -------------------------------------------------------------

MHZ = records.MHZ_PER_CM
PAPER = {"A": "A", "B": "B", "C": "C", "DJ": "DeltaJ", "DJK": "DeltaJK", "DK": "DeltaK", "dJ": "deltaJ", "dK": "deltaK"}


def _constants_record(theta, unit="MHz", sign=None):
    inv = {v: k for k, v in PAPER.items()}
    consts = {inv[n]: {"value": theta[n] * (MHZ if unit == "MHz" else 1.0) * (sign or {}).get(n, 1.0),
                       "uncertainty": None} for n in theta}
    return {"schema_version": 1, "kind": "constants", "molecule": {"formula": "XY2"}, "source": "measured",
            "provenance": CIT, "units": {"energy": unit}, "uncertainty_definition": "1sigma", "constants": consts,
            "convention": {"model": "asym_rotor", "reduction": "A", "representation": "Ir",
                           "operator_convention": "docs/conventions.md#9", "name_map": PAPER},
            "model_spec": rd.ModelSpec.asym_rotor(j_max=10, order=4).canonical(),
            "fit_statistics": {"rms": 0.05, "n_lines": 0}}  # MHz


def _references(lines, rng, n=150, sigma_mhz=0.05):
    out = []
    for l in lines[:n]:
        out.append({"schema_version": 1, "kind": "transition", "molecule": {"formula": "XY2"}, "source": "measured",
                    "provenance": CIT, "units": {"energy": "MHz"}, "uncertainty_definition": "1sigma",
                    "value": l.wavenumber * MHZ + sigma_mhz * rng.standard_normal(), "uncertainty": sigma_mhz,
                    "upper": {"label": l.upper_label}, "lower": {"label": l.lower_label}})
    return out


def test_round_trip_validates_and_catches_sign_and_unit_errors(rotor):
    m, levels, lines = rotor
    rng = np.random.default_rng(3)
    refs = _references(lines, rng)
    good = roundtrip.validate_constants(_constants_record(TRUE), refs, model=m)
    assert good["validation"]["status"] == "validated", good["validation"]
    flipped = roundtrip.validate_constants(_constants_record(TRUE, sign={"deltaK": -1.0}), refs, model=m)
    assert flipped["validation"]["status"] == "quarantined"
    wrong_unit = _constants_record(TRUE)
    wrong_unit["units"]["energy"] = "cm-1"  # MHz numbers labelled as cm-1
    wrong_unit["fit_statistics"]["rms"] = 0.05 / MHZ
    assert roundtrip.validate_constants(wrong_unit, refs, model=m)["validation"]["status"] == "quarantined"
    typo = _constants_record(TRUE)
    typo["constants"]["DK"]["value"] *= 10  # OCR-style digit error
    assert roundtrip.validate_constants(typo, refs, model=m)["validation"]["status"] == "quarantined"


def test_store_and_fit_from_records(tmp_path, rotor):
    pytest.importorskip("jsonschema")
    m, levels, lines = rotor
    rng = np.random.default_rng(4)
    st = store.RecordStore(tmp_path / "store")
    refs = _references(lines, rng, n=len(lines))
    for r in refs:
        st.add(r)
    st.add({**refs[0], "source": "ehamil_derived"})
    st.quarantine(refs[1], "test")
    assert len(st.records(formula="XY2")) == len(refs) + 1
    assert len(st.records(quarantined=True)) == 1
    obs = st.fit_observations(m, formula="XY2")
    assert len(obs) == len(refs) and st.last_refused == 1  # ehamil_derived refused
    f = rd.fit(m, obs, rd.synth.perturb_theta(TRUE, 1e-4, seed=5))
    assert f.converged and max(abs(f.theta[n] - TRUE[n]) / f.sigma[n] for n in TRUE) < 5
