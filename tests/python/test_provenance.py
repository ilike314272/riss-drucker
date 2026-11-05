"""Catalogue provenance (method review, finding 1): predicted lines are model output; measured lines
give term values by network inversion."""
import numpy as np
import pytest


def _lines(measured):
    from riss_drucker.data.catalog import CatLine
    E = {(0,): 0.0, (1,): 2.0, (2,): 6.0, (3,): 12.0}           # rigid rotor B = 1 cm^-1
    out = []
    for lo, up in (((0,), (1,)), ((1,), (2,)), ((2,), (3,)), ((0,), (2,))):
        out.append(CatLine(freq=E[up] - E[lo], err=1e-4, lgint=-3, elo=E[lo], gup=1, tag=-1 if measured else 1,
                           qn_up=up, qn_lo=lo, wavenumber=True, measured=measured))
    return out


def test_predicted_only_catalogue_is_refused_when_measured_only():
    from riss_drucker.data.catalog import levels_from_cat
    lv, rep = levels_from_cat(_lines(False))
    assert rep["n_measured_lines"] == 0 and rep["provenance"].startswith("ehamil_derived")
    with pytest.raises(ValueError, match="model output"):
        levels_from_cat(_lines(False), measured_only=True)


def test_levels_from_measured_network():
    from riss_drucker.data.catalog import levels_from_measured
    lines = _lines(True)
    lines[1].freq += 3e-4                       # a measured frequency differing from the model
    lv, rep = levels_from_measured(lines)
    assert rep["n_components"] == 1 and rep["n_levels"] == 4
    cat = {(0,): 0.0, (1,): 2.0, (2,): 6.0}
    anchored = [k for k, e in cat.items() if abs(lv[k][0] - e) < 1e-9]
    assert len(anchored) == 1                   # one catalogue anchor per network component
    # energy differences follow the measured frequencies (the 3e-4 shift is shared over the 0-1-2 / 0-2 cycle)
    assert abs(lv[(3,)][0] - lv[(2,)][0] - 6.0) < 1e-9
    assert 0 < abs((lv[(2,)][0] - lv[(1,)][0]) - 4.0) <= 3e-4 + 1e-12


def test_coverage_helper():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "review_followups.py"
    spec = importlib.util.spec_from_file_location("rf", p)
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    z = np.random.default_rng(1).normal(size=20000)
    c = rf.coverage(z)
    assert abs(c["within_1sigma"] - 0.683) < 0.01 and abs(c["within_2sigma"] - 0.954) < 0.01
    assert abs(rf.coverage(3 * z)["rms_z"] - 3) < 0.05
