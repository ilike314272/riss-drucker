"""ModelSpec hashing and the operator cache (README 5.1, 5.2, 12 regression)."""
import json
import subprocess
import sys

import numpy as np
import pytest

import riss_drucker as rd
from riss_drucker.cache import CacheError, cache_path, compile_cached


def test_hash_is_canonical_and_sensitive():
    a = rd.ModelSpec.asym_rotor(j_max=10, order=4, extra_operators=["J2"])
    b = rd.ModelSpec.asym_rotor(order=4, j_max=10, extra_operators=("J2",))
    assert a.hash() == b.hash()
    assert len(a.hash()) == 64
    for other in (rd.ModelSpec.asym_rotor(j_max=11, order=4, extra_operators=["J2"]),
                  rd.ModelSpec.asym_rotor(j_max=10, order=6, extra_operators=["J2"]),
                  rd.ModelSpec.asym_rotor(j_max=10, order=4, axis="IIIr", extra_operators=["J2"]),
                  rd.ModelSpec.asym_rotor(j_max=10, order=4, reduction="S", extra_operators=["J2"])):
        assert other.hash() != a.hash()
    assert rd.ModelSpec.torsion_rotor(v_orders=[2, 1]).hash() == rd.ModelSpec.torsion_rotor(v_orders=[1, 2, 2]).hash()
    assert rd.ModelSpec.from_canonical(a.canonical()).hash() == a.hash()


def test_hash_stable_across_processes():
    code = ("import riss_drucker as rd; print(rd.ModelSpec.asym_rotor(j_max=7, order=6, axis='IIIr', "
            "reduction='S').hash())")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={"PYTHONHASHSEED": str(seed), **__import__("os").environ}).stdout.strip()
           for seed in (0, 1, 12345)}
    assert len(out) == 1
    assert out.pop() == rd.ModelSpec.asym_rotor(j_max=7, order=6, axis="IIIr", reduction="S").hash()


def test_cache_round_trip_and_invalidation(tmp_path):
    spec = rd.ModelSpec.asym_rotor(j_max=6, order=4)
    m1, info1 = compile_cached(spec, tmp_path)
    m2, info2 = compile_cached(spec, tmp_path)
    assert not info1["hit"] and info2["hit"]
    assert m2.param_names == m1.param_names and m2.spec_json == m1.spec_json
    assert [b["key"] for b in m2.blocks] == [b["key"] for b in m1.blocks]
    th = [9.4, 1.3, 1.13, 2e-6, 1e-4, 4e-3, 3e-7, 3e-5]
    for b in range(m1.num_blocks):
        for k in range(len(th)):
            for x, y in zip(m1.operator(b, k)[:3], m2.operator(b, k)[:3]):
                np.testing.assert_array_equal(x, y)
    s1 = rd.solve(m1, th, nlowest=3)
    s2 = rd.solve(m2, th, nlowest=3)
    np.testing.assert_array_equal(s1.energies(), s2.energies())
    # A structural change gives a new hash, so a new directory: the old cache is never reused.
    spec2 = rd.ModelSpec.asym_rotor(j_max=7, order=4)
    assert cache_path(spec2, tmp_path) != cache_path(spec, tmp_path)
    _, info3 = compile_cached(spec2, tmp_path)
    assert not info3["hit"]


def test_cache_mismatch_is_an_error(tmp_path):
    spec = rd.ModelSpec.asym_rotor(j_max=3, order=2)
    _, info = compile_cached(spec, tmp_path)
    path = cache_path(spec, tmp_path)
    man = json.loads((path / "manifest.json").read_text())
    # Tampered spec hash.
    (path / "manifest.json").write_text(json.dumps({**man, "spec_hash": "0" * 64}))
    with pytest.raises(CacheError, match="spec hash"):
        compile_cached(spec, tmp_path)
    # Operator library version change.
    (path / "manifest.json").write_text(json.dumps({**man, "operator_library_version": "0"}))
    with pytest.raises(CacheError, match="library version"):
        compile_cached(spec, tmp_path)
    # Corrupted block file.
    (path / "manifest.json").write_text(json.dumps(man))
    blk = path / man["blocks"][0]["file"]
    blk.write_bytes(blk.read_bytes()[:-3] + b"xyz")
    with pytest.raises(CacheError, match="checksum"):
        compile_cached(spec, tmp_path)
    # refresh=True recompiles over it.
    _, info = compile_cached(spec, tmp_path, refresh=True)
    assert not info["hit"]
    compile_cached(spec, tmp_path)


def test_multiple_electronic_states_are_reserved():
    spec = rd.ModelSpec("asym_rotor", {"j_min": 0, "j_max": 1, "reduction": "A", "axis": "Ir", "order": 2,
                                       "extra_operators": []}, electronic_states=("X", "A"))
    with pytest.raises(NotImplementedError):
        spec.compile()
    with pytest.raises(ValueError):
        rd.ModelSpec("not_a_model")
