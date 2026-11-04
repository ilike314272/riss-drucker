"""M0 smoke tests for the installed wheel: import, provenance, SIMD/scalar
agreement, linalg backends, configuration and run records."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import riss_drucker as rd


def test_version_and_build_info():
    info = rd.build_info()
    assert info["version"] == rd.__version__
    assert "-march=x86-64-v2" in info["cxx_flags"]
    for bad in ("march=native", "ffast-math", "Ofast"):
        assert bad not in info["cxx_flags"]
    assert info["linalg_backends"][0] == "eigen"
    assert info["git_commit"]


def test_simd_levels_consistent_with_cpu():
    levels = rd.simd_levels_available()
    feats = rd.cpu_features()
    assert levels[0] == "scalar"
    assert rd.simd_level_active() in levels
    assert feats["x86_64_v2"]
    if "avx512" in levels:
        assert feats["avx512f"] and feats["avx512dq"]
    if "avx2" in levels:
        assert feats["avx2"] and feats["fma"]


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 33, 4097])
def test_dot_axpy_agree_across_levels(n):
    rng = np.random.default_rng(n)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    ref = rd.dot(x, y, level="scalar")
    tol = 4 * (n + 1) * np.finfo(float).eps * (np.abs(x * y).sum() + 1)
    assert abs(ref - float(x @ y)) <= tol
    for lvl in rd.simd_levels_available():
        assert abs(rd.dot(x, y, level=lvl) - ref) <= tol
        np.testing.assert_allclose(rd.axpy(0.25, x, y, level=lvl), y + 0.25 * x, rtol=0, atol=1e-15 * (1 + np.abs(y).max(initial=0)))


def _random_csr(rng, nrows, ncols, density):
    dense = rng.standard_normal((nrows, ncols)) * (rng.random((nrows, ncols)) < density)
    indptr, indices, data = [0], [], []
    for r in range(nrows):
        cols = np.nonzero(dense[r])[0]
        indices.extend(cols)
        data.extend(dense[r, cols])
        indptr.append(len(indices))
    return dense, np.array(indptr, np.int64), np.array(indices, np.int32), np.array(data)


def test_csr_matvec_matches_dense_on_all_levels():
    rng = np.random.default_rng(0)
    dense, indptr, indices, data = _random_csr(rng, 70, 45, 0.3)
    x = rng.standard_normal(45)
    for lvl in rd.simd_levels_available():
        np.testing.assert_allclose(rd.csr_matvec(indptr, indices, data, x, level=lvl), dense @ x, rtol=1e-13, atol=1e-13)


def test_csr_matvec_validates_input():
    with pytest.raises(ValueError):
        rd.csr_matvec(np.array([0, 1]), np.array([5], np.int32), np.array([1.0]), np.zeros(3))
    with pytest.raises(ValueError):
        rd.csr_matvec(np.array([0, 2]), np.array([0], np.int32), np.array([1.0]), np.zeros(3))


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        rd.dot(np.ones(3), np.ones(3), level="sse9")


@pytest.mark.parametrize("backend", rd.linalg_backends())
def test_eigh_backends(backend):
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    lam = np.linspace(-2, 5, 30)
    a = (q * lam) @ q.T
    w, v = rd.eigh(a, backend=backend)
    np.testing.assert_allclose(w, lam, atol=1e-12)
    np.testing.assert_allclose(a @ v, v * w, atol=1e-11)
    w2, v2 = rd.eigh(a, backend=backend, vectors=False)
    assert v2 is None
    np.testing.assert_allclose(w2, w, atol=1e-12)


def test_simd_env_override_in_subprocess():
    code = "import riss_drucker as rd; print(rd.simd_level_active())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "RISS_DRUCKER_SIMD": "scalar"}).stdout.strip()
    assert out == "scalar"


def test_runtime_config(tmp_path):
    cfg = rd.runtime_config({"RISS_DRUCKER_CACHE": str(tmp_path), "RISS_DRUCKER_THREADS": "3",
                             "RISS_DRUCKER_MEM_BUDGET": "2G"})
    assert cfg.cache_dir == str(tmp_path) and cfg.threads == 3
    assert cfg.memory_budget_bytes == 2 * 1024**3
    cfg.check_memory(1e9, "small block")
    with pytest.raises(MemoryError, match="RISS_DRUCKER_MEM_BUDGET"):
        cfg.check_memory(rd.dense_eigh_bytes(20000), "dense block N=20000")
    with pytest.raises(ValueError):
        rd.runtime_config({"RISS_DRUCKER_THREADS": "0"})


def test_run_record_roundtrip(tmp_path):
    path = rd.write_run_record(tmp_path / "run.json", spec_hash="abc", seeds={"numpy": 7})
    rec = json.loads(path.read_text())
    assert rec["spec_hash"] == "abc" and rec["seeds"] == {"numpy": 7}
    assert rec["build"]["git_commit"] == rd.build_info()["git_commit"]
    assert rec["runtime_config"]["threads"] >= 1
    assert not list(tmp_path.glob(".*.tmp"))  # atomic write left no temp files
