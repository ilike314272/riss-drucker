"""Operator cache (README 5.2): compile once per ModelSpec, reuse across fits
and sessions.

Layout: <cache_dir>/operators/<spec hash>/
    manifest.json          spec (canonical), spec hash, library version,
                           parameter names, blocks, per-file SHA-256
    block_00000.npz ...    CSR arrays of every operator in the block

A mismatch (spec hash, library version, file checksum) is an error, never a
warning: delete the directory or fix the spec. Files are written atomically and
the manifest last, so an interrupted write leaves no valid-looking cache.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import _core
from ._version import __version__
from .config import runtime_config
from .io_utils import atomic_write_bytes, atomic_write_json
from .spec import OPERATOR_LIBRARY_VERSION, ModelSpec

MANIFEST = "manifest.json"
CACHE_FORMAT = 1


class CacheError(RuntimeError):
    """The cache on disk does not match the requested spec or library."""


def cache_path(spec: ModelSpec, cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir or runtime_config().cache_dir)
    return root / "operators" / spec.hash()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_compiled(model, spec: ModelSpec, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    blocks, files = [], {}
    for b, info in enumerate(model.blocks):
        arrays = {}
        for k in range(len(model.param_names)):
            ip, ix, dt, shape = model.operator(b, k)
            arrays[f"op{k}_indptr"], arrays[f"op{k}_indices"], arrays[f"op{k}_data"] = ip, ix, dt
            arrays[f"op{k}_shape"] = np.asarray(shape, dtype=np.int64)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        name = f"block_{b:05d}.npz"
        data = buf.getvalue()
        atomic_write_bytes(path / name, data)
        files[name] = _sha256(data)
        blocks.append({k: info[k] for k in ("key", "two_j", "irrep", "spin_species", "electronic_state", "sub",
                                            "dim", "basis_labels")} | {"file": name})
    manifest = {
        "cache_format": CACHE_FORMAT, "spec": spec.canonical(), "spec_hash": spec.hash(),
        "operator_library_version": OPERATOR_LIBRARY_VERSION, "package_version": __version__,
        "git_commit": _core.build_info()["git_commit"], "model_name": model.name,
        "spec_json": model.spec_json, "param_names": list(model.param_names), "blocks": blocks,
        "compile_certificates": list(model.compile_certificates), "files": files,
    }
    atomic_write_json(path / MANIFEST, manifest)
    return path


def load_compiled(spec: ModelSpec, path: Path):
    mpath = path / MANIFEST
    m = json.loads(mpath.read_text())
    if m.get("cache_format") != CACHE_FORMAT:
        raise CacheError(f"{mpath}: cache format {m.get('cache_format')} != {CACHE_FORMAT}")
    if m.get("spec_hash") != spec.hash():
        raise CacheError(f"{mpath}: spec hash {m.get('spec_hash')} does not match requested {spec.hash()}")
    if m.get("spec") != spec.canonical():
        raise CacheError(f"{mpath}: stored spec differs from the requested spec")
    if m.get("operator_library_version") != OPERATOR_LIBRARY_VERSION:
        raise CacheError(f"{mpath}: operator library version {m.get('operator_library_version')} "
                         f"!= {OPERATOR_LIBRARY_VERSION}; delete the cache directory")
    blocks = []
    nparams = len(m["param_names"])
    for b in m["blocks"]:
        data = (path / b["file"]).read_bytes()
        if _sha256(data) != m["files"].get(b["file"]):
            raise CacheError(f"{path / b['file']}: checksum mismatch (corrupt or modified)")
        with np.load(io.BytesIO(data)) as z:
            ops = [(z[f"op{k}_indptr"], z[f"op{k}_indices"], z[f"op{k}_data"], tuple(int(x) for x in z[f"op{k}_shape"]))
                   for k in range(nparams)]
        blocks.append({**b, "ops": ops})
    return _core.CompiledModel.from_parts(m["model_name"], m["spec_json"], m["param_names"], blocks,
                                          m["compile_certificates"])


def compile_cached(spec: ModelSpec, cache_dir: str | Path | None = None, *, refresh: bool = False):
    """Compiled operators for ``spec`` from the cache, compiling on a miss.

    Returns (model, info) with info = {"path", "hit"}.
    """
    path = cache_path(spec, cache_dir)
    if not refresh and (path / MANIFEST).exists():
        return load_compiled(spec, path), {"path": str(path), "hit": True}
    model = spec.compile()
    save_compiled(model, spec, path)
    return model, {"path": str(path), "hit": False}
