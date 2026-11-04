"""Asymmetric-rotor helpers: J_{Ka,Kc} labels for Wang blocks.

C2 symmetry fixes the (Ka, Kc) parity class of every Wang block: C2 about the
representation z axis acts as (-1)^K, about x as gamma (-1)^J; about the a, b,
c axes the eigenvalues are (-1)^Ka, (-1)^(Ka+Kc), (-1)^Kc. Within a symmetry
class, levels are ordered by tau = Ka - Kc (non-crossing rule). So the n-th
level of a block carries the n-th label of its class. The labels depend only on
J, the block and the representation, not on theta.
"""
from __future__ import annotations

import json

_AXES = {"Ir": ("b", "c", "a"), "IIr": ("c", "a", "b"), "IIIr": ("a", "b", "c")}  # (x, y, z)


def ka_kc_from_tau(J: int, tau: int) -> tuple[int, int]:
    return (J + tau + 1) // 2, J - (J + tau) // 2


def class_labels(J: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """(Ka mod 2, Kc mod 2) -> [(Ka, Kc), ...] in increasing tau."""
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tau in range(-J, J + 1):
        ka, kc = ka_kc_from_tau(J, tau)
        out.setdefault((ka % 2, kc % 2), []).append((ka, kc))
    return out


def block_parity_class(J: int, irrep: str, axis: str) -> tuple[int, int]:
    """(Ka mod 2, Kc mod 2) of a Wang block "E+", "E-", "O+", "O-"."""
    p = 0 if irrep[0] == "E" else 1          # K mod 2
    g = 0 if irrep[1] == "+" else 1          # gamma = (-1)^g
    x_par = (J + g) % 2                      # C2(x) eigenvalue gamma (-1)^J
    x, _, z = _AXES[axis]
    par = {}                                 # parity of the exponent for each principal axis
    par[z], par[x] = p, x_par
    # (-1)^Ka for a, (-1)^Kc for c, (-1)^(Ka+Kc) for b: two of them determine all.
    if "a" in par and "c" in par:
        ka, kc = par["a"], par["c"]
    elif "a" in par:
        ka, kc = par["a"], (par["b"] - par["a"]) % 2
    else:
        kc, ka = par["c"], (par["b"] - par["c"]) % 2
    return ka, kc


def asym_top_labels(model) -> dict[str, list[str]]:
    """Block key -> level labels "J(Ka,Kc)" in energy order within the block."""
    if model.name != "asym_rotor":
        raise ValueError("asym_top_labels needs an asym_rotor model")
    axis = json.loads(model.spec_json)["axis"]
    out = {}
    for b in model.blocks:
        J = b["two_j"] // 2
        labs = class_labels(J).get(block_parity_class(J, b["irrep"], axis), [])
        if len(labs) != b["dim"]:
            raise RuntimeError(f"label count {len(labs)} != block dimension {b['dim']} for {b['key']}")
        out[b["key"]] = [f"{J}({ka},{kc})" for ka, kc in labs]
    return out


def label_index(model) -> dict[str, tuple[str, int]]:
    """Label "J(Ka,Kc)" -> (block key, index within block)."""
    return {lab: (key, i) for key, labs in asym_top_labels(model).items() for i, lab in enumerate(labs)}
