#!/usr/bin/env python3
"""Verify the portable-baseline rule (README sections 3 and 4).

Disassembles a static library or object files and fails if any object other
than the dispatched SIMD kernels contains AVX/AVX2/AVX-512 instructions
(VEX/EVEX-encoded mnemonics, ymm/zmm registers or opmask registers).
Those would crash on a Kaggle CPU without the ISA, so they may only appear in
translation units that are entered after runtime CPU detection.

Usage: check_portable_isa.py [--objdump PATH] FILE [FILE ...]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

ALLOWED_OBJECTS = ("kernels_avx2", "kernels_avx512")
# Non-SIMD instructions whose mnemonics also start with "v".
NON_SIMD_V = {"verr", "verw", "vmcall", "vmlaunch", "vmresume", "vmxoff", "vmxon",
              "vmread", "vmwrite", "vmptrld", "vmptrst", "vmclear", "vmfunc"}
REG_RE = re.compile(r"%(ymm|zmm)\d+|%k[0-7]\b")


def scan(objdump: str, path: str) -> dict[str, list[str]]:
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", path],
                         check=True, capture_output=True, text=True).stdout
    current = path
    hits: dict[str, list[str]] = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+\.o(?:bj)?):\s+file format", line)
        if m:
            current = m.group(1)
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not re.match(r"^\s*[0-9a-f]+:$", parts[0]):
            continue
        insn = parts[1].strip()
        mnemonic = insn.split(" ", 1)[0]
        if (mnemonic.startswith("v") and mnemonic not in NON_SIMD_V) or REG_RE.search(insn):
            hits.setdefault(current, []).append(insn)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objdump", default="objdump")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    bad = 0
    for f in args.files:
        for obj, insns in sorted(scan(args.objdump, f).items()):
            if any(a in obj for a in ALLOWED_OBJECTS):
                print(f"ok   {obj}: {len(insns)} SIMD instructions (dispatched TU)")
                continue
            bad += 1
            print(f"FAIL {obj}: {len(insns)} AVX-class instructions outside dispatched kernels, e.g.")
            for i in insns[:5]:
                print(f"       {i}")
    if bad:
        print(f"{bad} object(s) violate the portable baseline (-march=x86-64-v2)")
        return 1
    print("portable baseline verified: AVX-class code only in dispatched kernel TUs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
