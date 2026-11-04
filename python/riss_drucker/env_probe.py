"""Environment probe (README section 3.1, "Wheel and ABI").

Stdlib only, so it runs *before* the library is built. Records Python version,
compilers, CMake, BLAS/LAPACK libraries, CPU ISA features, core count, RAM,
disk, GPU presence and installed packages, and fails fast if a required piece
is missing.

  python scripts/env_probe.py                 # build-time checks, JSON to stdout
  python scripts/env_probe.py --runtime-only  # installing a prebuilt wheel
  python scripts/env_probe.py --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

MIN_PYTHON = (3, 9)
MIN_CMAKE = (3, 18)
MIN_GCC = (9, 0)
MIN_CLANG = (10, 0)
BASELINE_ISA = ("sse4_2", "popcnt", "ssse3")  # x86-64-v2 subset we rely on


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _version_tuple(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return tuple(int(g) for g in m.groups() if g is not None) if m else None


def probe_tools() -> dict:
    out: dict = {}
    for name, cmd in {
        "cmake": ["cmake", "--version"],
        "ninja": ["ninja", "--version"],
        "g++": ["g++", "-dumpfullversion", "-dumpversion"],
        "clang++": ["clang++", "--version"],
        "objdump": ["objdump", "--version"],
        "git": ["git", "--version"],
    }.items():
        path = shutil.which(cmd[0])
        text = _run(cmd) if path else None
        out[name] = {"path": path, "version": ".".join(map(str, _version_tuple(text) or ())) or None}
    return out


def probe_cpu() -> dict:
    info: dict = {"machine": platform.machine(), "logical_cores_affinity": None, "flags": {}}
    try:
        info["logical_cores_affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["logical_cores_affinity"] = os.cpu_count()
    info["logical_cores_total"] = os.cpu_count()
    try:
        text = Path("/proc/cpuinfo").read_text()
        model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        flags_line = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
        flags = set(flags_line.group(1).split()) if flags_line else set()
        info["model_name"] = model.group(1).strip() if model else None
        for f in (*BASELINE_ISA, "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512vl", "avx512_vnni"):
            info["flags"][f] = f in flags
    except OSError:
        info["model_name"] = None
    return info


def probe_memory() -> dict:
    out: dict = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            out["cgroup_limit"] = int(raw) if raw.isdigit() else raw
            break
        except OSError:
            continue
    return out


def probe_disk() -> dict:
    out = {}
    for p in ("/kaggle/working", os.getcwd(), "/tmp"):
        if os.path.isdir(p):
            u = shutil.disk_usage(p)
            out[p] = {"total": u.total, "free": u.free}
    return out


def probe_gpu() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"present": False}
    text = _run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpus = [g.strip() for g in (text or "").splitlines() if g.strip()]
    return {"present": bool(gpus), "devices": gpus}


def probe_blas() -> dict:
    libs: list[str] = []
    text = _run(["ldconfig", "-p"]) or ""
    for line in text.splitlines():
        name = line.strip().split(" ", 1)[0]
        if re.match(r"lib(openblas|lapack|blas|mkl_rt|mkl_core|blis)\b", name):
            libs.append(name)
    return {
        "shared_libs": sorted(set(libs)),
        "lapack_linkable": any(n == "liblapack.so" or n == "libopenblas.so" for n in libs),
        "mkl_linkable": any(n.startswith("libmkl_rt.so") for n in libs),
    }


def probe_python() -> dict:
    include = sysconfig.get_paths().get("include")
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "include_dir": include,
        "has_python_h": bool(include) and Path(include, "Python.h").is_file(),
        "soabi": sysconfig.get_config_var("SOABI"),
    }


def probe_packages(names=("numpy", "scipy", "setuptools", "wheel", "pip", "pytest",
                          "jsonschema", "sympy", "pandas", "pyarrow", "plotly", "riss-drucker")) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def check(report: dict, runtime_only: bool) -> list[str]:
    problems: list[str] = []
    py = tuple(int(x) for x in report["python"]["version"].split(".")[:2])
    if py < MIN_PYTHON:
        problems.append(f"Python >= {MIN_PYTHON} required, found {report['python']['version']}")
    if report["cpu"]["machine"] not in ("x86_64", "AMD64"):
        problems.append(f"x86-64 CPU required, found {report['cpu']['machine']}")
    missing_isa = [f for f in BASELINE_ISA if not report["cpu"]["flags"].get(f, False)]
    if missing_isa:
        problems.append(f"CPU lacks x86-64-v2 features needed by the baseline build: {missing_isa}")
    if report["packages"]["numpy"] is None:
        problems.append("numpy is required")
    if runtime_only:
        return problems
    tools = report["tools"]
    cm = _version_tuple(tools["cmake"]["version"])
    if not cm or cm < MIN_CMAKE:
        problems.append(f"cmake >= {'.'.join(map(str, MIN_CMAKE))} required, found {tools['cmake']['version']}")
    gcc = _version_tuple(tools["g++"]["version"])
    clang = _version_tuple(tools["clang++"]["version"])
    if not ((gcc and gcc >= MIN_GCC) or (clang and clang >= MIN_CLANG)):
        problems.append("a C++17 compiler is required (g++ >= 9 or clang++ >= 10)")
    if not report["python"]["has_python_h"]:
        problems.append(f"Python.h not found under {report['python']['include_dir']} (python dev headers)")
    for pkg in ("setuptools", "wheel"):
        if report["packages"][pkg] is None:
            problems.append(f"{pkg} is required to build the wheel")
    return problems


def probe(runtime_only: bool = False) -> dict:
    report = {
        "python": probe_python(),
        "platform": platform.platform(),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle"),
        "tools": probe_tools(),
        "cpu": probe_cpu(),
        "memory": probe_memory(),
        "disk": probe_disk(),
        "gpu": probe_gpu(),
        "blas": probe_blas(),
        "packages": probe_packages(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("RISS_DRUCKER_")
                or k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    report["problems"] = check(report, runtime_only)
    report["ok"] = not report["problems"]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="riss_drucker environment probe")
    ap.add_argument("--runtime-only", action="store_true",
                    help="skip build-tool checks (installing a prebuilt wheel)")
    ap.add_argument("--out", help="also write the JSON report to this path")
    args = ap.parse_args(argv)
    report = probe(runtime_only=args.runtime_only)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    if not report["ok"]:
        print("\nenv_probe: FAILED\n  - " + "\n  - ".join(report["problems"]), file=sys.stderr)
        return 1
    print("env_probe: ok", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
