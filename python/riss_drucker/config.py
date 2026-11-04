"""Runtime configuration from environment variables (README section 3).

Nothing here is hardcoded to a particular Kaggle machine: thread count and
memory budget are explicit parameters, with conservative defaults derived at
runtime and always recorded in the run record.

Environment variables
---------------------
RISS_DRUCKER_CACHE       cache directory (default /kaggle/working/cache on
                         Kaggle, else ~/.cache/riss_drucker)
RISS_DRUCKER_THREADS     worker threads for block-level parallelism (default 1)
RISS_DRUCKER_MEM_BUDGET  memory budget, e.g. "8G", "512M", "1.5e9" (default:
                         half of the memory available to this process)
RISS_DRUCKER_SIMD        scalar | avx2 | avx512 | auto (read by the C++ core)
RISS_DRUCKER_LINALG      eigen | lapack | mkl (read by the C++ core)
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

KAGGLE_WORKING = Path("/kaggle/working")

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([kKmMgGtT]?)i?[bB]?\s*$")
_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_size(text: str) -> int:
    """Parse '8G', '512M', '1.5e9' into bytes (binary prefixes)."""
    m = _SIZE_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse memory size {text!r} (examples: 8G, 512M, 1.5e9)")
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


def available_memory_bytes() -> int | None:
    """Memory available to this process: min(cgroup limit, MemAvailable)."""
    candidates: list[int] = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < (1 << 60):
            candidates.append(int(raw))
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    return min(candidates) if candidates else None


def default_cache_dir() -> Path:
    if KAGGLE_WORKING.is_dir():
        return KAGGLE_WORKING / "cache"
    return Path.home() / ".cache" / "riss_drucker"


@dataclass(frozen=True)
class RuntimeConfig:
    cache_dir: str
    threads: int
    memory_budget_bytes: int | None
    simd_request: str
    linalg_request: str

    def as_dict(self) -> dict:
        return asdict(self)

    def check_memory(self, nbytes: float, what: str) -> None:
        """Fail early with a clear message instead of an OOM kill (README 3.1)."""
        if self.memory_budget_bytes is not None and nbytes > self.memory_budget_bytes:
            raise MemoryError(
                f"{what} needs ~{nbytes / 1e9:.2f} GB, over the memory budget of "
                f"{self.memory_budget_bytes / 1e9:.2f} GB (RISS_DRUCKER_MEM_BUDGET). "
                "Use an iterative (windowed) solve or raise the budget."
            )


def runtime_config(env: dict | None = None) -> RuntimeConfig:
    env = os.environ if env is None else env
    cache = env.get("RISS_DRUCKER_CACHE") or str(default_cache_dir())

    threads_raw = env.get("RISS_DRUCKER_THREADS", "1")
    try:
        threads = int(threads_raw)
    except ValueError:
        raise ValueError(f"RISS_DRUCKER_THREADS must be a positive integer, got {threads_raw!r}")
    if threads < 1:
        raise ValueError(f"RISS_DRUCKER_THREADS must be >= 1, got {threads}")

    budget_raw = env.get("RISS_DRUCKER_MEM_BUDGET")
    if budget_raw:
        budget: int | None = parse_size(budget_raw)
    else:
        avail = available_memory_bytes()
        budget = avail // 2 if avail else None

    return RuntimeConfig(
        cache_dir=cache,
        threads=threads,
        memory_budget_bytes=budget,
        simd_request=env.get("RISS_DRUCKER_SIMD", "auto") or "auto",
        linalg_request=env.get("RISS_DRUCKER_LINALG", "eigen") or "eigen",
    )


def blas_thread_env(threads_per_block: int = 1) -> dict[str, str]:
    """Environment that pins every BLAS/OpenMP runtime to an explicit thread
    count. Apply it *before* importing numpy/scipy or riss_drucker, so block-
    level parallelism never multiplies with BLAS threads (README 3.1)."""
    n = str(int(threads_per_block))
    return {
        "OMP_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "BLIS_NUM_THREADS": n,
    }
