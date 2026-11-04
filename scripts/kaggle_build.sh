#!/usr/bin/env bash
# Build and/or install the riss_drucker wheel on Kaggle (or locally), offline.
#
#   scripts/kaggle_build.sh                    # probe, build wheel, install, smoke test
#   scripts/kaggle_build.sh --lapack           # also build the LAPACK/OpenBLAS backend
#   scripts/kaggle_build.sh --mkl              # oneMKL backend (MKL must be installed)
#   scripts/kaggle_build.sh --wheel DIR|FILE   # install a prebuilt wheel (fresh session)
#   scripts/kaggle_build.sh --no-install       # only build the wheel
#
# Environment: RISS_DRUCKER_WORK (default /kaggle/working if present, else
# ./build-kaggle), RISS_DRUCKER_JOBS, CMAKE_ARGS (passed through to CMake).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /kaggle/working ]]; then DEFAULT_WORK=/kaggle/working; else DEFAULT_WORK="$REPO/build-kaggle"; fi
WORK="${RISS_DRUCKER_WORK:-$DEFAULT_WORK}"
PY="${PYTHON:-python3}"
EXTRA_CMAKE=""
INSTALL=1
PREBUILT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lapack) EXTRA_CMAKE+=" -DRISS_DRUCKER_LAPACK=ON" ;;
    --mkl) EXTRA_CMAKE+=" -DRISS_DRUCKER_MKL=ON" ;;
    --openmp) EXTRA_CMAKE+=" -DRISS_DRUCKER_OPENMP=ON" ;;
    --no-simd) EXTRA_CMAKE+=" -DRISS_DRUCKER_SIMD_DISPATCH=OFF" ;;
    --no-install) INSTALL=0 ;;
    --wheel) PREBUILT="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$WORK"
WHEELHOUSE="$WORK/wheelhouse"

if [[ -n "$PREBUILT" ]]; then
  echo "== env_probe (runtime only)"
  "$PY" "$REPO/scripts/env_probe.py" --runtime-only --out "$WORK/env_probe.json" >/dev/null
  if [[ -d "$PREBUILT" ]]; then WHEEL_DIR="$PREBUILT"; else WHEEL_DIR="$(dirname "$PREBUILT")"; fi
  echo "== installing prebuilt wheel from $WHEEL_DIR (offline)"
  "$PY" -m pip install --no-index --no-deps --force-reinstall --find-links "$WHEEL_DIR" riss-drucker
else
  echo "== env_probe (build)"
  "$PY" "$REPO/scripts/env_probe.py" --out "$WORK/env_probe.json" >/dev/null

  # /kaggle/input is read-only and setuptools builds in-tree: copy the source.
  SRC="$REPO"
  if [[ ! -w "$REPO" ]]; then
    SRC="$WORK/src/riss-drucker"
    echo "== source is read-only; copying to $SRC"
    rm -rf "$SRC" && mkdir -p "$(dirname "$SRC")" && cp -r "$REPO" "$SRC"
  fi
  if [[ -z "${RISS_DRUCKER_GIT_COMMIT:-}" ]] && git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    RISS_DRUCKER_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
    [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] && RISS_DRUCKER_GIT_COMMIT+="-dirty"
    export RISS_DRUCKER_GIT_COMMIT
  fi

  echo "== building wheel (CMAKE_ARGS:${CMAKE_ARGS:-}${EXTRA_CMAKE})"
  export CMAKE_ARGS="${CMAKE_ARGS:-}${EXTRA_CMAKE}"
  rm -f "$WHEELHOUSE"/riss_drucker-*.whl
  "$PY" -m pip wheel "$SRC" --no-build-isolation --no-deps -w "$WHEELHOUSE"
  if [[ "$INSTALL" == 1 ]]; then
    "$PY" -m pip install --no-index --no-deps --force-reinstall --find-links "$WHEELHOUSE" riss-drucker
  fi
fi

if [[ "$INSTALL" == 1 || -n "$PREBUILT" ]]; then
  echo "== smoke test"
  (cd "$WORK" && "$PY" - "$WORK" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
import riss_drucker as rd

work = Path(sys.argv[1])
x = np.random.default_rng(0).standard_normal(1001)
ref = rd.dot(x, x, level="scalar")
for lvl in rd.simd_levels_available():
    assert abs(rd.dot(x, x, level=lvl) - ref) <= 1e-12 * ref, lvl
probe = json.loads((work / "env_probe.json").read_text())
rec = rd.write_run_record(work / "run_record_build.json", env_probe=probe)
print(f"riss_drucker {rd.__version__} ({rd.build_info()['git_commit'][:12]}): "
      f"simd={rd.simd_level_active()} of {rd.simd_levels_available()}, "
      f"linalg={rd.linalg_backends()}, run record -> {rec}")
PYEOF
  )
fi
[[ -z "$PREBUILT" ]] && echo "wheel: $(ls "$WHEELHOUSE"/riss_drucker-*.whl)"
exit 0
