"""Wheel build: CMake builds the C++ core and the pybind11 module.

Configuration passes through environment variables:
  CMAKE_ARGS            extra CMake args, e.g. "-DRISS_DRUCKER_LAPACK=ON"
  RISS_DRUCKER_JOBS     parallel build jobs (default: CPU count)
  CMAKE_BUILD_TYPE      default Release
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text().strip()

_pyver = re.search(r'__version__ = "([^"]+)"', (ROOT / "python/riss_drucker/_version.py").read_text())
if not _pyver or _pyver.group(1) != VERSION:
    sys.exit(f"python/riss_drucker/_version.py does not match VERSION ({VERSION})")


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        out_dir = ext_path.parent
        build_temp = Path(self.build_temp).resolve() / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DRD_PY_OUTPUT_DIR={out_dir}",
            "-DRISS_DRUCKER_PY=ON",
            "-DRISS_DRUCKER_BUILD_TESTS=OFF",
            "-DRISS_DRUCKER_BUILD_BENCH=OFF",
        ]
        if shutil.which("ninja"):
            args += ["-G", "Ninja"]
        args += shlex.split(os.environ.get("CMAKE_ARGS", ""))
        jobs = os.environ.get("RISS_DRUCKER_JOBS", str(os.cpu_count() or 1))

        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build_temp), *args], check=True)
        subprocess.run(["cmake", "--build", str(build_temp), "--target", "_core", "-j", jobs],
                       check=True)


setup(
    version=VERSION,
    packages=["riss_drucker"],
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("riss_drucker._core")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
