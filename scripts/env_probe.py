#!/usr/bin/env python3
"""Entry point for the environment probe; runs before the library is built.

The implementation lives in python/riss_drucker/env_probe.py (stdlib only) so
the installed package can embed the same probe in its run records.
"""
import runpy
import sys
from pathlib import Path

_impl = Path(__file__).resolve().parent.parent / "python" / "riss_drucker" / "env_probe.py"
sys.argv[0] = str(_impl)
runpy.run_path(str(_impl), run_name="__main__")
