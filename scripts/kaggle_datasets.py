#!/usr/bin/env python3
"""Move data and wheels between this repo's runs and Kaggle datasets.

  python scripts/kaggle_datasets.py download OWNER/DATASET --out DIR
  python scripts/kaggle_datasets.py upload DIR --slug OWNER/DATASET --title "..."   # creates or versions

Credentials are never stored in the repository. The token is read from the
KAGGLE_API_TOKEN environment variable or ~/.kaggle/access_token (chmod 600),
and passed to the official `kaggle` CLI (pip install kaggle), which must be able
to reach www.kaggle.com. Check source licenses before uploading data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _env() -> dict:
    env = dict(os.environ)
    if not env.get("KAGGLE_API_TOKEN"):
        tok = Path.home() / ".kaggle" / "access_token"
        if tok.exists():
            env["KAGGLE_API_TOKEN"] = tok.read_text().strip()
    if not env.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise SystemExit("no Kaggle credentials: set KAGGLE_API_TOKEN or create ~/.kaggle/access_token")
    return env


def _kaggle(args: list[str]) -> None:
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit("the `kaggle` CLI is not installed (pip install kaggle)")
    r = subprocess.run([exe, *args], env=_env())
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed (exit {r.returncode}); "
                         "in a sandbox, www.kaggle.com must be allowed by the network policy")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("dataset")
    d.add_argument("--out", default="data")
    u = sub.add_parser("upload")
    u.add_argument("folder")
    u.add_argument("--slug", required=True)
    u.add_argument("--title", required=True)
    u.add_argument("--message", default="update")
    a = ap.parse_args()
    if a.cmd == "download":
        _kaggle(["datasets", "download", "-d", a.dataset, "-p", a.out, "--unzip"])
    else:
        folder = Path(a.folder)
        meta = folder / "dataset-metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({"title": a.title, "id": a.slug, "licenses": [{"name": "other"}]}, indent=2))
            _kaggle(["datasets", "create", "-p", str(folder), "--dir-mode", "zip"])
        else:
            _kaggle(["datasets", "version", "-p", str(folder), "-m", a.message, "--dir-mode", "zip"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
