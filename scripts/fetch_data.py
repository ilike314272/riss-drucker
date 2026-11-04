#!/usr/bin/env python3
"""Fetch spectroscopic source data with provenance (README 6.1).

  python scripts/fetch_data.py --list
  python scripts/fetch_data.py exomol H2S/1H2-32S/AYT2 --out /kaggle/working/data
  python scripts/fetch_data.py url <URL> --out DIR --citation "..." [--doi ...]

Every download is recorded in <out>/manifest.json with URL, SHA-256, size,
UTC time and citation. Check each source's license before redistributing
(e.g. uploading to Kaggle). Needs network access to the host: in a sandbox
whose policy blocks it, the error says which host to allow.
Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

EXOMOL = "https://www.exomol.com/db/{path}/{iso}__{ds}.{ext}"

# Curated starting points (see docs/data_sources.md for citations and notes).
SOURCES = {
    "exomol:H2CO/1H2-12C-16O/AYTY": "Formaldehyde, ExoMol AYTY (MARVEL-refined 2021)",
    "exomol:H2S/1H2-32S/AYT2": "Hydrogen sulfide, ExoMol AYT2",
    "exomol:H2O2/1H2-16O2/APTY": "Hydrogen peroxide, ExoMol APTY (torsion baseline)",
    "exomol:CH4/12C-1H4/MM": "Methane, ExoMol MM (Td polyads, M8)",
}


def _get(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "riss-drucker-fetch/0.1"})
    h = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as fh:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                fh.write(chunk)
                h.update(chunk)
                size += len(chunk)
    except (urllib.error.URLError, OSError) as e:
        host = urllib.request.urlparse(url).hostname
        raise SystemExit(f"download failed for {url}: {e}\n"
                         f"If this runs in a sandbox, allow the host '{host}' in its network policy.")
    dest.with_suffix(dest.suffix + ".part").replace(dest)
    return {"url": url, "file": dest.name, "sha256": h.hexdigest(), "bytes": size,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def _record(out: Path, entry: dict) -> None:
    man = out / "manifest.json"
    data = json.loads(man.read_text()) if man.exists() else {"files": []}
    data["files"] = [f for f in data["files"] if f["file"] != entry["file"]] + [entry]
    man.write_text(json.dumps(data, indent=2) + "\n")


def fetch_exomol(spec: str, out: Path, with_trans: list[str]) -> None:
    mol, iso, ds = spec.split("/")
    base = f"{mol}/{iso}/{ds}"
    for ext in ("def.json", "states.bz2", "pf"):
        e = _get(EXOMOL.format(path=base, iso=iso, ds=ds, ext=ext), out / f"{iso}__{ds}.{ext}")
        e["citation"] = f"ExoMol {ds} ({mol}); cite the line-list paper listed in the .def.json"
        _record(out, e)
        print("fetched", e["file"], e["bytes"], "bytes")
    for name in with_trans:  # e.g. "00000-00100" range chunks named in the def file
        e = _get(EXOMOL.format(path=base, iso=iso, ds=f"{ds}__{name}", ext="trans.bz2"),
                 out / f"{iso}__{ds}__{name}.trans.bz2")
        _record(out, e)
        print("fetched", e["file"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", nargs="?", choices=["exomol", "url"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--out", default="data")
    ap.add_argument("--trans", nargs="*", default=[], help="ExoMol .trans range chunks to fetch")
    ap.add_argument("--citation", default="")
    ap.add_argument("--doi", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.kind:
        for k, v in SOURCES.items():
            print(f"{k:34s} {v}")
        return 0
    out = Path(a.out)
    if a.kind == "exomol":
        fetch_exomol(a.target, out, a.trans)
    else:
        name = a.target.rstrip("/").rsplit("/", 1)[-1] or "download"
        e = _get(a.target, out / name)
        e.update(citation=a.citation, doi=a.doi)
        _record(out, e)
        print("fetched", e["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
