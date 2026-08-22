#!/usr/bin/env python
"""
Build the zip to upload to Google Colab.

    python tools/make_colab_bundle.py

Packs only what the GPU run needs -- source, configs, contracts, datasets --
and leaves out everything Colab would only have to re-create: results,
checkpoints, caches, notebooks, screenshots, git history. That keeps the
upload small enough to drag into a Colab cell without waiting.

Writes zk_kgverify_colab.zip next to this repository.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "zk_kgverify_colab.zip"

# Directories copied wholesale.
INCLUDE_DIRS = ["src", "configs", "contracts", "data", "tools"]

# Individual files at the repository root.
INCLUDE_FILES = [
    "run_all.py",
    "run_step.py",
    "requirements.txt",
    "README.md",
    "README_RUN.md",
]

EXCLUDE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".pt", ".zip"}


def keep(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIX:
        return False
    return True


def main() -> None:
    if OUT.exists():
        OUT.unlink()

    added = 0
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in INCLUDE_FILES:
            p = ROOT / name
            if p.exists():
                z.write(p, p.name)
                added += 1
                total += p.stat().st_size
            else:
                print(f"  [warn] missing {name}")

        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.exists():
                print(f"  [warn] missing directory {d}/")
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and keep(p):
                    z.write(p, str(p.relative_to(ROOT)))
                    added += 1
                    total += p.stat().st_size

    size_mb = OUT.stat().st_size / 1024**2
    print(f"\n  {OUT.name}")
    print(f"  {added} files, {total/1024**2:.1f} MB raw -> {size_mb:.1f} MB zipped")
    print(f"  {OUT}")
    if size_mb > 100:
        print("\n  Over 100 MB -- upload via Google Drive rather than the")
        print("  files.upload() cell, which is unreliable at this size.")
        sys.exit(0)
    print("\n  Upload this in the notebook's 'Upload the code' cell.")


if __name__ == "__main__":
    main()
