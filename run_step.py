#!/usr/bin/env python
"""
Logged wrapper around run_all.py -- keeps a permanent record of every step.

    python run_step.py --label 01-selftest --stages selftest,local-evm

Runs run_all.py with whatever arguments follow, while:

  * teeing the console output to results/logs/<timestamp>__<label>.log
  * snapshotting every results/ artefact afterwards into
    results/runs/<timestamp>__<label>/
  * writing a manifest (arguments, exit code, duration, environment)

The snapshots are what make the run auditable months later, when a reviewer
asks where a number came from and results/ has been overwritten five times.
Nothing is ever deleted -- each invocation adds a new directory.

The private key is never logged. If --private-key is passed it is replaced
with "***" everywhere in the manifest; prefer the SEPOLIA_PRIVATE_KEY
environment variable so it never reaches the command line at all.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
RUNS = RESULTS / "runs"

SNAPSHOT_GLOBS = ("*.json", "*.tex", "*.png", "*.pdf", "*.csv")


def redact(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            out.append("***")
            skip = False
            continue
        if a == "--private-key":
            out.append(a)
            skip = True
        elif a.startswith("--private-key="):
            out.append("--private-key=***")
        else:
            out.append(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a pipeline step with full logging and snapshotting.",
        epilog="Any further arguments are passed straight through to run_all.py.",
    )
    ap.add_argument("--label", required=True,
                    help="Short name for this step, e.g. 01-selftest")
    args, passthrough = ap.parse_known_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}__{args.label}"
    LOGS.mkdir(parents=True, exist_ok=True)
    snapshot_dir = RUNS / tag
    log_path = LOGS / f"{tag}.log"

    cmd = [sys.executable, "-u", str(ROOT / "run_all.py"), *passthrough]
    safe_cmd = redact(cmd)

    header = (
        f"{'=' * 72}\n"
        f"  ZK-KGVerify step: {args.label}\n"
        f"  started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  command : {' '.join(safe_cmd[1:])}\n"
        f"  host    : {platform.node()}  {platform.platform()}\n"
        f"  log     : {log_path.relative_to(ROOT)}\n"
        f"{'=' * 72}\n"
    )
    print(header, end="")

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert proc.stdout is not None
        for chunk in proc.stdout:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
            # Flush per line, not per 8 KB block. A long training stage emits
            # only a few hundred bytes an hour, so the default buffering left
            # the log empty for the whole run and there was no way to tell a
            # working run from a wedged one -- which is the failure mode this
            # wrapper exists to rule out.
            log.flush()
        rc = proc.wait()
        elapsed = time.time() - t0
        log.write(f"\n{'=' * 72}\n  exit code {rc} after {elapsed:.1f}s\n")

    # Snapshot whatever the step produced, so it survives the next run.
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in SNAPSHOT_GLOBS:
        for src in sorted(RESULTS.glob(pattern)):
            shutil.copy2(src, snapshot_dir / src.name)
            copied.append(src.name)

    manifest = {
        "label": args.label,
        "timestamp": stamp,
        "command": safe_cmd[1:],
        "exit_code": rc,
        "duration_s": round(elapsed, 1),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "log": str(log_path.relative_to(ROOT)),
        "artefacts": copied,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n  log      -> {log_path.relative_to(ROOT)}")
    print(f"  snapshot -> {snapshot_dir.relative_to(ROOT)}  ({len(copied)} files)")
    if rc != 0:
        print(f"\n  STEP FAILED (exit {rc}). Send me the log above before continuing.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
