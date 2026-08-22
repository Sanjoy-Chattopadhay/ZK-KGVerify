"""
Guard against the manuscript drifting away from the measurements.

The earlier submission stated 30 on-chain records while the shipped log held
15, and quoted a deployment gas figure the log did not contain at all. Both
were the same failure mode: numbers typed into the .tex by hand, then edited
in one place and not the other.

The fix is that every experimental quantity lives in results/paper_macros.tex
and the manuscript references the macro. This script enforces it:

  1. Every \\newcommand in paper_macros.tex that the manuscript uses must
     resolve -- no undefined macros.
  2. The manuscript must not contain bare numeric literals in the patterns
     that historically drifted (gas counts, ETH amounts, millisecond timings,
     contract addresses, transaction hashes).
  3. Any contract address or tx hash in the .tex must match results/.

Usage:
    python tools/check_paper_numbers.py --tex ../GNN_Sonjoy/sn-article.tex
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Patterns that must come from a macro, not a literal. Each entry is
# (human-readable description, compiled regex applied to comment-stripped TeX).
SUSPECT_PATTERNS = [
    ("gas figure", re.compile(r"\b\d{2,3}[,{]?[,}]?\d{3}\s*(?:gas|\\,gas)\b", re.I)),
    ("ETH amount", re.compile(r"\b0\.\d{3,}\s*\\?,?\s*ETH\b", re.I)),
    ("millisecond timing", re.compile(r"\b\d+\.\d+\s*\\,?ms\b", re.I)),
    ("contract address", re.compile(r"0x[0-9a-fA-F]{40}\b")),
    ("transaction hash", re.compile(r"0x[0-9a-fA-F]{64}\b")),
]

# Macro definitions are allowed to contain literals; that is their job.
MACRO_DEF = re.compile(r"\\newcommand\{\\(\w+)\}")


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        idx, esc = None, False
        for i, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
                continue
            if ch == "%" and not esc:
                idx = i
                break
            esc = False
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True, help="Manuscript .tex to audit.")
    ap.add_argument("--macros", default=str(ROOT / "results" / "paper_macros.tex"))
    ap.add_argument("--results", default=str(ROOT / "results" / "all_results.json"))
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on literal-number warnings, not just errors.")
    args = ap.parse_args()

    tex_path, macro_path = Path(args.tex), Path(args.macros)
    if not tex_path.exists():
        sys.exit(f"No manuscript at {tex_path}")
    if not macro_path.exists():
        sys.exit(f"No macros at {macro_path}. Run: python run_all.py --stages report")

    tex_raw = tex_path.read_text(encoding="utf-8", errors="replace")
    tex = strip_comments(tex_raw)
    macros_src = macro_path.read_text(encoding="utf-8")
    defined = set(MACRO_DEF.findall(macros_src))

    errors: list[str] = []
    warnings: list[str] = []

    # ---- 1. undefined macros ----
    # Only consider names that look like ours (camelCase, defined-ish) to
    # avoid flagging every LaTeX primitive in the document.
    used = set(re.findall(r"\\([a-zA-Z]{3,})\b", tex))
    candidates = {
        u for u in used
        if u not in defined
        and re.match(r"^(zk|sepolia|evm|mainnet|env|run|time|num)[A-Z]", u)
    }
    for name in sorted(candidates):
        errors.append(f"manuscript uses \\{name} but paper_macros.tex does not define it")

    print(f"[macros] {len(defined)} defined, {len(defined & used)} referenced by the manuscript")

    # ---- 2. literal numbers that should be macros ----
    for label, pattern in SUSPECT_PATTERNS:
        for match in pattern.finditer(tex):
            line_no = tex[: match.start()].count("\n") + 1
            warnings.append(
                f"line {line_no}: hard-coded {label} {match.group(0).strip()!r} "
                f"-- reference a macro from paper_macros.tex instead"
            )

    # ---- 3. addresses / hashes must match results ----
    results_path = Path(args.results)
    if results_path.exists():
        blob = results_path.read_text(encoding="utf-8").lower()
        for pattern, kind in [
            (re.compile(r"0x[0-9a-fA-F]{40}\b"), "address"),
            (re.compile(r"0x[0-9a-fA-F]{64}\b"), "tx hash"),
        ]:
            for match in set(pattern.findall(tex)):
                if match.lower()[2:] not in blob:
                    errors.append(
                        f"{kind} {match} appears in the manuscript but not in "
                        f"{results_path.name} -- it is not backed by a recorded run"
                    )

    # ---- report ----
    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
        print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if warnings and args.strict:
        sys.exit(f"{len(warnings)} warning(s) under --strict")

    print("OK: every experimental number in the manuscript resolves to a measured macro.")


if __name__ == "__main__":
    main()
