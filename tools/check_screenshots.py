"""Check that every screenshot the README embeds exists and will render.

GitHub renders a broken-image icon for a missing file rather than an error,
so a stale or misnamed screenshot is easy to ship and easy to miss. This
fails loudly instead.

    python tools/check_screenshots.py
"""

import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SHOTS = ROOT / "docs" / "screenshots"

# The V2 deployment. Screenshots of the earlier V1 contract show something
# that verifies nothing and must not ship alongside a README describing this
# one.
#
# There is no reliable automatic check for that: file mtimes looked like a
# usable signal until a git merge rewrote them all and this script cheerfully
# passed three images of the wrong contract. Only your eyes work here --
# open each file and confirm the address in the URL bar reads CONTRACT.
CONTRACT = "0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3"

WANTED = {
    "contract_overview.png": f"https://sepolia.etherscan.io/address/{CONTRACT}",
    "contract_code.png": f"https://sepolia.etherscan.io/address/{CONTRACT}#code",
    "transactions.png": f"https://sepolia.etherscan.io/address/{CONTRACT}",
    "events_tab.png": f"https://sepolia.etherscan.io/address/{CONTRACT}#events",
    "log_verification_tx.png": "any verifyAndLog transaction from the list",
}


def png_size(path):
    """Width and height from the PNG header, without needing Pillow."""
    with path.open("rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


referenced = set(re.findall(r"docs/screenshots/([\w.]+)", README.read_text(encoding="utf-8")))

problems = []
print(f"screenshots for {CONTRACT}\n")

for name in sorted(referenced):
    path = SHOTS / name
    if not path.exists():
        print(f"  MISSING  {name:<26} -> {WANTED.get(name, 'referenced by README')}")
        problems.append(name)
        continue

    size = png_size(path)
    dims = f"{size[0]}x{size[1]}" if size else "not a PNG"
    kb = path.stat().st_size / 1024

    flags = []
    if not size:
        flags.append("NOT A PNG")
    elif size[0] < 900:
        flags.append(f"narrow ({size[0]}px) -- text may be unreadable")

    status = "ok  " if not flags else "WARN"
    print(f"  {status}     {name:<26} {dims:>10}  {kb:6.0f} KB  {'; '.join(flags)}")
    if flags:
        problems.append(name)

orphans = sorted(p.name for p in SHOTS.glob("*.png") if p.name not in referenced)
if orphans:
    print("\n  not referenced by the README (safe to delete if they show the old contract):")
    for name in orphans:
        print(f"    {name}")

print()
if problems:
    print(f"FAIL -- {len(problems)} screenshot(s) need attention")
    sys.exit(1)
print("PASS -- every embedded screenshot exists and looks usable")
print(f"     now open each one and confirm it shows {CONTRACT}")
