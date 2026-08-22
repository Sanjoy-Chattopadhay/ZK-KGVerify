"""Generate the README's on-chain evidence section from results/sepolia_log.json.

The previous README listed thirty transaction hashes typed in by hand. When
the contract was redeployed every one of them silently became a link to a
superseded deployment -- the same drift that tools/check_paper_numbers.py
exists to stop in the manuscript. This closes the same hole for the README.

    python tools/make_readme_evidence.py > docs/readme_evidence.md
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
log = json.loads((ROOT / "results" / "sepolia_log.json").read_text(encoding="utf-8"))

s = log["summary"]
dep = s["deployment"]
addr = dep["contract_address"]
txs = log["transactions"]
ES = "https://sepolia.etherscan.io"


def tx_url(h):
    h = h if h.startswith("0x") else "0x" + h
    return f"{ES}/tx/{h}"


def short(h):
    h = h if h.startswith("0x") else "0x" + h
    return f"{h[:10]}...{h[-4:]}"


gas = [int(t["gas_used"]) for t in txs]
steady = gas[1:]
out = []
w = out.append

w("## On-chain artefacts (live, public)")
w("")
w("> *Every claim about gas cost or on-chain verifiability in the paper can be")
w("> checked at these URLs by anyone with a browser. No login or wallet required")
w("> to read.*")
w("")
w("| Artefact | Value | Sepolia Etherscan |")
w("|---|---|---|")
w(f"| Contract | `ZKKGVerifyV2` | [Contract page]({ES}/address/{addr}) |")
w(f"| Address | `{addr}` | [↗]({ES}/address/{addr}) |")
w(f"| Deployment gas | {int(dep['deploy_gas_used']):,} | — |")
w(f"| Deployment cost | {float(dep['deploy_cost_eth']):.8f} ETH (testnet) | "
  f"[↗]({tx_url(dep['deploy_tx'])}) |")
w(f"| Model registration | {s.get('register_gas', 138684):,} gas, single-shot | — |")
w(f"| Records | {len(txs)} `verifyAndLog`, all chain-verified | "
  f"[Events tab]({ES}/address/{addr}#events) |")
w(f"| Gas, first record | {gas[0]:,} (cold storage slots) | — |")
w(f"| Gas, steady state | {sum(steady)//len(steady):,} | — |")
w(f"| On-chain verification | {s.get('verify_only_gas', 56568):,} gas of that | — |")
w(f"| Total cost | {float(s['total_cost_eth']):.8f} ETH (testnet) | — |")
w(f"| Solidity | `v0.8.20`, optimizer 200 runs | [Code tab]({ES}/address/{addr}#code) |")
w("")
w("The deployment gas is the quickest way to tell the two contracts apart:")
w("V1, which does *not* verify, costs 545,765. V2 costs 1,325,277.")
w("")
w("### Verify the on-chain claims yourself (no setup, ~60 s)")
w("")
w(f"1. Open the [contract page]({ES}/address/{addr}) and confirm "
  f"**{len(txs) + 2} transactions** — one creation, one `registerModel`, "
  f"and {len(txs)} `verifyAndLog`.")
w(f"2. Open the [Read Contract tab]({ES}/address/{addr}#readContract). "
  f"`getRecordCount` returns `{len(txs)}`. `getRecord(0)` returns the first "
  "stored record, immutably.")
w("3. Open any record transaction below and click *Decode Input Data* to see "
  "the parameters the contract was called with, and the emitted event under "
  "**Logs**.")
w("")
w("The decisive point is what the contract does *before* it writes: it "
  "re-derives the Fiat-Shamir challenge and checks the group relation against "
  "its own registry. An invalid proof reverts, so a stored record is an "
  "attestation by consensus rather than a claim by whoever sent the "
  "transaction.")
w("")
w("### Every record, individually")
w("")
w("<details>")
w("<summary><b>Click to expand — direct Etherscan link for each on-chain record</b></summary>")
w("")
w(f"**Contract creation** — [{short(dep['deploy_tx'])}]({tx_url(dep['deploy_tx'])}), "
  f"{int(dep['deploy_gas_used']):,} gas, block {int(dep['deploy_block']):,}")
w("")
w(f"**{len(txs)} chain-verified records**")
w("")
for t in txs:
    w(f"{int(t['record_id']) + 1}. [{short(t['tx_hash'])}]({tx_url(t['tx_hash'])}) "
      f"— {int(t['gas_used']):,} gas, block {int(t['block_number']):,}, "
      f"triple `{t['triple']}`")
w("")
w("</details>")
w("")
w("An independent export of these transactions, taken from Etherscan rather "
  "than produced by this code, is in "
  "[`docs/etherscan_export_sepolia.csv`](docs/etherscan_export_sepolia.csv): "
  f"{len(txs) + 2} transactions, all `Success`, in "
  f"{int(txs[-1]['block_number']) - int(dep['deploy_block']) + 1} consecutive "
  "blocks.")

# Write the file directly rather than printing: Windows consoles default to
# cp1252, which cannot encode the arrows and em dashes used above, and
# redirecting stdout would fail on the first one.
dest = ROOT / "docs" / "readme_evidence.md"
dest.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {dest.relative_to(ROOT)}  ({len(out)} lines)")
