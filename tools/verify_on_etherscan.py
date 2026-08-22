"""Submit the deployed contract to Etherscan for source verification.

The web form has five fields that must all be right, and gets no second
chance at telling you which one was wrong. This sends the same request over
the API with every field filled from the same constants the deployment used,
so the only thing you supply is an API key.

    python tools/verify_on_etherscan.py --api-key YOUR_KEY

Get a free key at https://etherscan.io/myapikey (any Etherscan account works;
the key is shared across all their chains).

Nothing here can touch your funds: verification is a read-only claim about
source code and needs no private key.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solidity import OPTIMIZER_RUNS, SOLC_VERSION  # noqa: E402

# Etherscan wants the full compiler build, not just the semver.
SOLC_BUILD = {"0.8.20": "v0.8.20+commit.a1b79de6"}[SOLC_VERSION]
CHAIN_ID = 11155111  # Sepolia
API = "https://api.etherscan.io/v2/api"
LICENSE_MIT = 3


def post(payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(f"{API}?chainid={CHAIN_ID}", data=data)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def get(params: dict) -> dict:
    qs = urllib.parse.urlencode({**params, "chainid": CHAIN_ID})
    with urllib.request.urlopen(f"{API}?{qs}", timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="from https://etherscan.io/myapikey")
    ap.add_argument("--address", default=None,
                    help="defaults to the contract in results/sepolia_log.json")
    ap.add_argument("--contract", default="ZKKGVerifyV2")
    args = ap.parse_args()

    address = args.address
    if not address:
        log = json.loads((ROOT / "results" / "sepolia_log.json").read_text(encoding="utf-8"))
        address = log["summary"]["contract_address"]

    source = (ROOT / "contracts" / f"{args.contract}.sol").read_text(encoding="utf-8")

    # The failure mode worth guarding: pasting compiled bytecode instead of
    # source. It is what the web form silently accepts and then rejects with
    # a parser error pointing at column 1.
    if source.lstrip().startswith("0x"):
        sys.exit("That file looks like bytecode, not Solidity.")

    print(f"  contract  : {args.contract}")
    print(f"  address   : {address}")
    print(f"  compiler  : {SOLC_BUILD}")
    print(f"  optimizer : enabled, {OPTIMIZER_RUNS} runs")
    print(f"  source    : {len(source):,} chars\n")

    result = post({
        "apikey": args.api_key,
        "module": "contract",
        "action": "verifysourcecode",
        "contractaddress": address,
        "sourceCode": source,
        "codeformat": "solidity-single-file",
        "contractname": args.contract,
        "compilerversion": SOLC_BUILD,
        "optimizationUsed": 1,
        "runs": OPTIMIZER_RUNS,
        "constructorArguements": "",   # Etherscan's spelling, not ours
        "licenseType": LICENSE_MIT,
    })

    if result.get("status") != "1":
        msg = result.get("result", result)
        if "already verified" in str(msg).lower():
            print(f"Already verified.\n  https://sepolia.etherscan.io/address/{address}#code")
            return
        sys.exit(f"Submission refused: {msg}")

    guid = result["result"]
    print(f"  submitted, guid {guid}\n  waiting for the compiler ...")

    for _ in range(30):
        time.sleep(5)
        check = get({"apikey": args.api_key, "module": "contract",
                     "action": "checkverifystatus", "guid": guid})
        status = str(check.get("result", ""))
        if "Pending" in status:
            continue
        if check.get("status") == "1" or "Verified" in status:
            print(f"\n  VERIFIED\n  https://sepolia.etherscan.io/address/{address}#code")
            print("\n  The transaction list will now show verifyAndLog instead of")
            print("  the raw selector 0xb43e985b.")
            return
        sys.exit(f"\n  Verification failed: {status}")

    sys.exit("  Timed out waiting for Etherscan. Re-check the address page in a minute.")


if __name__ == "__main__":
    main()
