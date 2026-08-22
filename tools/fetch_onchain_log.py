"""
Rebuild results/sepolia_log.json directly from the Ethereum Sepolia chain.

This script needs no private key and no local state: it reads the deployed
ZKKGVerify contract, replays every VerificationLogged event, fetches the
receipt for each logging transaction, and writes a complete audit log.

The point is independent verifiability. Anyone -- a reviewer, an editor, a
reader -- can run this and reproduce byte-for-byte the gas and cost table
reported in the paper, without trusting our local measurements.

Usage
-----
    python tools/fetch_onchain_log.py
    python tools/fetch_onchain_log.py --rpc https://sepolia.infura.io/v3/<KEY>
    python tools/fetch_onchain_log.py --address 0x... --deploy-tx 0x...

Latency note
------------
Wall-clock submit-to-receipt latency is a client-side measurement and cannot
be recovered from the chain. What *is* recoverable is inter-block spacing
between consecutive records, which we report as `block_delta_s`. The paper
reports both and is explicit about which is which.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from web3 import Web3

# Deployed audit contract used for the results reported in the paper.
DEFAULT_ADDRESS = "0xdE79F732C74701D168dfdc912a9e42FC8958e56A"
DEFAULT_DEPLOY_TX = (
    "0xef0027065c000a3cde353c99c3c131e8d07a4297e312aebaa0e76db8347791e2"
)

# Public endpoints, tried in order. All are free and need no API key.
FALLBACK_RPCS = [
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://sepolia.drpc.org",
    "https://rpc.sepolia.org",
]

# Minimal ABI: just what this script reads.
ABI = json.loads(
    """
[
 {"anonymous":false,"inputs":[
   {"indexed":true,"internalType":"uint256","name":"recordId","type":"uint256"},
   {"indexed":false,"internalType":"string","name":"modelId","type":"string"},
   {"indexed":false,"internalType":"bool","name":"verified","type":"bool"},
   {"indexed":false,"internalType":"uint256","name":"timestamp","type":"uint256"}],
  "name":"VerificationLogged","type":"event"},
 {"inputs":[],"name":"getRecordCount",
  "outputs":[{"internalType":"uint256","name":"","type":"uint256"}],
  "stateMutability":"view","type":"function"},
 {"inputs":[{"internalType":"uint256","name":"_id","type":"uint256"}],
  "name":"getRecord","outputs":[{"components":[
   {"internalType":"string","name":"modelId","type":"string"},
   {"internalType":"uint256","name":"head","type":"uint256"},
   {"internalType":"uint256","name":"relation","type":"uint256"},
   {"internalType":"uint256","name":"tail","type":"uint256"},
   {"internalType":"int256","name":"score","type":"int256"},
   {"internalType":"bytes32","name":"commitment","type":"bytes32"},
   {"internalType":"bool","name":"verified","type":"bool"},
   {"internalType":"bytes32","name":"proofHash","type":"bytes32"},
   {"internalType":"uint256","name":"timestamp","type":"uint256"}],
   "internalType":"struct ZKKGVerify.VerificationRecord",
   "name":"","type":"tuple"}],
  "stateMutability":"view","type":"function"}
]
"""
)


def connect(rpc: str | None) -> Web3:
    candidates = [rpc] if rpc else FALLBACK_RPCS
    for url in candidates:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 40}))
            if w3.is_connected():
                print(f"[rpc] connected: {url}  (head block {w3.eth.block_number:,})")
                return w3
        except Exception as exc:  # noqa: BLE001 - want to try the next endpoint
            print(f"[rpc] {url} failed: {type(exc).__name__}: {exc}")
    sys.exit("Could not reach any Sepolia RPC endpoint.")


def collect_events(w3: Web3, contract, from_block: int, to_block: int):
    """Page through VerificationLogged logs.

    Public RPCs cap eth_getLogs ranges, so we walk in windows and shrink on
    rejection rather than assuming any particular provider limit.
    """
    events = []
    window = 40_000
    start = from_block
    while start <= to_block:
        end = min(start + window, to_block)
        try:
            batch = contract.events.VerificationLogged().get_logs(
                from_block=start, to_block=end
            )
            events.extend(batch)
            start = end + 1
        except Exception as exc:  # noqa: BLE001 - provider range limits vary
            if window <= 1_000:
                print(f"[logs] giving up on {start}-{end}: {exc}")
                start = end + 1
                continue
            window //= 4
            print(f"[logs] range rejected, shrinking window to {window:,}")
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default=None, help="Sepolia RPC URL (optional).")
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--deploy-tx", default=DEFAULT_DEPLOY_TX)
    ap.add_argument("--out", default="results/sepolia_log.json")
    args = ap.parse_args()

    w3 = connect(args.rpc)
    address = Web3.to_checksum_address(args.address)

    code = w3.eth.get_code(address)
    if len(code) == 0:
        sys.exit(f"No contract code at {address} on this network.")
    print(f"[chain] contract {address}: {len(code)} bytes of code")

    contract = w3.eth.contract(address=address, abi=ABI)
    count = contract.functions.getRecordCount().call()
    print(f"[chain] getRecordCount() = {count}")

    # ---- deployment ----
    deployment = None
    if args.deploy_tx:
        dr = w3.eth.get_transaction_receipt(args.deploy_tx)
        if dr.contractAddress and Web3.to_checksum_address(dr.contractAddress) != address:
            sys.exit(
                f"--deploy-tx created {dr.contractAddress}, not {address}. Refusing to mix."
            )
        deployment = {
            "contract_address": address,
            "deploy_tx": args.deploy_tx,
            "deploy_block": dr.blockNumber,
            "deploy_gas_used": dr.gasUsed,
            "deploy_effective_gas_price_wei": dr.effectiveGasPrice,
            "deploy_cost_wei": dr.gasUsed * dr.effectiveGasPrice,
            "deploy_cost_eth": float(
                w3.from_wei(dr.gasUsed * dr.effectiveGasPrice, "ether")
            ),
        }
        print(
            f"[chain] deployment gas {dr.gasUsed:,} "
            f"cost {deployment['deploy_cost_eth']:.8f} ETH  block {dr.blockNumber:,}"
        )

    # ---- logging transactions ----
    head = w3.eth.block_number
    start_block = deployment["deploy_block"] if deployment else max(0, head - 500_000)
    events = collect_events(w3, contract, start_block, head)
    print(f"[chain] found {len(events)} VerificationLogged events")

    txs = []
    block_ts: dict[int, int] = {}
    for ev in sorted(events, key=lambda e: (e.blockNumber, e.logIndex)):
        rc = w3.eth.get_transaction_receipt(ev.transactionHash)
        if rc.blockNumber not in block_ts:
            block_ts[rc.blockNumber] = w3.eth.get_block(rc.blockNumber).timestamp
        rec = contract.functions.getRecord(ev.args.recordId).call()
        txs.append(
            {
                "record_id": int(ev.args.recordId),
                "tx_hash": ev.transactionHash.hex(),
                "block_number": rc.blockNumber,
                "block_timestamp": block_ts[rc.blockNumber],
                "gas_used": rc.gasUsed,
                "effective_gas_price_wei": rc.effectiveGasPrice,
                "cost_wei": rc.gasUsed * rc.effectiveGasPrice,
                "verified": bool(rec[6]),
                "triple": [int(rec[1]), int(rec[2]), int(rec[3])],
                "score_scaled_1e6": int(rec[4]),
                "commitment_bytes32": "0x" + rec[5].hex(),
                "proof_hash_bytes32": "0x" + rec[7].hex(),
                "model_id": rec[0],
            }
        )
        print(
            f"  [{rec[0]:>8}] id={ev.args.recordId:>3} "
            f"block={rc.blockNumber:,} gas={rc.gasUsed:,}"
        )

    if not txs:
        sys.exit("No logging transactions found; nothing to write.")

    # Inter-record block spacing -- an on-chain-derivable proxy for the
    # client-side latency we measured during the live run.
    deltas = [
        txs[i]["block_timestamp"] - txs[i - 1]["block_timestamp"]
        for i in range(1, len(txs))
        if txs[i]["block_timestamp"] > txs[i - 1]["block_timestamp"]
    ]

    gas = [t["gas_used"] for t in txs]
    cost = [t["cost_wei"] for t in txs]
    summary = {
        "network": "sepolia",
        "chain_id": w3.eth.chain_id,
        "contract_address": address,
        "source": "reconstructed from chain via tools/fetch_onchain_log.py",
        "deployment": deployment,
        "num_tx": len(txs),
        "record_count_onchain": count,
        "total_gas_used": sum(gas),
        "avg_gas_per_tx": statistics.fmean(gas),
        "std_gas_per_tx": statistics.pstdev(gas) if len(gas) > 1 else 0.0,
        "min_gas_per_tx": min(gas),
        "max_gas_per_tx": max(gas),
        "total_cost_wei": sum(cost),
        "total_cost_eth": float(w3.from_wei(sum(cost), "ether")),
        "avg_cost_per_tx_eth": float(w3.from_wei(sum(cost) // len(cost), "ether")),
        "median_inter_record_block_delta_s": (
            statistics.median(deltas) if deltas else None
        ),
        "mean_inter_record_block_delta_s": (
            statistics.fmean(deltas) if deltas else None
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "transactions": txs}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 66)
    print("  ON-CHAIN AUDIT LOG (reconstructed, no local state trusted)")
    print("=" * 66)
    print(f"  Contract        : {address}")
    print(f"  Records on-chain: {count}")
    print(f"  Events replayed : {len(txs)}")
    if deployment:
        print(f"  Deployment gas  : {deployment['deploy_gas_used']:,}")
    print(f"  Total gas       : {summary['total_gas_used']:,}")
    print(
        f"  Gas per record  : {summary['avg_gas_per_tx']:,.0f} "
        f"+/- {summary['std_gas_per_tx']:,.0f}"
    )
    print(f"  Min / Max gas   : {summary['min_gas_per_tx']:,} / {summary['max_gas_per_tx']:,}")
    print(f"  Total cost      : {summary['total_cost_eth']:.8f} ETH (testnet)")
    if summary["median_inter_record_block_delta_s"] is not None:
        print(
            f"  Block spacing   : median {summary['median_inter_record_block_delta_s']:.1f}s"
        )
    print(f"  Written to      : {out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
