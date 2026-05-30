"""
Run ZK-KGVerify against the Sepolia Ethereum testnet.

Prerequisites (one-time):
  1. Have a Metamask account with Sepolia ETH (use a public faucet if empty).
  2. Get an RPC URL:  Infura, Alchemy, or the free https://rpc.sepolia.org
  3. Export both as environment variables (PowerShell example):
        $env:SEPOLIA_RPC_URL = "https://sepolia.infura.io/v3/<KEY>"
        $env:SEPOLIA_PRIVATE_KEY = "0xabc...your_throwaway_testnet_key"
     (Use a throwaway key with only Sepolia ETH on it -- never your mainnet key.)
  4. pip install -r requirements.txt   (adds web3, py_ecc, py-solc-x)

Then:
    python run_sepolia.py --num-tx 50

The script will:
  - Generate `num_tx` Schnorr-Fiat-Shamir proofs on BN128 (real EC crypto)
  - Deploy ZKKGVerify.sol once to Sepolia
  - Log every (verified) proof on-chain via logVerification(...)
  - Save measured gas + cost per tx to results/sepolia_log.json
  - Print a summary suitable for the paper's gas-cost table.

For an honest paper, 30-100 transactions is plenty -- they cost real
testnet ETH and Sepolia confirms ~12s per tx, so this takes ~10-30 min.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.zkp_module import (
    generate_proof, verify_proof, batch_generate_proofs, batch_verify_proofs
)
from src.sepolia_module import SepoliaBackend


def synthetic_predictions(n: int, embed_dim: int = 384, seed: int = 42):
    rng = np.random.default_rng(seed)
    vecs = [rng.standard_normal(embed_dim).astype("float32") for _ in range(n)]
    scores = [float(rng.uniform(-3, 3)) for _ in range(n)]
    triples = [
        (int(rng.integers(0, 14541)), int(rng.integers(0, 237)), int(rng.integers(0, 14541)))
        for _ in range(n)
    ]
    return vecs, scores, triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tx", type=int, default=30,
                    help="Number of on-chain verification records to log (default 30).")
    ap.add_argument("--model-id", type=str, default="CompGCN")
    ap.add_argument("--attach", type=str, default=None,
                    help="Skip deployment; attach to an existing contract address.")
    ap.add_argument("--out", type=str, default="results/sepolia_log.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print(f"  ZK-KGVerify -- Sepolia run ({args.num_tx} tx)")
    print("=" * 70)

    # ----- generate proofs locally (no on-chain) -----
    vecs, scores, triples = synthetic_predictions(args.num_tx, seed=args.seed)
    t0 = time.time()
    proofs, gen_stats = batch_generate_proofs(vecs, scores, triples, args.model_id)
    print(f"[local] Generated {len(proofs)} proofs in {time.time()-t0:.2f}s "
          f"(avg {gen_stats['avg_gen_time']*1000:.2f}ms / proof)")

    results, ver_stats = batch_verify_proofs(proofs)
    print(f"[local] Verified locally: {ver_stats['num_valid']}/{ver_stats['num_verified']} "
          f"(avg {ver_stats['avg_verify_time']*1000:.2f}ms / verify)")
    if ver_stats["num_valid"] != ver_stats["num_verified"]:
        sys.exit("Local verification failed -- aborting before touching Sepolia.")

    # ----- connect to Sepolia -----
    sep = SepoliaBackend()

    if args.attach:
        sep.attach(args.attach)
    else:
        sep.deploy()

    # ----- log every proof on-chain -----
    for i, (proof, ok) in enumerate(zip(proofs, results)):
        tx = sep.log_verification(proof, ok)
        print(f"  [{i+1:>3}/{len(proofs)}] tx={tx.tx_hash[:14]}... "
              f"block={tx.block_number}  gas={tx.gas_used:,}  "
              f"latency={tx.latency_s:.1f}s")

    # ----- save + summarise -----
    sep.save_log(args.out)
    summary = sep.summary()
    print()
    print("=" * 70)
    print("  SEPOLIA RESULTS (paste into paper)")
    print("=" * 70)
    print(f"  Network          : sepolia (chainId {summary['chain_id']})")
    print(f"  Contract         : {summary['contract_address']}")
    if summary.get("deployment"):
        d = summary["deployment"]
        print(f"  Deployment gas   : {d['deploy_gas_used']:,}")
        print(f"  Deployment cost  : {d['deploy_cost_eth']:.8f} ETH (testnet)")
    print(f"  Transactions     : {summary['num_tx']}")
    print(f"  Total gas        : {summary['total_gas_used']:,}")
    print(f"  Avg gas / tx     : {summary['avg_gas_per_tx']:,.0f}")
    print(f"  Min / Max gas    : {summary['min_gas_per_tx']:,} / {summary['max_gas_per_tx']:,}")
    print(f"  Avg latency / tx : {summary['avg_latency_s']:.2f} s")
    print(f"  Total cost       : {summary['total_cost_eth']:.8f} ETH (testnet)")
    print(f"  Full log         : {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
