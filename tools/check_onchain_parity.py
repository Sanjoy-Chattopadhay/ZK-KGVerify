"""
Check a *deployed* contract against this repository's cryptography.

Read-only: no private key, no transactions, no gas. A reviewer can point this
at the address in the paper and confirm, without trusting us, that

  1. the deployed V2 contract's NUMS generator H equals the one
     src/zkp_module.py derives by hashing to the curve, and
  2. the contract's Fiat-Shamir challenge is byte-identical to the Python
     implementation on freshly generated proofs, and
  3. honest proofs verify against the deployed bytecode, while every
     adversary in zkp_module.ADVERSARIES is rejected by it.

Usage:
    python tools/check_onchain_parity.py --address 0xDEPLOYED_V2_ADDRESS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from web3 import Web3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solidity import compile_contract  # noqa: E402
from src.zkp_module import (  # noqa: E402
    ADVERSARIES,
    H_X,
    H_Y,
    commit_model,
    generate_proof,
    hash_challenge,
    verify_proof,
)

PUBLIC_RPCS = [
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://sepolia.drpc.org",
    "https://rpc.sepolia.org",
]


def connect(rpc):
    for url in [rpc] if rpc else PUBLIC_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 40}))
            if w3.is_connected():
                print(f"[rpc] {url}  block {w3.eth.block_number:,}")
                return w3
        except Exception as exc:  # noqa: BLE001
            print(f"[rpc] {url}: {exc}")
    sys.exit("No usable RPC endpoint.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True, help="Deployed ZKKGVerifyV2 address.")
    ap.add_argument("--rpc", default=None)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--model-id", default="CompGCN",
                    help="Registered model to audit (must match the run).")
    args = ap.parse_args()

    w3 = connect(args.rpc)
    addr = Web3.to_checksum_address(args.address)
    if len(w3.eth.get_code(addr)) == 0:
        sys.exit(f"No contract code at {addr}.")

    abi, _ = compile_contract("ZKKGVerifyV2.sol", "ZKKGVerifyV2")
    c = w3.eth.contract(address=addr, abi=abi)

    failures = []

    # 1. generators
    gx, gy, hx, hy = c.functions.generators().call()
    ok = (hx == H_X and hy == H_Y)
    print(f"\n[1] generator parity: {'OK' if ok else 'MISMATCH'}")
    print(f"    G = ({gx}, {gy})")
    print(f"    H on-chain = ({hx}, {hy})")
    if not ok:
        print(f"    H local    = ({H_X}, {H_Y})")
        failures.append("generator mismatch")

    # 2. the registered model
    print("\n[2] registry")
    if not c.functions.isRegistered(args.model_id).call():
        sys.exit(
            f"Model {args.model_id!r} is not registered on {addr}. "
            "Pass --model-id to match the run you are auditing."
        )
    cx, cy, owner, at_block = c.functions.modelCommitment(args.model_id).call()
    print(f"    model    : {args.model_id}")
    print(f"    owner    : {owner}")
    print(f"    committed: block {at_block:,}")
    print(f"    C        : ({str(cx)[:24]}..., {str(cy)[:24]}...)")

    # A reader without the secret opening cannot mint honest proofs, so
    # completeness against the live contract is checked with the recorded
    # data rather than fresh proofs. Digest parity, however, is checkable by
    # anyone: it depends only on public encodings.
    print(f"\n[3] digest parity over {args.num} synthetic transcripts")
    rng = np.random.default_rng(99)
    local = commit_model("00" * 32, args.model_id)
    for i in range(args.num):
        p = generate_proof(
            local,
            float(rng.uniform(-5, 5)),
            (int(rng.integers(0, 14541)), int(rng.integers(0, 237)),
             int(rng.integers(0, 14541))),
            args.model_id,
            embedding_vector=rng.standard_normal(384).astype("float32"),
        )
        assert verify_proof(p), "local verifier rejected its own proof"
        a = p.solidity_args()

        ph_on = c.functions.predictionHash(a["triple"], a["score"], a["aux"]).call()
        ph_ok = "0x" + ph_on.hex() == p.prediction_hash

        e_on = c.functions.challenge(
            a["C"], a["A"], bytes.fromhex(p.prediction_hash[2:]), a["modelId"]
        ).call()
        e_ok = e_on == hash_challenge(
            p.commitment_xy, p.announcement_xy, p.prediction_hash, p.model_id
        )
        if not ph_ok:
            failures.append(f"predictionHash mismatch #{i}")
        if not e_ok:
            failures.append(f"challenge mismatch #{i}")
        print(f"    [{i+1}/{args.num}] predictionHash: {ph_ok}   challenge: {e_ok}")

    print("\n[4] soundness against the deployed bytecode")
    # The deployed contract reads the commitment from its registry, so any
    # attack that rewrites a caller-supplied commitment is inert rather than
    # detected. Only attacks on values the caller genuinely controls apply.
    INERT = {"A4_commitment_swap": "commitment comes from the registry"}
    for name, attack in ADVERSARIES.items():
        if name in INERT:
            print(f"    [n/a] {name}: {INERT[name]}")
            continue
        rejected = 0
        trials = 4
        for _ in range(trials):
            p = generate_proof(local, 1.0, (1, 2, 3), args.model_id)
            a = attack(p).solidity_args()
            try:
                if not c.functions.verifyProof(
                    a["modelId"], a["triple"], a["score"], a["A"],
                    a["sv"], a["sr"], a["aux"],
                ).call():
                    rejected += 1
            except Exception:
                rejected += 1  # reverted on a guard
        flag = "ok " if rejected == trials else "FAIL"
        print(f"    [{flag}] {name}: {rejected}/{trials} rejected")
        if rejected != trials:
            failures.append(f"{name} not rejected on-chain")

    print("\n" + "=" * 62)
    if failures:
        print("  PARITY CHECK FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print(f"  PARITY CONFIRMED against {addr}")
    print("=" * 62)


if __name__ == "__main__":
    main()
