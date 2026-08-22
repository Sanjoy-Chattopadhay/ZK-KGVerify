"""
Superseded entry point.

This script generated proofs over *synthetic random vectors* and logged them
with the V1 contract, which stores whatever `verified` flag the caller sends.
Both are now wrong:

  - the records were unrelated to any real model prediction, so they
    demonstrated the plumbing rather than the claim;
  - V1 does not verify anything, so an entry saying a proof checked out was
    only as trustworthy as whoever wrote it.

The replacement generates proofs from genuine predictions of a trained model,
registers the model commitment on-chain first, and submits through V2, which
performs the Schnorr check inside the EVM and reverts on an invalid proof.
"""

import sys

MESSAGE = """
run_sepolia.py is superseded. Use run_all.py.

    python run_all.py --private-key 0x... --stages sepolia,report
        deploy V2, register the model, and submit chain-verified records
        built from real model predictions.

    python run_all.py --private-key 0x... --attach 0xCONTRACT \\
        --stages sepolia,report
        reuse an already-deployed contract.

Read-only auditing needs no key:

    python tools/fetch_onchain_log.py --address 0xCONTRACT
    python tools/check_onchain_parity.py --address 0xCONTRACT --model-id CompGCN

V1 (contracts/ZKKGVerify.sol) is retained as the measurement baseline the
paper compares against, not as a deployment path.
"""

if __name__ == "__main__":
    print(MESSAGE)
    sys.exit(1)
