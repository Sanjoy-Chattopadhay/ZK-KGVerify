"""
Superseded entry point.

src/pipeline.py drove the original design, in which a fresh Pedersen
commitment was generated per prediction. That construction is vacuous -- the
prover chooses the committed value after seeing the query, so the proof
establishes nothing about the model (see docs/THEORY.md, Proposition 2). The
protocol now registers one commitment before serving and proves knowledge of
*that* opening, which changes the proof API.

Rather than leave this script to fail with a confusing signature error, it
redirects.
"""

import sys

MESSAGE = """
run.py is superseded. Use run_all.py.

    python run_all.py --profile smoke --stages selftest,local-evm
        crypto self-tests and on-chain verification on an in-memory EVM.
        No private key, no network, about two minutes.

    python run_all.py --private-key 0xYOUR_SEPOLIA_TESTNET_KEY
        the full pipeline, including deployment and on-chain records.

    python run_all.py --help
        profiles, stages, and how to split training across machines.

Why: the proof protocol changed. Commitments are now registered once, before
any query is served, and the chain verifies proofs against the registry
instead of storing a boolean the caller asserts. See docs/THEORY.md.
"""

if __name__ == "__main__":
    print(MESSAGE)
    sys.exit(1)
