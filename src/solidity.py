"""
Single place where Solidity gets compiled.

Both the local EVM tests and the Sepolia deployment path go through
`compile_contract`, so the bytecode exercised in tools/test_onchain_verifier.py
is the same bytecode that reaches the testnet. Keeping one compiler entry
point also keeps the optimizer settings identical, which matters because gas
numbers are reported in the paper.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Tuple

SOLC_VERSION = "0.8.20"
OPTIMIZER_RUNS = 200

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def _ensure_solc() -> None:
    import solcx

    try:
        solcx.set_solc_version(SOLC_VERSION)
    except Exception:
        print(f"[solc] installing solc {SOLC_VERSION} (one-time)...")
        solcx.install_solc(SOLC_VERSION)
        solcx.set_solc_version(SOLC_VERSION)


@functools.lru_cache(maxsize=None)
def compile_contract(filename: str, contract_name: str) -> Tuple[list, str]:
    """Compile `contracts/<filename>` and return (abi, bytecode).

    Results are cached because solc startup dominates the cost and the
    experiment scripts compile the same contract repeatedly.
    """
    import solcx

    _ensure_solc()
    source_path = CONTRACTS_DIR / filename
    if not source_path.exists():
        raise FileNotFoundError(f"No such contract: {source_path}")

    compiled = solcx.compile_source(
        source_path.read_text(encoding="utf-8"),
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
        optimize=True,
        optimize_runs=OPTIMIZER_RUNS,
    )

    # solc keys entries as "<stdin>:<ContractName>".
    for key, iface in compiled.items():
        if key.split(":")[-1] == contract_name:
            return iface["abi"], iface["bin"]

    available = [k.split(":")[-1] for k in compiled]
    raise KeyError(f"{contract_name} not found in {filename}; got {available}")


def contract_source(filename: str) -> str:
    return (CONTRACTS_DIR / filename).read_text(encoding="utf-8")
