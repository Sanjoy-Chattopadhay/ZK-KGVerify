"""
Ethereum Sepolia backend for ZK-KGVerify.

Deploys the audit contract, submits verification records, and captures the
gas / cost / latency measurements reported in the paper.

Two contracts are supported:

  V1  ZKKGVerify.sol    -- `logVerification(...)` stores a `verified` flag the
                           caller asserts. Kept because the published results
                           were produced with it and must stay reproducible.
  V2  ZKKGVerifyV2.sol  -- `verifyAndLog(...)` runs the Schnorr check inside
                           the EVM via the bn128 precompiles and reverts on an
                           invalid proof, so a stored record is an on-chain
                           attestation rather than a claim.

Credentials
-----------
Only a private key is strictly required; a public RPC endpoint is selected
automatically if SEPOLIA_RPC_URL is unset.

    SEPOLIA_PRIVATE_KEY  hex key of a Sepolia-funded account (throwaway only)
    SEPOLIA_RPC_URL      optional; Infura/Alchemy endpoint if you have one

Never point this at an account holding mainnet value. Sepolia ETH is free
from public faucets.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .solidity import compile_contract

# Public Sepolia endpoints, tried in order. None require an API key.
PUBLIC_RPCS = [
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://sepolia.drpc.org",
    "https://rpc.sepolia.org",
    "https://1rpc.io/sepolia",
]

CHAIN_ID = 11155111


@dataclass
class OnChainTx:
    record_id: int
    tx_hash: str
    block_number: int
    gas_used: int
    effective_gas_price_wei: int
    cost_wei: int
    latency_s: float
    verified: bool
    triple: list
    model_id: str
    status: int = 1


@dataclass
class DeploymentInfo:
    contract_address: str
    deploy_tx: str
    deploy_block: int
    deploy_gas_used: int
    deploy_cost_wei: int
    contract_version: str


class SepoliaBackend:
    """Live-chain deployer and logger."""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        version: str = "v2",
        chain_id: int = CHAIN_ID,
        max_priority_fee_gwei: float = 1.5,
    ):
        if version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        self.version = version
        self.chain_id = chain_id
        self.max_priority_fee_gwei = max_priority_fee_gwei

        key = private_key or os.environ.get("SEPOLIA_PRIVATE_KEY")
        if not key:
            raise RuntimeError(
                "No private key. Pass --private-key or set SEPOLIA_PRIVATE_KEY "
                "to a throwaway Sepolia account (fund it at a public faucet)."
            )
        self.private_key = key if key.startswith("0x") else "0x" + key

        from eth_account import Account
        from web3 import Web3

        self.Web3 = Web3
        self.w3 = self._connect(rpc_url or os.environ.get("SEPOLIA_RPC_URL"))
        self.account = Account.from_key(self.private_key)
        self.address = self.account.address

        self.abi, self.bytecode = self._load_contract()
        self.contract = None
        self.deployment: Optional[DeploymentInfo] = None
        self.log: List[OnChainTx] = []
        # Nonce tracked locally: load-balanced RPCs can lag a transaction
        # behind, and refetching mid-batch produces duplicate nonces.
        self._next_nonce: Optional[int] = None

        bal = self.w3.eth.get_balance(self.address)
        self.starting_balance_wei = bal
        print(f"[sepolia] account {self.address}")
        print(f"[sepolia] balance {self.w3.from_wei(bal, 'ether'):.6f} ETH")
        if bal == 0:
            raise RuntimeError(
                f"Account {self.address} has 0 Sepolia ETH. Fund it at "
                "https://sepoliafaucet.com or https://www.alchemy.com/faucets/ethereum-sepolia "
                "and re-run."
            )

    # ---------- connection ----------

    def _connect(self, rpc_url: Optional[str]):
        from web3 import Web3

        candidates = [rpc_url] if rpc_url else PUBLIC_RPCS
        errors = []
        for url in candidates:
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 60}))
                if w3.is_connected() and w3.eth.chain_id == self.chain_id:
                    print(f"[sepolia] rpc {url} (block {w3.eth.block_number:,})")
                    return w3
                errors.append(f"{url}: wrong chain or not connected")
            except Exception as exc:  # noqa: BLE001 - fall through to next RPC
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        raise RuntimeError("No usable Sepolia RPC.\n  " + "\n  ".join(errors))

    def _load_contract(self):
        if self.version == "v2":
            return compile_contract("ZKKGVerifyV2.sol", "ZKKGVerifyV2")
        return compile_contract("ZKKGVerify.sol", "ZKKGVerify")

    # ---------- transaction plumbing ----------

    def _get_nonce(self) -> int:
        if self._next_nonce is None:
            self._next_nonce = self.w3.eth.get_transaction_count(self.address, "pending")
        n = self._next_nonce
        self._next_nonce += 1
        return n

    def _fee_fields(self) -> dict:
        """EIP-1559 fees with headroom for base-fee drift between blocks."""
        base = self.w3.eth.get_block("latest").get("baseFeePerGas", 0)
        tip = self.w3.to_wei(self.max_priority_fee_gwei, "gwei")
        return {"maxFeePerGas": base * 2 + tip, "maxPriorityFeePerGas": tip}

    def _send(self, tx: dict, timeout: int = 300):
        """Sign, send, and await a receipt, resyncing once on a nonce clash."""
        signed = self.account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        t0 = time.time()
        try:
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "nonce" not in msg and "already known" not in msg:
                raise
            print(f"  [resync] {exc}; refetching nonce from 'pending'")
            self._next_nonce = self.w3.eth.get_transaction_count(self.address, "pending")
            tx["nonce"] = self._get_nonce()
            signed = self.account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        return receipt, time.time() - t0

    # ---------- deployment ----------

    def deploy(self) -> DeploymentInfo:
        Contract = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx = Contract.constructor().build_transaction(
            {
                "from": self.address,
                "nonce": self._get_nonce(),
                "chainId": self.chain_id,
                "gas": 2_500_000,
                **self._fee_fields(),
            }
        )
        print(f"[sepolia] deploying {self.version} ...")
        receipt, _ = self._send(tx)
        if receipt.status != 1:
            raise RuntimeError("Deployment transaction reverted.")

        cost = receipt.gasUsed * receipt.effectiveGasPrice
        self.contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self.abi)
        self.deployment = DeploymentInfo(
            contract_address=receipt.contractAddress,
            deploy_tx=receipt.transactionHash.hex(),
            deploy_block=receipt.blockNumber,
            deploy_gas_used=receipt.gasUsed,
            deploy_cost_wei=cost,
            contract_version=self.version,
        )
        print(f"[sepolia] deployed at {receipt.contractAddress}")
        print(
            f"[sepolia] deploy gas {receipt.gasUsed:,}  "
            f"cost {self.w3.from_wei(cost, 'ether'):.8f} ETH"
        )
        print(f"[sepolia] etherscan https://sepolia.etherscan.io/address/{receipt.contractAddress}")
        return self.deployment

    def attach(self, contract_address: str) -> None:
        addr = self.Web3.to_checksum_address(contract_address)
        if len(self.w3.eth.get_code(addr)) == 0:
            raise RuntimeError(f"No contract code at {addr} on Sepolia.")
        self.contract = self.w3.eth.contract(address=addr, abi=self.abi)
        print(f"[sepolia] attached to {addr}")

    # ---------- parity check ----------

    def assert_generator_parity(self) -> None:
        """Fail fast if the deployed contract's H differs from the local one.

        A mismatch would make every locally generated proof fail on-chain for
        a reason that looks like a crypto bug, so it is worth one call.
        """
        if self.version != "v2":
            return
        from .zkp_module import H_X, H_Y

        _, _, hx, hy = self.contract.functions.generators().call()
        if (hx, hy) != (H_X, H_Y):
            raise RuntimeError(
                "Generator mismatch between contract and src/zkp_module.py.\n"
                f"  contract H = ({hx}, {hy})\n"
                f"  python   H = ({H_X}, {H_Y})\n"
                "The deployed contract was built from different constants; redeploy."
            )
        print("[sepolia] generator parity OK (contract H == python H)")

    def assert_digest_parity(self, proof) -> None:
        """Confirm Solidity and Python derive identical digests.

        Both the prediction hash and the Fiat-Shamir challenge are checked.
        A divergence here would surface later as every proof failing on-chain
        for no visible reason, so it is worth two eth_calls before spending
        gas on a batch.
        """
        if self.version != "v2":
            return
        from .zkp_module import hash_challenge

        a = proof.solidity_args()

        ph_on = self.contract.functions.predictionHash(
            a["triple"], a["score"], a["aux"]
        ).call()
        if "0x" + ph_on.hex() != proof.prediction_hash:
            raise RuntimeError(
                f"predictionHash mismatch: contract=0x{ph_on.hex()} "
                f"python={proof.prediction_hash}"
            )

        e_on = self.contract.functions.challenge(
            a["C"], a["A"], bytes.fromhex(proof.prediction_hash[2:]), a["modelId"]
        ).call()
        e_off = hash_challenge(
            proof.commitment_xy, proof.announcement_xy,
            proof.prediction_hash, proof.model_id,
        )
        if e_on != e_off:
            raise RuntimeError(
                f"Challenge mismatch: contract={e_on} python={e_off}. "
                "Fiat-Shamir encodings have diverged."
            )
        print("[sepolia] digest parity OK (predictionHash and challenge match)")

    # Older name kept so existing scripts do not break.
    assert_challenge_parity = assert_digest_parity

    # ---------- record submission ----------

    @staticmethod
    def _commitment_bytes32(commitment_xy) -> bytes:
        x, y = commitment_xy
        return hashlib.sha256(x.to_bytes(32, "big") + y.to_bytes(32, "big")).digest()

    @staticmethod
    def _proof_bytes32(proof) -> bytes:
        return hashlib.sha256(proof.wire_bytes()).digest()

    def register_model(self, model_commitment) -> dict:
        """Publish the model commitment that all later proofs must open.

        Must happen before any record is submitted; the registration block
        is what timestamps the owner's commitment to a specific set of
        weights. Registration is single-shot per modelId on-chain, so a
        repeat run against the same contract reuses the existing entry.
        """
        if self.version != "v2":
            return {}
        if self.contract is None:
            raise RuntimeError("Call deploy() or attach() first.")

        mid = model_commitment.model_id
        if self.contract.functions.isRegistered(mid).call():
            cx, cy, owner, at_block = self.contract.functions.modelCommitment(mid).call()
            if (cx, cy) != tuple(model_commitment.commitment_xy):
                raise RuntimeError(
                    f"Model '{mid}' is already registered on this contract with a "
                    f"different commitment (registered at block {at_block} by {owner}).\n"
                    "Registration is deliberately single-shot -- re-registering would "
                    "let an owner swap models after publishing predictions.\n"
                    "Use a fresh --model-id, or deploy a new contract."
                )
            print(f"[sepolia] model '{mid}' already registered at block {at_block}")
            return {"already_registered": True, "registered_at_block": at_block,
                    "gas_used": 0}

        fn = self.contract.functions.registerModel(
            mid,
            list(model_commitment.commitment_xy),
            bytes.fromhex(model_commitment.weight_digest),
        )
        tx = fn.build_transaction(
            {
                "from": self.address,
                "nonce": self._get_nonce(),
                "chainId": self.chain_id,
                "gas": int(fn.estimate_gas({"from": self.address}) * 1.25),
                **self._fee_fields(),
            }
        )
        receipt, latency = self._send(tx)
        if receipt.status != 1:
            raise RuntimeError("registerModel reverted.")
        print(f"[sepolia] registered model '{mid}'  gas={receipt.gasUsed:,}")
        return {
            "already_registered": False,
            "tx_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed,
            "cost_wei": receipt.gasUsed * receipt.effectiveGasPrice,
            "latency_s": latency,
            "weight_digest": model_commitment.weight_digest,
            "commitment_xy": list(model_commitment.commitment_xy),
        }

    def estimate_verify_gas(self, proof) -> Optional[int]:
        """Gas for verification alone, with no storage write (V2 only)."""
        if self.version != "v2":
            return None
        a = proof.solidity_args()
        return self.contract.functions.verifyProof(
            a["modelId"], a["triple"], a["score"], a["A"], a["sv"], a["sr"], a["aux"]
        ).estimate_gas({"from": self.address})

    def log_verification(self, proof, verified: bool = True) -> OnChainTx:
        if self.contract is None:
            raise RuntimeError("Call deploy() or attach() first.")

        a = proof.solidity_args()
        if self.version == "v2":
            fn = self.contract.functions.verifyAndLog(
                a["modelId"], a["triple"], a["score"],
                a["A"], a["sv"], a["sr"], a["aux"],
            )
        else:
            h, r, t = a["triple"]
            fn = self.contract.functions.logVerification(
                a["modelId"], h, r, t, a["score"],
                self._commitment_bytes32(proof.commitment_xy),
                bool(verified),
                self._proof_bytes32(proof),
            )

        try:
            gas = int(fn.estimate_gas({"from": self.address}) * 1.25)
        except Exception as exc:  # noqa: BLE001
            # For V2 a failing estimate means the chain rejected the proof.
            if self.version == "v2":
                raise RuntimeError(
                    f"On-chain verification rejected this proof: {exc}"
                ) from exc
            gas = 300_000

        tx = fn.build_transaction(
            {
                "from": self.address,
                "nonce": self._get_nonce(),
                "chainId": self.chain_id,
                "gas": gas,
                **self._fee_fields(),
            }
        )
        receipt, latency = self._send(tx, timeout=240)

        record = OnChainTx(
            record_id=len(self.log),
            tx_hash=receipt.transactionHash.hex(),
            block_number=receipt.blockNumber,
            gas_used=receipt.gasUsed,
            effective_gas_price_wei=receipt.effectiveGasPrice,
            cost_wei=receipt.gasUsed * receipt.effectiveGasPrice,
            latency_s=latency,
            verified=bool(verified),
            triple=[int(x) for x in proof.triple],
            model_id=proof.model_id,
            status=receipt.status,
        )
        self.log.append(record)
        return record

    # ---------- reporting ----------

    def summary(self) -> Dict[str, Any]:
        if not self.log:
            return {"num_tx": 0, "contract_version": self.version}

        import statistics as st

        gas = [t.gas_used for t in self.log]
        cost = [t.cost_wei for t in self.log]
        lat = [t.latency_s for t in self.log]
        steady = gas[1:] if len(gas) > 1 else gas

        deploy = asdict(self.deployment) if self.deployment else None
        if deploy:
            deploy["deploy_cost_eth"] = float(
                self.w3.from_wei(deploy["deploy_cost_wei"], "ether")
            )

        return {
            "network": "sepolia",
            "chain_id": self.chain_id,
            "contract_version": self.version,
            "contract_address": self.contract.address if self.contract else None,
            "account": self.address,
            "deployment": deploy,
            "num_tx": len(self.log),
            "total_gas_used": sum(gas),
            "avg_gas_per_tx": st.fmean(gas),
            "first_tx_gas": gas[0],
            "steady_state_avg_gas": st.fmean(steady),
            "steady_state_std_gas": st.pstdev(steady) if len(steady) > 1 else 0.0,
            "min_gas_per_tx": min(gas),
            "max_gas_per_tx": max(gas),
            "total_cost_wei": sum(cost),
            "total_cost_eth": float(self.w3.from_wei(sum(cost), "ether")),
            "avg_cost_per_tx_eth": float(self.w3.from_wei(sum(cost) // len(cost), "ether")),
            "avg_latency_s": st.fmean(lat),
            "median_latency_s": st.median(lat),
            "all_tx_succeeded": all(t.status == 1 for t in self.log),
        }

    def save_log(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "summary": self.summary(),
            "transactions": [asdict(t) for t in self.log],
        }
        if extra:
            payload["context"] = extra
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[sepolia] wrote {len(self.log)} records to {p}")
