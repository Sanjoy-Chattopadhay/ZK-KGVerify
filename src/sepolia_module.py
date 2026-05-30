"""
Sepolia Deployment Module for ZK-KGVerify.

Deploys ZKKGVerify.sol to the Ethereum Sepolia testnet, then logs
verification records on-chain and captures *real* gas measurements
that can be reported in the paper alongside (or instead of) the
simulated Python-blockchain numbers.

Environment variables (read at runtime):
    SEPOLIA_RPC_URL   -- e.g. https://sepolia.infura.io/v3/<KEY>
                          or  https://eth-sepolia.g.alchemy.com/v2/<KEY>
                          or  https://rpc.sepolia.org   (public, slower)
    SEPOLIA_PRIVATE_KEY  -- hex private key of the funded account
                            (NEVER commit this; use a throwaway key for testnet)

Typical use
-----------
    from src.sepolia_module import SepoliaBackend
    sep = SepoliaBackend()
    sep.deploy()                          # one-time, returns address
    for proof, ok in zip(proofs, results):
        sep.log_verification(proof, ok)
    print(sep.summary())
    sep.save_log("results/sepolia_log.json")
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


# ============================================================
# Solidity source: kept inline so this module is self-contained,
# but it is byte-identical (modulo whitespace) to contracts/ZKKGVerify.sol.
# ============================================================

_CONTRACT_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ZKKGVerify {
    struct VerificationRecord {
        string modelId;
        uint256 head;
        uint256 relation;
        uint256 tail;
        int256 score;
        bytes32 commitment;
        bool verified;
        bytes32 proofHash;
        uint256 timestamp;
    }
    VerificationRecord[] public records;
    uint256 public recordCount;
    event VerificationLogged(
        uint256 indexed recordId,
        string modelId,
        bool verified,
        uint256 timestamp
    );
    function logVerification(
        string memory _modelId,
        uint256 _head,
        uint256 _relation,
        uint256 _tail,
        int256 _score,
        bytes32 _commitment,
        bool _verified,
        bytes32 _proofHash
    ) public returns (uint256) {
        records.push(VerificationRecord({
            modelId: _modelId,
            head: _head,
            relation: _relation,
            tail: _tail,
            score: _score,
            commitment: _commitment,
            verified: _verified,
            proofHash: _proofHash,
            timestamp: block.timestamp
        }));
        uint256 recordId = recordCount;
        recordCount++;
        emit VerificationLogged(recordId, _modelId, _verified, block.timestamp);
        return recordId;
    }
    function getRecord(uint256 _id) public view returns (VerificationRecord memory) {
        require(_id < recordCount, "Record does not exist");
        return records[_id];
    }
    function getRecordCount() public view returns (uint256) { return recordCount; }
}
"""


@dataclass
class OnChainTx:
    """A single on-chain log transaction's measured properties."""
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


@dataclass
class DeploymentInfo:
    contract_address: str
    deploy_tx: str
    deploy_block: int
    deploy_gas_used: int
    deploy_cost_wei: int


class SepoliaBackend:
    """Real-chain logger. Falls back to RuntimeError if env vars are missing
    so the user is forced to supply credentials rather than silently mocking."""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        chain_id: int = 11155111,  # Sepolia
        solc_version: str = "0.8.20",
        max_priority_fee_gwei: float = 1.5,
    ):
        self.rpc_url = rpc_url or os.environ.get("SEPOLIA_RPC_URL")
        self.private_key = private_key or os.environ.get("SEPOLIA_PRIVATE_KEY")
        if not self.rpc_url:
            raise RuntimeError(
                "SEPOLIA_RPC_URL is not set. Get a free key from "
                "Infura/Alchemy or use https://rpc.sepolia.org and export it."
            )
        if not self.private_key:
            raise RuntimeError(
                "SEPOLIA_PRIVATE_KEY is not set. Export the hex private key "
                "of a Sepolia-funded account (use a throwaway key)."
            )
        self.chain_id = chain_id
        self.solc_version = solc_version
        self.max_priority_fee_gwei = max_priority_fee_gwei

        # Lazy imports so users that only run the local simulation don't pay.
        from web3 import Web3
        from eth_account import Account

        self.Web3 = Web3
        self.Account = Account
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 60}))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to RPC {self.rpc_url}")
        self.account = Account.from_key(self.private_key)
        self.address = self.account.address

        self.abi = None
        self.bytecode = None
        self.contract = None
        self.deployment: Optional[DeploymentInfo] = None
        self.log: List[OnChainTx] = []

        bal_wei = self.w3.eth.get_balance(self.address)
        print(f"[Sepolia] Connected as {self.address}")
        print(f"[Sepolia] Balance: {self.w3.from_wei(bal_wei, 'ether'):.6f} ETH")

    # ---------- compilation ----------
    def _compile(self) -> None:
        import solcx
        try:
            solcx.set_solc_version(self.solc_version)
        except Exception:
            solcx.install_solc(self.solc_version)
            solcx.set_solc_version(self.solc_version)

        compiled = solcx.compile_source(
            _CONTRACT_SOURCE,
            output_values=["abi", "bin"],
            solc_version=self.solc_version,
            optimize=True,
            optimize_runs=200,
        )
        _, iface = compiled.popitem()
        self.abi = iface["abi"]
        self.bytecode = iface["bin"]

    # ---------- deployment ----------
    def deploy(self) -> DeploymentInfo:
        if self.abi is None:
            self._compile()

        Contract = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        nonce = self.w3.eth.get_transaction_count(self.address)
        gas_price = self.w3.eth.gas_price

        tx = Contract.constructor().build_transaction({
            "from": self.address,
            "nonce": nonce,
            "chainId": self.chain_id,
            "gas": 2_500_000,
            "maxFeePerGas": gas_price * 2,
            "maxPriorityFeePerGas": self.w3.to_wei(self.max_priority_fee_gwei, "gwei"),
        })
        signed = self.account.sign_transaction(tx)
        raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        print(f"[Sepolia] Deploy tx submitted: {tx_hash.hex()}  (waiting for receipt...)")

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        cost_wei = receipt.gasUsed * receipt.effectiveGasPrice

        self.contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self.abi)
        self.deployment = DeploymentInfo(
            contract_address=receipt.contractAddress,
            deploy_tx=tx_hash.hex(),
            deploy_block=receipt.blockNumber,
            deploy_gas_used=receipt.gasUsed,
            deploy_cost_wei=cost_wei,
        )
        print(f"[Sepolia] Contract deployed at {receipt.contractAddress}")
        print(f"[Sepolia] Deploy gas: {receipt.gasUsed:,}  "
              f"cost: {self.w3.from_wei(cost_wei, 'ether'):.8f} ETH")
        return self.deployment

    # ---------- attach (skip redeploy on re-runs) ----------
    def attach(self, contract_address: str) -> None:
        if self.abi is None:
            self._compile()
        self.contract = self.w3.eth.contract(address=contract_address, abi=self.abi)
        print(f"[Sepolia] Attached to existing contract at {contract_address}")

    # ---------- per-proof logging ----------
    @staticmethod
    def _commitment_bytes32(commitment_xy) -> bytes:
        x, y = commitment_xy
        return hashlib.sha256(x.to_bytes(32, "big") + y.to_bytes(32, "big")).digest()

    @staticmethod
    def _proof_bytes32(proof) -> bytes:
        # bind the full Schnorr-Fiat-Shamir transcript into a single 32-byte digest
        m = hashlib.sha256()
        m.update(str(proof.commitment_xy).encode())
        m.update(str(proof.announcement_xy).encode())
        m.update(str(proof.challenge).encode())
        m.update(str(proof.response_v).encode())
        m.update(str(proof.response_r).encode())
        m.update(proof.prediction_hash.encode())
        m.update(proof.model_id.encode())
        return m.digest()

    def log_verification(self, proof, verified: bool) -> OnChainTx:
        if self.contract is None:
            raise RuntimeError("Contract not deployed/attached. Call deploy() or attach() first.")

        nonce = self.w3.eth.get_transaction_count(self.address)
        gas_price = self.w3.eth.gas_price

        score_i = int(round(proof.score * 1_000_000))
        score_i = max(min(score_i, 2**255 - 1), -(2**255))  # clamp to int256

        h, r, t = (int(x) for x in proof.triple)

        fn = self.contract.functions.logVerification(
            proof.model_id,
            h, r, t,
            score_i,
            self._commitment_bytes32(proof.commitment_xy),
            bool(verified),
            self._proof_bytes32(proof),
        )

        # Estimate gas, build, sign, send.
        try:
            est_gas = fn.estimate_gas({"from": self.address}) + 20_000
        except Exception:
            est_gas = 250_000

        tx = fn.build_transaction({
            "from": self.address,
            "nonce": nonce,
            "chainId": self.chain_id,
            "gas": est_gas,
            "maxFeePerGas": gas_price * 2,
            "maxPriorityFeePerGas": self.w3.to_wei(self.max_priority_fee_gwei, "gwei"),
        })
        signed = self.account.sign_transaction(tx)
        raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction

        t0 = time.time()
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        latency = time.time() - t0
        cost = receipt.gasUsed * receipt.effectiveGasPrice

        record = OnChainTx(
            record_id=len(self.log),
            tx_hash=tx_hash.hex(),
            block_number=receipt.blockNumber,
            gas_used=receipt.gasUsed,
            effective_gas_price_wei=receipt.effectiveGasPrice,
            cost_wei=cost,
            latency_s=latency,
            verified=bool(verified),
            triple=[h, r, t],
            model_id=proof.model_id,
        )
        self.log.append(record)
        return record

    # ---------- summary stats ----------
    def summary(self) -> Dict[str, Any]:
        if not self.log:
            return {"num_tx": 0}
        gas = [tx.gas_used for tx in self.log]
        cost_wei = [tx.cost_wei for tx in self.log]
        lat = [tx.latency_s for tx in self.log]
        total_cost_wei = sum(cost_wei)

        deploy = asdict(self.deployment) if self.deployment else None
        if deploy:
            deploy["deploy_cost_eth"] = float(self.w3.from_wei(deploy["deploy_cost_wei"], "ether"))

        return {
            "network": "sepolia",
            "chain_id": self.chain_id,
            "contract_address": self.contract.address if self.contract else None,
            "deployment": deploy,
            "num_tx": len(self.log),
            "total_gas_used": int(sum(gas)),
            "avg_gas_per_tx": float(sum(gas) / len(gas)),
            "min_gas_per_tx": int(min(gas)),
            "max_gas_per_tx": int(max(gas)),
            "total_cost_wei": int(total_cost_wei),
            "total_cost_eth": float(self.w3.from_wei(total_cost_wei, "ether")),
            "avg_cost_per_tx_eth": float(self.w3.from_wei(total_cost_wei // max(len(self.log), 1), "ether")),
            "avg_latency_s": float(sum(lat) / len(lat)),
        }

    def save_log(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "transactions": [asdict(tx) for tx in self.log],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Sepolia] Wrote {len(self.log)} tx records to {path}")
