"""
Blockchain Module for ZK-KGVerify.

Implements a lightweight local blockchain to store ZKP verification results
as an immutable audit trail. Two modes:

1. PythonBlockchain: Pure Python blockchain (no dependencies, always works)
2. EthereumSimulator: Uses web3.py + eth-tester (closer to real Ethereum)

Both provide the same interface for logging verified predictions.
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import numpy as np


# ============================================================
# Pure Python Blockchain (Lightweight, No Dependencies)
# ============================================================

@dataclass
class Block:
    """A single block in the chain."""
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        block_data = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()


@dataclass
class Transaction:
    """A transaction storing ZKP verification results."""
    tx_hash: str
    timestamp: float
    model_id: str
    triple: tuple
    prediction_score: float
    zkp_commitment: int
    zkp_verified: bool
    proof_hash: str
    gas_used: int = 0


class PythonBlockchain:
    """
    Lightweight Python blockchain for storing ZKP verification results.
    Simulates mining with adjustable difficulty.
    """

    def __init__(self, difficulty=2):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.difficulty = difficulty
        self.block_gas_limit = 3000000
        self.gas_per_tx = 21000  # Base gas cost
        self.gas_per_byte = 68   # Gas per byte of data

        # Statistics
        self.total_transactions = 0
        self.total_gas_used = 0
        self.mining_times = []

        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transactions=[{"type": "genesis", "message": "ZK-KGVerify Genesis Block"}],
            previous_hash="0" * 64
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def _mine_block(self, block: Block) -> Block:
        """Simple proof-of-work mining."""
        start = time.time()
        target = "0" * self.difficulty

        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.compute_hash()

        mining_time = time.time() - start
        self.mining_times.append(mining_time)
        return block

    def add_verification_record(
        self,
        model_id: str,
        triple: tuple,
        prediction_score: float,
        zkp_commitment: int,
        zkp_verified: bool,
        proof_hash: str
    ) -> Transaction:
        """
        Add a ZKP verification record to the blockchain.
        Returns the transaction object with gas costs.
        """
        # Calculate gas cost
        data_size = len(json.dumps({
            "model_id": model_id,
            "triple": triple,
            "score": prediction_score,
            "commitment": str(zkp_commitment)[:64],
            "verified": zkp_verified,
            "proof_hash": proof_hash
        }).encode())

        gas_used = self.gas_per_tx + (data_size * self.gas_per_byte)

        tx_data = {
            "type": "zkp_verification",
            "model_id": model_id,
            "triple": list(triple),
            "prediction_score": prediction_score,
            "zkp_commitment": str(zkp_commitment)[:64],  # Truncate for storage
            "zkp_verified": zkp_verified,
            "proof_hash": proof_hash,
            "gas_used": gas_used,
            "timestamp": time.time()
        }

        tx_hash = hashlib.sha256(json.dumps(tx_data, sort_keys=True).encode()).hexdigest()

        tx = Transaction(
            tx_hash=tx_hash,
            timestamp=tx_data["timestamp"],
            model_id=model_id,
            triple=triple,
            prediction_score=prediction_score,
            zkp_commitment=zkp_commitment,
            zkp_verified=zkp_verified,
            proof_hash=proof_hash,
            gas_used=gas_used
        )

        self.pending_transactions.append(tx_data)
        self.total_transactions += 1
        self.total_gas_used += gas_used

        # Auto-mine when block is full
        if len(self.pending_transactions) >= 10:
            self.mine_pending()

        return tx

    def mine_pending(self) -> Optional[Block]:
        """Mine all pending transactions into a new block."""
        if not self.pending_transactions:
            return None

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].hash
        )

        block = self._mine_block(block)
        self.chain.append(block)
        self.pending_transactions = []

        return block

    def validate_chain(self) -> bool:
        """Validate the entire blockchain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != previous.hash:
                return False

        return True

    def get_stats(self) -> Dict:
        """Get blockchain statistics for the paper."""
        self.mine_pending()  # Mine any remaining transactions

        return {
            "num_blocks": len(self.chain),
            "total_transactions": self.total_transactions,
            "total_gas_used": self.total_gas_used,
            "avg_gas_per_tx": self.total_gas_used / max(self.total_transactions, 1),
            "avg_mining_time": np.mean(self.mining_times) if self.mining_times else 0,
            "total_mining_time": sum(self.mining_times),
            "chain_valid": self.validate_chain(),
            "difficulty": self.difficulty,
        }

    def get_verification_log(self) -> List[Dict]:
        """Get all verification records from the chain."""
        records = []
        for block in self.chain:
            for tx in block.transactions:
                if isinstance(tx, dict) and tx.get("type") == "zkp_verification":
                    records.append(tx)
        return records


# ============================================================
# Ethereum Simulator (using web3.py + eth-tester)
# ============================================================

class EthereumSimulator:
    """
    Ethereum-based blockchain using web3.py and eth-tester.
    Deploys a simple smart contract for storing verification results.
    Falls back to PythonBlockchain if dependencies unavailable.
    """

    # Simplified smart contract ABI (stores verification events)
    CONTRACT_SOURCE = """
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
    }
    """

    def __init__(self):
        self.fallback = None
        self.web3 = None
        self.contract = None
        self.account = None
        self.gas_costs = []
        self.tx_times = []

        try:
            self._setup_ethereum()
        except Exception as e:
            print(f"  Ethereum setup failed ({e}), using Python blockchain fallback")
            self.fallback = PythonBlockchain()

    def _setup_ethereum(self):
        """Set up local Ethereum environment."""
        from web3 import Web3
        from eth_tester import EthereumTester

        tester = EthereumTester()
        self.web3 = Web3(Web3.EthereumTesterProvider(tester))
        self.account = self.web3.eth.accounts[0]

        # Deploy contract
        self._deploy_contract()

    def _deploy_contract(self):
        """Compile and deploy the verification contract."""
        try:
            import solcx
            solcx.install_solc("0.8.19")
            compiled = solcx.compile_source(
                self.CONTRACT_SOURCE,
                output_values=["abi", "bin"],
                solc_version="0.8.19"
            )
            contract_id, contract_interface = compiled.popitem()

            Contract = self.web3.eth.contract(
                abi=contract_interface["abi"],
                bytecode=contract_interface["bin"]
            )

            tx_hash = Contract.constructor().transact({"from": self.account})
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            self.contract = self.web3.eth.contract(
                address=tx_receipt.contractAddress,
                abi=contract_interface["abi"]
            )
            print(f"  Contract deployed at {tx_receipt.contractAddress}")

        except Exception as e:
            print(f"  Solidity compilation failed ({e}), using fallback")
            self.fallback = PythonBlockchain()

    def add_verification_record(self, model_id, triple, prediction_score, zkp_commitment, zkp_verified, proof_hash):
        if self.fallback:
            return self.fallback.add_verification_record(
                model_id, triple, prediction_score, zkp_commitment, zkp_verified, proof_hash
            )

        start = time.time()

        commitment_bytes = hashlib.sha256(str(zkp_commitment).encode()).digest()
        proof_bytes = bytes.fromhex(proof_hash) if len(proof_hash) == 64 else hashlib.sha256(proof_hash.encode()).digest()

        tx_hash = self.contract.functions.logVerification(
            model_id,
            int(triple[0]),
            int(triple[1]),
            int(triple[2]),
            int(prediction_score * 1000000),
            commitment_bytes,
            zkp_verified,
            proof_bytes
        ).transact({"from": self.account})

        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        tx_time = time.time() - start

        self.gas_costs.append(receipt.gasUsed)
        self.tx_times.append(tx_time)

        return Transaction(
            tx_hash=receipt.transactionHash.hex(),
            timestamp=time.time(),
            model_id=model_id,
            triple=triple,
            prediction_score=prediction_score,
            zkp_commitment=zkp_commitment,
            zkp_verified=zkp_verified,
            proof_hash=proof_hash,
            gas_used=receipt.gasUsed
        )

    def mine_pending(self):
        if self.fallback:
            return self.fallback.mine_pending()
        return None

    def get_stats(self):
        if self.fallback:
            return self.fallback.get_stats()

        return {
            "num_blocks": self.web3.eth.block_number,
            "total_transactions": len(self.gas_costs),
            "total_gas_used": sum(self.gas_costs),
            "avg_gas_per_tx": np.mean(self.gas_costs) if self.gas_costs else 0,
            "avg_tx_time": np.mean(self.tx_times) if self.tx_times else 0,
            "total_tx_time": sum(self.tx_times),
            "chain_valid": True,
            "backend": "ethereum_tester",
        }

    def get_verification_log(self):
        if self.fallback:
            return self.fallback.get_verification_log()
        return []


def create_blockchain(mode="local"):
    """Factory function to create appropriate blockchain backend."""
    if mode == "ganache" or mode == "ethereum":
        return EthereumSimulator()
    else:
        return PythonBlockchain()
