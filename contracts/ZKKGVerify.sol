// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ZKKGVerify
 * @notice Stores zero-knowledge proof verification records for
 *         knowledge graph link predictions on an immutable ledger.
 *
 * Each record captures:
 *   - Which model made the prediction (modelId)
 *   - The predicted triple (head, relation, tail entity IDs)
 *   - The prediction score (scaled to int for Solidity)
 *   - The Pedersen commitment to the embedding vector
 *   - Whether the ZK proof was verified successfully
 *   - A hash of the full proof for off-chain lookup
 *
 * USAGE (from Python with web3.py):
 *   contract.functions.logVerification(
 *       "CompGCN",       # modelId
 *       1234,            # head entity ID
 *       42,              # relation ID
 *       5678,            # tail entity ID
 *       -2310000,        # score * 1e6 (int)
 *       commitmentHash,  # bytes32 Pedersen commitment hash
 *       True,            # zkpVerified
 *       proofHash        # bytes32 hash of the full proof
 *   ).transact({"from": account})
 */
contract ZKKGVerify {

    struct VerificationRecord {
        string modelId;
        uint256 head;
        uint256 relation;
        uint256 tail;
        int256 score;           // prediction score * 1e6
        bytes32 commitment;     // Pedersen commitment hash
        bool verified;          // ZKP verification result
        bytes32 proofHash;      // SHA-256 of the full proof
        uint256 timestamp;      // block.timestamp
    }

    // Storage
    VerificationRecord[] public records;
    uint256 public recordCount;

    // Events — indexed for efficient off-chain querying
    event VerificationLogged(
        uint256 indexed recordId,
        string modelId,
        bool verified,
        uint256 timestamp
    );

    /**
     * @notice Log a verified KG prediction on-chain.
     * @return recordId The ID of the newly created record.
     */
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

    /**
     * @notice Retrieve a verification record by ID.
     */
    function getRecord(uint256 _id)
        public view returns (VerificationRecord memory)
    {
        require(_id < recordCount, "Record does not exist");
        return records[_id];
    }

    /**
     * @notice Get the total number of verification records.
     */
    function getRecordCount() public view returns (uint256) {
        return recordCount;
    }
}
