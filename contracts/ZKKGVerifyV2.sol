// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ZKKGVerifyV2
 * @notice Trust-minimised audit ledger for knowledge-graph link predictions.
 *
 * WHAT THIS CONTRACT ENFORCES
 * ---------------------------
 * A model owner registers, once and in public, a Pedersen commitment to a
 * digest of its trained weights. From then on, every prediction it publishes
 * must carry a Schnorr proof of knowledge of the opening of *that registered
 * commitment*, with the prediction bound into the Fiat-Shamir challenge. The
 * EVM checks the proof itself using the alt_bn128 precompiles at 0x06 (ECADD)
 * and 0x07 (ECMUL) and reverts if it fails, so a stored record is an
 * attestation by consensus rather than a claim by the writer.
 *
 * TWO DESIGN POINTS THAT ARE EASY TO GET WRONG
 * --------------------------------------------
 * 1. The commitment is read from the registry, never from the caller.
 *    Accepting a caller-supplied commitment would make the whole exercise
 *    vacuous: the prover would commit to a value after seeing the query and
 *    then prove it knows the value it just chose. Pre-registration is what
 *    turns "I know some opening" into "I am the party that registered this
 *    model", and the registration block number timestamps that claim.
 *
 * 2. The prediction digest is recomputed here from the fields being stored.
 *    If the caller supplied an opaque predHash, the proof would bind to that
 *    blob while the stored triple and score could say anything. Recomputing
 *    makes the record itself the object the proof commits to.
 *
 * GENERATOR H
 * -----------
 * H is a nothing-up-my-sleeve point from hashing
 * "ZK-KGVerify/BN128/NUMS-H/v2" to the curve (src/zkp_module.py::hash_to_curve).
 * Nobody knows log_G(H), which is what makes the commitment binding. Deriving
 * H as h_seed*G for a public h_seed -- as an earlier version of this project
 * did -- destroys binding outright, because C then collapses to a commitment
 * to the single scalar (v + r*h_seed) and reopens to any value via one
 * modular inverse.
 *
 * WHAT IT DOES NOT PROVE
 * ----------------------
 * That the forward pass was executed correctly. This scheme gives accountable
 * attribution and non-repudiation, not verified computation. See docs/THEORY.md.
 */
contract ZKKGVerifyV2 {
    // ---- alt_bn128 constants ----

    uint256 internal constant N =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;
    uint256 internal constant P =
        21888242871839275222246405745257275088696311157297823662689037894645226208583;

    uint256 internal constant GX = 1;
    uint256 internal constant GY = 2;

    /// @dev H = HashToCurve("ZK-KGVerify/BN128/NUMS-H/v2").
    uint256 internal constant HX =
        9451862166084347742712146928789237371442632068834107658244224723734334334814;
    uint256 internal constant HY =
        6132405762163060342931858727720913388474455288098316721609868025918973292334;

    bytes internal constant PRED_TAG = "ZK-KGVerify-prediction-v3";

    // ---- registry ----

    struct Model {
        uint256 cx;
        uint256 cy;
        address owner;
        uint64  registeredAtBlock;
        bytes32 weightDigest; // informational; the binding is via (cx, cy)
        bool    exists;
    }

    mapping(bytes32 => Model) public models; // keccak256(modelId) => Model

    // ---- ledger ----

    struct Record {
        string  modelId;
        uint256 head;
        uint256 relation;
        uint256 tail;
        int256  score;      // prediction score scaled by 1e6
        bytes32 aux;        // commitment to private context (e.g. embedding)
        bytes32 predHash;   // recomputed here, not supplied
        uint64  timestamp;
    }

    Record[] public records;
    uint256 public recordCount;

    event ModelRegistered(
        bytes32 indexed modelKey,
        string  modelId,
        address indexed owner,
        uint256 cx,
        uint256 cy
    );

    event VerificationLogged(
        uint256 indexed recordId,
        bytes32 indexed modelKey,
        bytes32 indexed predHash,
        uint256 timestamp
    );

    error PointNotOnCurve();
    error ScalarOutOfRange();
    error ProofRejected();
    error PrecompileFailed();
    error ModelNotRegistered();
    error ModelAlreadyRegistered();

    // ---- registration ----

    /**
     * @notice Publish the commitment that all future proofs must open.
     * @dev Registration is single-shot per modelId. Allowing re-registration
     *      would let an owner swap the committed model after publishing
     *      predictions, which is precisely the substitution the ledger exists
     *      to rule out. A new model needs a new modelId.
     */
    function registerModel(
        string calldata modelId,
        uint256[2] calldata C,
        bytes32 weightDigest
    ) external returns (bytes32 modelKey) {
        uint256[2] memory Cm = [C[0], C[1]];
        if (!_onCurve(Cm)) revert PointNotOnCurve();

        modelKey = keccak256(bytes(modelId));
        if (models[modelKey].exists) revert ModelAlreadyRegistered();

        models[modelKey] = Model({
            cx: C[0],
            cy: C[1],
            owner: msg.sender,
            registeredAtBlock: uint64(block.number),
            weightDigest: weightDigest,
            exists: true
        });

        emit ModelRegistered(modelKey, modelId, msg.sender, C[0], C[1]);
    }

    function isRegistered(string calldata modelId) external view returns (bool) {
        return models[keccak256(bytes(modelId))].exists;
    }

    function modelCommitment(string calldata modelId)
        external
        view
        returns (uint256 cx, uint256 cy, address owner, uint64 atBlock)
    {
        Model storage m = models[keccak256(bytes(modelId))];
        if (!m.exists) revert ModelNotRegistered();
        return (m.cx, m.cy, m.owner, m.registeredAtBlock);
    }

    // ---- elliptic curve helpers ----

    function _ecAdd(uint256[2] memory a, uint256[2] memory b)
        internal
        view
        returns (uint256[2] memory r)
    {
        uint256[4] memory input = [a[0], a[1], b[0], b[1]];
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x06, input, 0x80, r, 0x40)
        }
        if (!ok) revert PrecompileFailed();
    }

    function _ecMul(uint256[2] memory p, uint256 s)
        internal
        view
        returns (uint256[2] memory r)
    {
        uint256[3] memory input = [p[0], p[1], s];
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x07, input, 0x60, r, 0x40)
        }
        if (!ok) revert PrecompileFailed();
    }

    /**
     * @dev Reject anything that is not a finite affine point of y^2 = x^3 + 3.
     *      Load-bearing: without it a forger could submit an off-curve
     *      announcement and drive the precompiles outside the intended group.
     *      G1 on alt_bn128 has prime order equal to the group order, so
     *      on-curve implies in-subgroup and no cofactor check is needed.
     */
    function _onCurve(uint256[2] memory p) internal pure returns (bool) {
        if (p[0] >= P || p[1] >= P) return false;
        if (p[0] == 0 && p[1] == 0) return false;
        uint256 lhs = mulmod(p[1], p[1], P);
        uint256 rhs = addmod(mulmod(mulmod(p[0], p[0], P), p[0], P), 3, P);
        return lhs == rhs;
    }

    // ---- digests ----

    /// @dev Must stay byte-identical to zkp_module.hash_prediction.
    function predictionHash(
        uint256[3] memory triple,
        int256 score,
        bytes32 aux
    ) public pure returns (bytes32) {
        return keccak256(
            abi.encodePacked(PRED_TAG, triple[0], triple[1], triple[2], score, aux)
        );
    }

    /// @dev Must stay byte-identical to zkp_module.hash_challenge.
    function challenge(
        uint256[2] memory C,
        uint256[2] memory A,
        bytes32 predHash,
        string memory modelId
    ) public pure returns (uint256) {
        return uint256(
            keccak256(abi.encodePacked(C[0], C[1], A[0], A[1], predHash, modelId))
        ) % N;
    }

    // ---- verification ----

    /**
     * @dev The group equation, factored out of `verifyProof`.
     *
     *      Keeping the elliptic-curve arithmetic in its own frame is not
     *      cosmetic: `verifyProof` takes a calldata string (two stack slots)
     *      plus eight further arguments, and inlining the four intermediate
     *      points here overflows the EVM's 16-slot reachable stack.
     */
    function _checkSchnorr(
        uint256[2] memory C,
        uint256[2] memory A,
        uint256 sv,
        uint256 sr,
        uint256 e
    ) internal view returns (bool) {
        uint256[2] memory lhs = _ecAdd(_ecMul([GX, GY], sv), _ecMul([HX, HY], sr));
        uint256[2] memory rhs = _ecAdd(A, _ecMul(C, e));
        return lhs[0] == rhs[0] && lhs[1] == rhs[1];
    }

    /**
     * @notice Check a proof against the registered commitment, no state written.
     * @dev Free via eth_call; eth_estimateGas isolates verification cost.
     */
    function verifyProof(
        string calldata modelId,
        uint256[3] calldata triple,
        int256 score,
        uint256[2] calldata A,
        uint256 sv,
        uint256 sr,
        bytes32 aux
    ) public view returns (bool) {
        if (sv >= N || sr >= N) revert ScalarOutOfRange();

        uint256[2] memory Cm;
        {
            Model storage m = models[keccak256(bytes(modelId))];
            if (!m.exists) revert ModelNotRegistered();
            Cm = [m.cx, m.cy];
        }

        uint256[2] memory Am = [A[0], A[1]];
        if (!_onCurve(Am)) revert PointNotOnCurve();

        uint256 e = challenge(
            Cm,
            Am,
            predictionHash([triple[0], triple[1], triple[2]], score, aux),
            modelId
        );

        return _checkSchnorr(Cm, Am, sv, sr, e);
    }

    /**
     * @notice Verify, then append an audit record only if the proof holds.
     */
    function verifyAndLog(
        string calldata modelId,
        uint256[3] calldata triple,
        int256 score,
        uint256[2] calldata A,
        uint256 sv,
        uint256 sr,
        bytes32 aux
    ) external returns (uint256 recordId) {
        if (!verifyProof(modelId, triple, score, A, sv, sr, aux)) {
            revert ProofRejected();
        }

        bytes32 ph = predictionHash([triple[0], triple[1], triple[2]], score, aux);

        records.push(
            Record({
                modelId: modelId,
                head: triple[0],
                relation: triple[1],
                tail: triple[2],
                score: score,
                aux: aux,
                predHash: ph,
                timestamp: uint64(block.timestamp)
            })
        );

        recordId = recordCount;
        recordCount++;

        emit VerificationLogged(recordId, keccak256(bytes(modelId)), ph, block.timestamp);
    }

    // ---- views ----

    function getRecord(uint256 id) external view returns (Record memory) {
        require(id < recordCount, "Record does not exist");
        return records[id];
    }

    function getRecordCount() external view returns (uint256) {
        return recordCount;
    }

    /// @notice Expose the generators so clients can assert parameter parity.
    function generators()
        external
        pure
        returns (uint256 gx, uint256 gy, uint256 hx, uint256 hy)
    {
        return (GX, GY, HX, HY);
    }
}
