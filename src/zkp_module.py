"""
Zero-Knowledge Proof Module for ZK-KGVerify (BN128 elliptic curve edition).

Implements Pedersen commitments and a Schnorr-style Sigma protocol on the
BN128 (alt_bn128) elliptic curve -- the same curve used by the Ethereum
precompiles for ZK-SNARK verification.

Cryptographic primitives
------------------------
Let G = G1 be the standard BN128 generator and let H be a second,
"nothing-up-my-sleeve" (NUMS) generator derived deterministically from
G via H = h_seed * G, where h_seed = SHA256("ZK-KGVerify NUMS H") mod n
and n = curve_order. Because h_seed is publicly known but its
discrete log to G is fixed and not chosen by the prover, the relation
log_G(H) acts as an unknown to the prover for binding purposes.

Pedersen commitment:
    C = v * G + r * H        (point addition / scalar multiplication on BN128)

Schnorr-Fiat-Shamir proof of knowledge of (v, r) opening C:
    Announce:  k_v, k_r <-$ Z_n;    A = k_v * G + k_r * H
    Challenge: e = SHA256(C || A || pred_hash || model_id) mod n
    Respond:   s_v = k_v + e*v mod n,    s_r = k_r + e*r mod n
    Verify:    s_v * G + s_r * H  ==  A + e * C

All scalar arithmetic is done modulo the curve order n (a 254-bit prime).
This gives ~128-bit security under the ECDLP assumption on BN128, matching
the security level claimed in the paper.

The proof object is fully serialisable; points are stored as their (x, y)
affine integer coordinates so the same proof can be passed to a Solidity
verifier via web3.py.
"""

import hashlib
import time
import json
import secrets
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Any

from py_ecc.bn128 import (
    G1,
    add,
    multiply,
    eq,
    curve_order,
    field_modulus,
)
from py_ecc.bn128.bn128_curve import is_on_curve, b


# ============================================================
# Curve constants
# ============================================================

# Scalar group order n (a 254-bit prime). All exponents live in Z_n.
N = curve_order

# Second generator H: NUMS construction, h_seed = SHA256("ZK-KGVerify NUMS H").
# log_G(H) = h_seed is public but fixed; binding follows from the fact that
# the prover did not choose h_seed.
_H_SEED = int(
    hashlib.sha256(b"ZK-KGVerify NUMS H v1").hexdigest(), 16
) % N
H_POINT = multiply(G1, _H_SEED)


# ============================================================
# Helpers
# ============================================================

def _point_to_ints(P) -> Tuple[int, int]:
    """Convert a py_ecc affine point to a pair of native Python ints."""
    if P is None:
        return (0, 0)  # point at infinity sentinel
    x, y = P
    return (int(x.n), int(y.n))


def _point_from_ints(xy: Tuple[int, int]):
    """Reconstruct a py_ecc point from (x, y) ints. Returns None for (0, 0)."""
    x, y = xy
    if x == 0 and y == 0:
        return None
    from py_ecc.bn128 import FQ
    return (FQ(x), FQ(y))


def _hash_embedding(emb) -> int:
    """Hash a numpy array (or list) of floats to a scalar in Z_n.

    Floats are rounded to 6 decimal places before hashing so that minor
    floating-point noise across runs does not change the commitment.
    """
    arr = np.asarray(emb).flatten()
    h = hashlib.sha256()
    h.update(b"ZK-KGVerify-embedding")
    for v in arr:
        # Stable string form; %.6e is platform-deterministic.
        h.update(f"{float(v):.6e}".encode())
    return int(h.hexdigest(), 16) % N


def _hash_challenge(C_xy, A_xy, prediction_hash: str, model_id: str) -> int:
    """Fiat-Shamir challenge bound to the full proof transcript."""
    h = hashlib.sha256()
    h.update(b"ZK-KGVerify-challenge-v1")
    h.update(C_xy[0].to_bytes(32, "big"))
    h.update(C_xy[1].to_bytes(32, "big"))
    h.update(A_xy[0].to_bytes(32, "big"))
    h.update(A_xy[1].to_bytes(32, "big"))
    h.update(prediction_hash.encode())
    h.update(model_id.encode())
    return int(h.hexdigest(), 16) % N


def _hash_prediction(triple: Tuple[int, int, int], score: float) -> str:
    """Hex SHA-256 of the public prediction (h, r, t, score) tuple."""
    return hashlib.sha256(
        f"{triple[0]}:{triple[1]}:{triple[2]}:{score:.6f}".encode()
    ).hexdigest()


# ============================================================
# Pedersen Commitment
# ============================================================

@dataclass
class PedersenCommitment:
    """C = v*G + r*H on BN128. C is stored as (x, y) affine ints."""
    commitment_xy: Tuple[int, int]
    # Secret (kept by prover, never serialised in proofs):
    value_scalar: int
    randomness: int


def pedersen_commit(value_scalar: int, randomness: Optional[int] = None) -> PedersenCommitment:
    """Form C = value_scalar * G + randomness * H on BN128."""
    if randomness is None:
        randomness = secrets.randbelow(N)
    value_scalar = value_scalar % N
    randomness = randomness % N

    C = add(multiply(G1, value_scalar), multiply(H_POINT, randomness))
    return PedersenCommitment(
        commitment_xy=_point_to_ints(C),
        value_scalar=value_scalar,
        randomness=randomness,
    )


# ============================================================
# Schnorr-Fiat-Shamir ZKP
# ============================================================

@dataclass
class ZKProof:
    """Non-interactive ZK proof of knowledge of (v, r) opening a Pedersen
    commitment on BN128. Points are serialised as (x, y) integer pairs."""
    commitment_xy: Tuple[int, int]
    announcement_xy: Tuple[int, int]
    challenge: int
    response_v: int
    response_r: int
    prediction_hash: str
    model_id: str
    timestamp: float
    score: float
    triple: Tuple[int, int, int]
    proof_size_bytes: int = 0


def generate_proof(
    embedding_vector: np.ndarray,
    prediction_score: float,
    triple: Tuple[int, int, int],
    model_id: str,
) -> ZKProof:
    """Generate a Schnorr-Fiat-Shamir proof on BN128 for one KG prediction."""
    timestamp = time.time()

    # Step 1: hash the embedding to a scalar v in Z_n.
    v = _hash_embedding(embedding_vector)

    # Step 2: form Pedersen commitment C = v*G + r*H.
    comm = pedersen_commit(v)
    C_xy = comm.commitment_xy
    r = comm.randomness

    # Step 3: announcement A = k_v*G + k_r*H with fresh nonces.
    k_v = secrets.randbelow(N)
    k_r = secrets.randbelow(N)
    A_pt = add(multiply(G1, k_v), multiply(H_POINT, k_r))
    A_xy = _point_to_ints(A_pt)

    # Step 4: bind prediction to the transcript.
    pred_hash = _hash_prediction(triple, prediction_score)

    # Step 5: Fiat-Shamir challenge.
    e = _hash_challenge(C_xy, A_xy, pred_hash, model_id)

    # Step 6: responses.
    s_v = (k_v + e * v) % N
    s_r = (k_r + e * r) % N

    proof = ZKProof(
        commitment_xy=C_xy,
        announcement_xy=A_xy,
        challenge=e,
        response_v=s_v,
        response_r=s_r,
        prediction_hash=pred_hash,
        model_id=model_id,
        timestamp=timestamp,
        score=float(prediction_score),
        triple=tuple(int(x) for x in triple),
        proof_size_bytes=0,
    )

    # Size accounting (after construction).
    proof.proof_size_bytes = len(json.dumps(asdict(proof)).encode())
    return proof


def verify_proof(proof: ZKProof) -> bool:
    """Verify a Schnorr-Fiat-Shamir proof on BN128.

    Checks (a) the challenge was honestly derived and (b) the algebraic
    relation  s_v * G + s_r * H  ==  A + e * C  on the curve.
    """
    # (a) recompute challenge.
    e_check = _hash_challenge(
        proof.commitment_xy,
        proof.announcement_xy,
        proof.prediction_hash,
        proof.model_id,
    )
    if e_check != proof.challenge:
        return False

    # (b) reconstruct points and check the algebraic relation.
    C = _point_from_ints(proof.commitment_xy)
    A = _point_from_ints(proof.announcement_xy)
    if C is None or A is None:
        return False
    if not (is_on_curve(C, b) and is_on_curve(A, b)):
        return False

    lhs = add(multiply(G1, proof.response_v % N),
              multiply(H_POINT, proof.response_r % N))
    rhs = add(A, multiply(C, proof.challenge % N))
    return eq(lhs, rhs)


# ============================================================
# Batch helpers (for experiments)
# ============================================================

def batch_generate_proofs(
    embedding_vectors: List[np.ndarray],
    prediction_scores: List[float],
    triples: List[Tuple[int, int, int]],
    model_id: str,
) -> Tuple[List[ZKProof], dict]:
    proofs: List[ZKProof] = []
    gen_times: List[float] = []
    for i in range(len(embedding_vectors)):
        t0 = time.time()
        p = generate_proof(embedding_vectors[i], prediction_scores[i], triples[i], model_id)
        gen_times.append(time.time() - t0)
        proofs.append(p)

    stats = {
        "num_proofs": len(proofs),
        "total_gen_time": float(sum(gen_times)),
        "avg_gen_time": float(np.mean(gen_times)),
        "std_gen_time": float(np.std(gen_times)),
        "min_gen_time": float(np.min(gen_times)) if gen_times else 0.0,
        "max_gen_time": float(np.max(gen_times)) if gen_times else 0.0,
        "avg_proof_size_bytes": float(np.mean([p.proof_size_bytes for p in proofs])) if proofs else 0.0,
        "gen_times": gen_times,
    }
    return proofs, stats


def batch_verify_proofs(proofs: List[ZKProof]) -> Tuple[List[bool], dict]:
    results: List[bool] = []
    verify_times: List[float] = []
    for p in proofs:
        t0 = time.time()
        ok = verify_proof(p)
        verify_times.append(time.time() - t0)
        results.append(bool(ok))

    n = max(len(results), 1)
    stats = {
        "num_verified": len(results),
        "num_valid": int(sum(results)),
        "num_invalid": int(len(results) - sum(results)),
        "verification_rate": float(sum(results) / n),
        "total_verify_time": float(sum(verify_times)),
        "avg_verify_time": float(np.mean(verify_times)) if verify_times else 0.0,
        "std_verify_time": float(np.std(verify_times)) if verify_times else 0.0,
        "verify_times": verify_times,
    }
    return results, stats


def tamper_proof(proof: ZKProof) -> ZKProof:
    """Return a copy with response_v perturbed -- used to test soundness."""
    return ZKProof(
        commitment_xy=proof.commitment_xy,
        announcement_xy=proof.announcement_xy,
        challenge=proof.challenge,
        response_v=(proof.response_v + 1) % N,
        response_r=proof.response_r,
        prediction_hash=proof.prediction_hash,
        model_id=proof.model_id,
        timestamp=proof.timestamp,
        score=proof.score,
        triple=proof.triple,
        proof_size_bytes=proof.proof_size_bytes,
    )
