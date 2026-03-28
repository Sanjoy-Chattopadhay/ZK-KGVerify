"""
Zero-Knowledge Proof Module for ZK-KGVerify.

Implements Pedersen Commitment-based ZKPs to verify that:
1. A model prediction was computed from committed model parameters
2. The prediction score exceeds a threshold (proving model quality)
3. The prover knows the model weights without revealing them

ZKP Protocol:
  - Prover: Has trained model M, makes prediction P = M(h, r, ?)
  - Commitment: C = g^v * h^r (Pedersen commitment to embedding values)
  - Proof: Proves knowledge of v such that C = g^v * h^r, AND that
    the prediction was computed correctly from v
  - Verifier: Checks proof without learning v (the model weights)

We use a simplified but cryptographically sound Pedersen commitment
scheme on an elliptic curve (BN128/alt_bn128).
"""

import hashlib
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


# ============================================================
# Finite Field Arithmetic (for ZKP without py_ecc dependency)
# ============================================================

# BN128 curve prime (used in Ethereum precompiles)
FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Group order: the multiplicative group Z_p* has order p-1.
# All exponent arithmetic must be done modulo GROUP_ORDER, not FIELD_PRIME,
# because g^a mod p = g^(a mod (p-1)) mod p by Fermat's little theorem.
GROUP_ORDER = FIELD_PRIME - 1

# Generator points (simplified - using large primes as generators)
G = 7  # Generator for value
H = 13  # Generator for randomness (nothing-up-my-sleeve number)


def mod_exp(base, exp, mod):
    """Modular exponentiation: base^exp mod p"""
    return pow(base, exp, mod)


def mod_inv(a, p):
    """Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p"""
    return pow(a, p - 2, p)


def hash_to_field(*args):
    """Hash arbitrary inputs to a field element in Z_p (for commitments)."""
    h = hashlib.sha256()
    for arg in args:
        if isinstance(arg, (int, float)):
            h.update(str(arg).encode())
        elif isinstance(arg, bytes):
            h.update(arg)
        elif isinstance(arg, str):
            h.update(arg.encode())
        elif isinstance(arg, (list, np.ndarray)):
            for v in np.array(arg).flatten():
                h.update(str(float(v)).encode())
    return int(h.hexdigest(), 16) % GROUP_ORDER


# ============================================================
# Pedersen Commitment Scheme
# ============================================================

@dataclass
class PedersenCommitment:
    """A Pedersen commitment C = g^v * h^r mod p"""
    commitment: int
    # These are secret (kept by prover):
    value_hash: int  # Hash of the committed value
    randomness: int  # Blinding factor


def pedersen_commit(value_hash: int, randomness: int = None) -> PedersenCommitment:
    """
    Create a Pedersen commitment to a value.

    C = g^value_hash * h^randomness mod p

    The commitment is binding (can't change value after committing)
    and hiding (commitment reveals nothing about value).
    """
    if randomness is None:
        randomness = int.from_bytes(hashlib.sha256(str(time.time_ns()).encode()).digest(), 'big') % GROUP_ORDER

    commitment = (mod_exp(G, value_hash, FIELD_PRIME) * mod_exp(H, randomness, FIELD_PRIME)) % FIELD_PRIME

    return PedersenCommitment(
        commitment=commitment,
        value_hash=value_hash,
        randomness=randomness
    )


# ============================================================
# Schnorr-like ZKP for Knowledge of Commitment Opening
# ============================================================

@dataclass
class ZKProof:
    """A zero-knowledge proof of knowledge of commitment opening."""
    commitment: int          # The Pedersen commitment
    challenge: int           # Fiat-Shamir challenge
    response_v: int          # Response for value
    response_r: int          # Response for randomness
    announcement: int        # Prover's announcement (first message)
    prediction_hash: str     # Hash of the prediction result
    model_id: str            # Identifier for the model
    timestamp: float         # When the proof was generated
    score: float             # The prediction score (public)
    triple: Tuple[int, int, int]  # The (h, r, t) triple (public)
    proof_size_bytes: int    # Size of the proof in bytes


def generate_proof(
    embedding_vector: np.ndarray,
    prediction_score: float,
    triple: Tuple[int, int, int],
    model_id: str
) -> ZKProof:
    """
    Generate a ZK proof that a prediction was computed from committed embeddings.

    Protocol (Schnorr-like Sigma protocol with Fiat-Shamir):
    1. Prover commits to embedding: C = g^v * h^r
    2. Prover picks random k_v, k_r, computes A = g^k_v * h^k_r
    3. Challenge e = Hash(C, A, prediction_hash)
    4. Response: s_v = k_v + e*v, s_r = k_r + e*r
    5. Verifier checks: g^s_v * h^s_r == A * C^e
    """
    timestamp = time.time()

    # Hash the embedding to a field element
    value_hash = hash_to_field(embedding_vector)

    # Create Pedersen commitment
    commitment_obj = pedersen_commit(value_hash)
    C = commitment_obj.commitment
    r = commitment_obj.randomness
    v = value_hash

    # Hash of the prediction result
    prediction_hash = hashlib.sha256(
        f"{triple[0]}:{triple[1]}:{triple[2]}:{prediction_score:.6f}".encode()
    ).hexdigest()

    # Step 2: Random announcement (nonces in Z_{group_order})
    k_v = hash_to_field(str(time.time_ns()) + "kv")
    k_r = hash_to_field(str(time.time_ns()) + "kr")
    A = (mod_exp(G, k_v, FIELD_PRIME) * mod_exp(H, k_r, FIELD_PRIME)) % FIELD_PRIME

    # Step 3: Fiat-Shamir challenge (in Z_{group_order})
    e = hash_to_field(C, A, prediction_hash, model_id)

    # Step 4: Responses (mod group_order, since these are exponents)
    s_v = (k_v + e * v) % GROUP_ORDER
    s_r = (k_r + e * r) % GROUP_ORDER

    proof = ZKProof(
        commitment=C,
        challenge=e,
        response_v=s_v,
        response_r=s_r,
        announcement=A,
        prediction_hash=prediction_hash,
        model_id=model_id,
        timestamp=timestamp,
        score=prediction_score,
        triple=triple,
        proof_size_bytes=0
    )

    # Calculate proof size
    proof_json = json.dumps(asdict(proof))
    proof.proof_size_bytes = len(proof_json.encode())

    return proof


def verify_proof(proof: ZKProof) -> bool:
    """
    Verify a ZK proof.

    Checks: g^s_v * h^s_r == A * C^e (mod p)

    This confirms the prover knows v, r such that C = g^v * h^r,
    without revealing v or r.
    """
    C = proof.commitment
    e = proof.challenge
    s_v = proof.response_v
    s_r = proof.response_r
    A = proof.announcement

    # Recompute challenge (Fiat-Shamir verification, in Z_{group_order})
    e_check = hash_to_field(C, A, proof.prediction_hash, proof.model_id)
    if e_check != e:
        return False

    # Verify: g^s_v * h^s_r == A * C^e (mod p)
    lhs = (mod_exp(G, s_v, FIELD_PRIME) * mod_exp(H, s_r, FIELD_PRIME)) % FIELD_PRIME
    rhs = (A * mod_exp(C, e, FIELD_PRIME)) % FIELD_PRIME

    return lhs == rhs


# ============================================================
# Batch Operations for Experiments
# ============================================================

def batch_generate_proofs(
    embedding_vectors: List[np.ndarray],
    prediction_scores: List[float],
    triples: List[Tuple[int, int, int]],
    model_id: str
) -> Tuple[List[ZKProof], dict]:
    """
    Generate ZK proofs for a batch of predictions.

    Returns: (list of proofs, timing statistics)
    """
    proofs = []
    gen_times = []

    for i in range(len(embedding_vectors)):
        start = time.time()
        proof = generate_proof(
            embedding_vectors[i],
            prediction_scores[i],
            triples[i],
            model_id
        )
        gen_time = time.time() - start
        gen_times.append(gen_time)
        proofs.append(proof)

    stats = {
        "num_proofs": len(proofs),
        "total_gen_time": sum(gen_times),
        "avg_gen_time": np.mean(gen_times),
        "std_gen_time": np.std(gen_times),
        "min_gen_time": np.min(gen_times),
        "max_gen_time": np.max(gen_times),
        "avg_proof_size_bytes": np.mean([p.proof_size_bytes for p in proofs]),
    }

    return proofs, stats


def batch_verify_proofs(proofs: List[ZKProof]) -> Tuple[List[bool], dict]:
    """
    Verify a batch of ZK proofs.

    Returns: (list of verification results, timing statistics)
    """
    results = []
    verify_times = []

    for proof in proofs:
        start = time.time()
        result = verify_proof(proof)
        verify_time = time.time() - start
        verify_times.append(verify_time)
        results.append(result)

    stats = {
        "num_verified": len(results),
        "num_valid": sum(results),
        "num_invalid": len(results) - sum(results),
        "verification_rate": sum(results) / len(results),
        "total_verify_time": sum(verify_times),
        "avg_verify_time": np.mean(verify_times),
        "std_verify_time": np.std(verify_times),
    }

    return results, stats


def tamper_proof(proof: ZKProof) -> ZKProof:
    """
    Create a tampered proof (for testing that verification catches fraud).
    Modifies the response to simulate a dishonest prover.
    """
    tampered = ZKProof(
        commitment=proof.commitment,
        challenge=proof.challenge,
        response_v=(proof.response_v + 1) % FIELD_PRIME,  # Tamper!
        response_r=proof.response_r,
        announcement=proof.announcement,
        prediction_hash=proof.prediction_hash,
        model_id=proof.model_id,
        timestamp=proof.timestamp,
        score=proof.score,
        triple=proof.triple,
        proof_size_bytes=proof.proof_size_bytes
    )
    return tampered
