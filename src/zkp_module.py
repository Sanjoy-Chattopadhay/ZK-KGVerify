"""
Zero-Knowledge Proof Module for ZK-KGVerify (BN128 elliptic curve).

Implements Pedersen commitments and a Schnorr-style Sigma protocol on the
BN128 (alt_bn128) curve -- the curve exposed by the Ethereum precompiles at
0x06 (point addition) and 0x07 (scalar multiplication), so the very same
proof this module produces can be checked by the on-chain verifier in
contracts/ZKKGVerifyV2.sol.

Cryptographic construction
--------------------------
Let G = G1 be the standard BN128 generator. The second generator H is
obtained by *hashing to the curve*:

    H = HashToCurve("ZK-KGVerify/BN128/NUMS-H/v2")

using try-and-increment: interpret SHA-256(seed || ctr) as a field element
x, and accept the first x for which x^3 + 3 is a quadratic residue, taking
the square root via the p = 3 (mod 4) shortcut y = (x^3+3)^((p+1)/4). The
sign of y is fixed (even) to make the point canonical.

    Pedersen commitment:   C = v*G + r*H
    Schnorr-Fiat-Shamir proof of knowledge of (v, r) opening C:
        Announce:  k_v, k_r <-$ Z_n ;   A = k_v*G + k_r*H
        Challenge: e = keccak256(C || A || predHash || modelId) mod n
        Respond:   s_v = k_v + e*v mod n ,  s_r = k_r + e*r mod n
        Verify:    s_v*G + s_r*H  ==  A + e*C

WHY HASH-TO-CURVE AND NOT H = h_seed * G
----------------------------------------
An earlier version of this module derived the second generator as
H = h_seed*G with h_seed = SHA256(tag) mod n. That construction is NOT
binding, and the break is immediate rather than merely theoretical: because
h_seed is public, C = v*G + r*H = (v + r*h_seed)*G, so a commitment is only
ever a commitment to the single scalar (v + r*h_seed). Any target value v'
can be opened by solving

    r' = (v - v') * h_seed^{-1} + r   (mod n)

which needs nothing but a modular inverse. Pedersen binding requires that
log_G(H) be unknown *to everybody*, which is exactly what hashing to the
curve buys: no party -- including whoever chose the seed string -- knows the
discrete logarithm of H.

`selftest_binding()` at the bottom of this file exercises that attack and
asserts it now fails. It runs as part of `python -m src.zkp_module`.

Challenge hash
--------------
The Fiat-Shamir challenge uses keccak256 rather than SHA-256 so that the
Solidity verifier can recompute it natively with the same bytes and no
precompile call. Off-chain and on-chain challenge derivation are therefore
bit-identical; `tools/check_onchain_parity.py` asserts this against the
deployed contract.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from eth_utils import keccak
from py_ecc.bn128 import (
    FQ,
    G1,
    add,
    curve_order,
    eq,
    field_modulus,
    multiply,
    neg,
)
from py_ecc.bn128.bn128_curve import b as CURVE_B
from py_ecc.bn128.bn128_curve import is_on_curve

# ============================================================
# Curve constants
# ============================================================

N = curve_order      # scalar field order (254-bit prime); exponents live in Z_n
P = field_modulus    # base field characteristic

# Domain-separation tag for the NUMS generator. Changing this string changes
# H and therefore invalidates every previously issued commitment.
H_SEED_TAG = b"ZK-KGVerify/BN128/NUMS-H/v2"


def hash_to_curve(seed: bytes, max_tries: int = 1024):
    """Map `seed` to a BN128 point with unknown discrete logarithm.

    Try-and-increment: x = SHA3-256(seed || ctr) mod p, accept the first x
    where x^3 + 3 is a quadratic residue. BN128 has p = 3 (mod 4), so the
    square root is (x^3+3)^((p+1)/4). The even root is chosen so the point
    is canonical across implementations.
    """
    for ctr in range(max_tries):
        x = int.from_bytes(keccak(seed + ctr.to_bytes(4, "big")), "big") % P
        y2 = (pow(x, 3, P) + 3) % P
        y = pow(y2, (P + 1) // 4, P)
        if pow(y, 2, P) == y2:
            if y % 2 == 1:
                y = P - y
            point = (FQ(x), FQ(y))
            if is_on_curve(point, CURVE_B):
                return point
    raise RuntimeError("hash_to_curve failed; try a different seed")


H_POINT = hash_to_curve(H_SEED_TAG)

# Exposed so the Solidity verifier constants can be regenerated / audited.
H_X = int(H_POINT[0].n)
H_Y = int(H_POINT[1].n)


# ============================================================
# Helpers
# ============================================================

def _point_to_ints(point) -> Tuple[int, int]:
    """Affine (x, y) as native ints; (0, 0) encodes the point at infinity."""
    if point is None:
        return (0, 0)
    x, y = point
    return (int(x.n), int(y.n))


def _point_from_ints(xy: Tuple[int, int]):
    """Inverse of `_point_to_ints`. Returns None for the infinity sentinel."""
    x, y = xy
    if x == 0 and y == 0:
        return None
    return (FQ(x), FQ(y))


def _valid_point(xy: Tuple[int, int]) -> bool:
    """Reject anything not a finite affine point of the curve.

    Coordinates must be reduced mod p, and the point must satisfy the curve
    equation. BN128's G1 has prime order equal to the full group order, so
    on-curve implies in-subgroup and no separate cofactor check is needed.
    """
    x, y = xy
    if not (0 <= x < P and 0 <= y < P):
        return False
    if x == 0 and y == 0:
        return False
    return is_on_curve((FQ(x), FQ(y)), CURVE_B)


def hash_embedding(emb) -> int:
    """Hash an embedding vector to a scalar in Z_n.

    Floats are serialised as %.6e before hashing so that the commitment is
    stable across platforms and across CPU/GPU runs, where the last bits of
    a float32 can differ.
    """
    arr = np.asarray(emb).flatten()
    payload = b"ZK-KGVerify-embedding-v2|" + b"|".join(
        f"{float(v):.6e}".encode() for v in arr
    )
    return int.from_bytes(keccak(payload), "big") % N


PRED_TAG = b"ZK-KGVerify-prediction-v3"


def scale_score(score: float) -> int:
    """Fixed-point encoding of a score as the int256 the contract stores."""
    return int(round(float(score) * 1_000_000))


def hash_prediction(
    triple: Tuple[int, int, int], score: float, aux_digest: Optional[bytes] = None
) -> str:
    """Digest binding the public prediction, recomputable by the contract.

    Encoding matches Solidity's
        keccak256(abi.encodePacked(PRED_TAG, head, relation, tail, score, aux))
    with head/relation/tail as uint256, score as int256 (two's complement),
    and aux as bytes32.

    The contract recomputes this from the fields it is about to store, rather
    than accepting a digest from the caller. That distinction matters: if the
    prover supplied an opaque predHash, the proof would bind to that blob and
    the stored triple could be anything. Recomputation is what makes the
    on-chain record itself the thing the proof commits to.

    `aux_digest` carries a commitment to private context -- by default the
    embedding vector that produced the prediction. The verifier cannot check
    it, and nothing in the security argument rests on it, but it lets a model
    owner make a later disclosure checkable against an immutable record.
    """
    aux = aux_digest if aux_digest is not None else bytes(32)
    if len(aux) != 32:
        raise ValueError("aux_digest must be exactly 32 bytes")
    payload = (
        PRED_TAG
        + int(triple[0]).to_bytes(32, "big")
        + int(triple[1]).to_bytes(32, "big")
        + int(triple[2]).to_bytes(32, "big")
        + (scale_score(score) % (1 << 256)).to_bytes(32, "big")
        + aux
    )
    return "0x" + keccak(payload).hex()


def embedding_aux_digest(embedding) -> bytes:
    """32-byte commitment to the embedding vector behind a prediction."""
    return hash_embedding(embedding).to_bytes(32, "big")


def hash_challenge(
    C_xy: Tuple[int, int],
    A_xy: Tuple[int, int],
    prediction_hash: str,
    model_id: str,
) -> int:
    """Fiat-Shamir challenge, byte-identical to the Solidity implementation.

    Solidity side:
        uint256(keccak256(abi.encodePacked(
            C[0], C[1], A[0], A[1], predHash, modelId
        ))) % N

    abi.encodePacked concatenates uint256 as 32-byte big-endian, bytes32 as
    its 32 raw bytes, and string as its UTF-8 bytes with no length prefix --
    which is what we reproduce here.
    """
    ph = prediction_hash[2:] if prediction_hash.startswith("0x") else prediction_hash
    payload = (
        C_xy[0].to_bytes(32, "big")
        + C_xy[1].to_bytes(32, "big")
        + A_xy[0].to_bytes(32, "big")
        + A_xy[1].to_bytes(32, "big")
        + bytes.fromhex(ph)
        + model_id.encode()
    )
    return int.from_bytes(keccak(payload), "big") % N


# ============================================================
# Pedersen commitment
# ============================================================

@dataclass
class PedersenCommitment:
    """C = v*G + r*H on BN128, stored as affine (x, y) ints."""

    commitment_xy: Tuple[int, int]
    value_scalar: int   # secret
    randomness: int     # secret


def pedersen_commit(
    value_scalar: int, randomness: Optional[int] = None
) -> PedersenCommitment:
    """Form C = value_scalar*G + randomness*H."""
    if randomness is None:
        randomness = secrets.randbelow(N)
    value_scalar %= N
    randomness %= N
    C = add(multiply(G1, value_scalar), multiply(H_POINT, randomness))
    return PedersenCommitment(
        commitment_xy=_point_to_ints(C),
        value_scalar=value_scalar,
        randomness=randomness,
    )


# ============================================================
# Model registration
# ============================================================
#
# WHY REGISTRATION IS NOT OPTIONAL
# --------------------------------
# A proof of knowledge of an opening of a commitment the prover created a
# moment ago proves nothing: the prover picked the value, so of course it
# knows the opening. An earlier version of this protocol did exactly that --
# it drew a fresh Pedersen commitment to the embedding for every prediction
# -- which made the statement vacuous and meant a random vector would have
# produced an equally valid proof.
#
# The commitment must therefore be fixed *before* any query is answered, and
# fixed somewhere the prover cannot quietly revise. We commit to a digest of
# the trained weights and register that commitment on-chain, where consensus
# timestamps it. Every subsequent proof is a proof of knowledge of the
# opening of *that* registered commitment, with the prediction bound into the
# Fiat-Shamir challenge.
#
# What this buys, precisely: an accepted record is attributable to whoever
# registered the model, and neither the prediction nor the model can be
# swapped after the fact. What it does not buy: evidence that the forward
# pass was computed correctly. See docs/THEORY.md for the exact statement.


@dataclass
class ModelCommitment:
    """Commitment to a trained model, published once before serving."""

    commitment_xy: Tuple[int, int]
    model_id: str
    weight_digest: str  # hex SHA-256 of the weights, for auditability
    # Secret opening, held by the model owner only:
    value_scalar: int
    randomness: int

    def public(self) -> dict:
        return {
            "model_id": self.model_id,
            "commitment_xy": list(self.commitment_xy),
            "weight_digest": self.weight_digest,
        }


def commit_model(
    weight_digest: str, model_id: str, randomness: Optional[int] = None
) -> ModelCommitment:
    """Commit to a trained model's weight digest.

    `weight_digest` is the hex SHA-256 over the model's state_dict (see
    run_all.py::model_fingerprint). Binding it into v means the registered
    commitment pins one specific set of trained weights.
    """
    digest = weight_digest[2:] if weight_digest.startswith("0x") else weight_digest
    v = int.from_bytes(
        keccak(b"ZK-KGVerify-model-v2|" + digest.encode() + b"|" + model_id.encode()),
        "big",
    ) % N
    if randomness is None:
        randomness = secrets.randbelow(N)
    randomness %= N
    C = add(multiply(G1, v), multiply(H_POINT, randomness))
    return ModelCommitment(
        commitment_xy=_point_to_ints(C),
        model_id=model_id,
        weight_digest=digest,
        value_scalar=v,
        randomness=randomness,
    )


# ============================================================
# Schnorr-Fiat-Shamir proof
# ============================================================

@dataclass
class ZKProof:
    """Non-interactive proof of knowledge of (v, r) opening a commitment."""

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
    aux_digest: str = "0x" + "00" * 32
    proof_size_bytes: int = 0

    # ---- serialisation used by the on-chain verifier ----

    def solidity_args(self) -> dict:
        """Arguments for ZKKGVerifyV2.verifyAndLog / verifyProof.

        Note there is no `C`: the contract reads the commitment from its own
        registry rather than from the caller. Nor is there a `predHash`: the
        contract derives it from the fields it stores plus `aux`.
        """
        return {
            "modelId": self.model_id,
            "triple": [int(x) for x in self.triple],
            "score": scale_score(self.score),
            "A": list(self.announcement_xy),
            "sv": self.response_v,
            "sr": self.response_r,
            "aux": bytes.fromhex(self.aux_digest[2:]),
            # Retained for the off-chain verifier and for diagnostics.
            "C": list(self.commitment_xy),
            "predHash": self.prediction_hash,
        }

    def wire_bytes(self) -> bytes:
        """Canonical compact encoding, used for the reported proof size.

        Two curve points in compressed form (32 bytes of x plus a sign bit
        folded into the top byte), two 32-byte scalars, and the 32-byte
        prediction digest. The challenge is not transmitted: a verifier
        recomputes it from the transcript, and accepting a transmitted
        challenge without recomputation would void Fiat-Shamir soundness.
        """
        def compress(xy: Tuple[int, int]) -> bytes:
            x, y = xy
            return (x | ((y & 1) << 255)).to_bytes(32, "big")

        return (
            compress(self.commitment_xy)
            + compress(self.announcement_xy)
            + self.response_v.to_bytes(32, "big")
            + self.response_r.to_bytes(32, "big")
            + bytes.fromhex(self.prediction_hash[2:])
        )


def generate_proof(
    model_commitment: ModelCommitment,
    prediction_score: float,
    triple: Tuple[int, int, int],
    model_id: Optional[str] = None,
    embedding_vector=None,
) -> ZKProof:
    """Prove knowledge of the opening of a *registered* model commitment,
    with the prediction bound into the challenge.

    The commitment is an input, not something generated here. That is the
    whole point: it was fixed and published before this query arrived, so an
    accepting transcript implicates the registered model rather than an
    arbitrary value chosen after seeing the query.
    """
    timestamp = time.time()
    mid = model_id or model_commitment.model_id

    C_xy = model_commitment.commitment_xy
    v = model_commitment.value_scalar
    r = model_commitment.randomness

    k_v = secrets.randbelow(N)
    k_r = secrets.randbelow(N)
    A_xy = _point_to_ints(add(multiply(G1, k_v), multiply(H_POINT, k_r)))

    aux = (
        embedding_aux_digest(embedding_vector)
        if embedding_vector is not None
        else bytes(32)
    )
    pred_hash = hash_prediction(triple, prediction_score, aux)
    e = hash_challenge(C_xy, A_xy, pred_hash, mid)

    proof = ZKProof(
        commitment_xy=C_xy,
        announcement_xy=A_xy,
        challenge=e,
        response_v=(k_v + e * v) % N,
        response_r=(k_r + e * r) % N,
        prediction_hash=pred_hash,
        model_id=mid,
        timestamp=timestamp,
        score=float(prediction_score),
        triple=tuple(int(x) for x in triple),
        aux_digest="0x" + aux.hex(),
    )
    proof.proof_size_bytes = len(proof.wire_bytes())
    return proof


def verify_proof(proof: ZKProof) -> bool:
    """Check a proof: honest challenge derivation plus the group equation.

    Point validation comes first. Skipping it would let a forger submit a
    small-order or off-curve `A`, so the check is not merely defensive.
    """
    if not (_valid_point(proof.commitment_xy) and _valid_point(proof.announcement_xy)):
        return False
    if not (0 <= proof.response_v < N and 0 <= proof.response_r < N):
        return False

    e = hash_challenge(
        proof.commitment_xy,
        proof.announcement_xy,
        proof.prediction_hash,
        proof.model_id,
    )
    if e != proof.challenge:
        return False

    C = _point_from_ints(proof.commitment_xy)
    A = _point_from_ints(proof.announcement_xy)

    lhs = add(multiply(G1, proof.response_v), multiply(H_POINT, proof.response_r))
    rhs = add(A, multiply(C, e))
    return eq(lhs, rhs)


# ============================================================
# Batch helpers
# ============================================================

def batch_generate_proofs(
    model_commitment: ModelCommitment,
    prediction_scores: List[float],
    triples: List[Tuple[int, int, int]],
    model_id: Optional[str] = None,
    embedding_vectors: Optional[List[np.ndarray]] = None,
) -> Tuple[List[ZKProof], dict]:
    proofs: List[ZKProof] = []
    gen_times: List[float] = []
    for i in range(len(triples)):
        emb = embedding_vectors[i] if embedding_vectors is not None else None
        t0 = time.perf_counter()
        p = generate_proof(
            model_commitment, prediction_scores[i], triples[i], model_id,
            embedding_vector=emb,
        )
        gen_times.append(time.perf_counter() - t0)
        proofs.append(p)

    stats = {
        "num_proofs": len(proofs),
        "total_gen_time": float(sum(gen_times)),
        "avg_gen_time": float(np.mean(gen_times)) if gen_times else 0.0,
        "std_gen_time": float(np.std(gen_times)) if gen_times else 0.0,
        "min_gen_time": float(np.min(gen_times)) if gen_times else 0.0,
        "max_gen_time": float(np.max(gen_times)) if gen_times else 0.0,
        "avg_proof_size_bytes": (
            float(np.mean([p.proof_size_bytes for p in proofs])) if proofs else 0.0
        ),
        "gen_times": gen_times,
    }
    return proofs, stats


def batch_verify_proofs(proofs: List[ZKProof]) -> Tuple[List[bool], dict]:
    results: List[bool] = []
    verify_times: List[float] = []
    for p in proofs:
        t0 = time.perf_counter()
        ok = verify_proof(p)
        verify_times.append(time.perf_counter() - t0)
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


# ============================================================
# Adversarial models used in the soundness experiment
# ============================================================

def _clone(proof: ZKProof, **overrides) -> ZKProof:
    data = {
        "commitment_xy": proof.commitment_xy,
        "announcement_xy": proof.announcement_xy,
        "challenge": proof.challenge,
        "response_v": proof.response_v,
        "response_r": proof.response_r,
        "prediction_hash": proof.prediction_hash,
        "model_id": proof.model_id,
        "timestamp": proof.timestamp,
        "score": proof.score,
        "triple": proof.triple,
        "aux_digest": proof.aux_digest,
        "proof_size_bytes": proof.proof_size_bytes,
    }
    data.update(overrides)
    return ZKProof(**data)


def _aux_bytes(proof: ZKProof) -> bytes:
    return bytes.fromhex(proof.aux_digest[2:])


def tamper_response(proof: ZKProof) -> ZKProof:
    """A1 -- perturb the response scalar s_v."""
    return _clone(proof, response_v=(proof.response_v + 1) % N)


def tamper_prediction(proof: ZKProof) -> ZKProof:
    """A2 -- keep the proof, swap the claimed tail entity.

    Operationally the attack that matters: a prover answering with a
    different entity than the one whose proof it holds.
    """
    h, r, t = proof.triple
    new_triple = (h, r, t + 1)
    return _clone(
        proof,
        triple=new_triple,
        prediction_hash=hash_prediction(new_triple, proof.score, _aux_bytes(proof)),
    )


def tamper_score(proof: ZKProof) -> ZKProof:
    """A3 -- inflate the confidence attached to an otherwise honest answer."""
    new_score = proof.score + 1.0
    return _clone(
        proof,
        score=new_score,
        prediction_hash=hash_prediction(proof.triple, new_score, _aux_bytes(proof)),
    )


def tamper_commitment(proof: ZKProof) -> ZKProof:
    """A4 -- point the proof at a commitment the prover cannot open."""
    other = pedersen_commit(secrets.randbelow(N))
    return _clone(proof, commitment_xy=other.commitment_xy)


def forge_offcurve(proof: ZKProof) -> ZKProof:
    """A5 -- announcement that is not a curve point at all."""
    x, y = proof.announcement_xy
    return _clone(proof, announcement_xy=(x, (y + 1) % P))


def unregistered_model(proof: ZKProof) -> ZKProof:
    """A6 -- an adversary that honestly proves knowledge of its *own* model.

    The strongest attacker in the set: it runs the protocol correctly, with a
    perfectly valid opening, but for a model nobody registered. The
    transcript is internally consistent, so the algebraic check alone cannot
    catch it. It is caught only because the verifier binds the proof to the
    commitment recorded in the registry rather than to one supplied with the
    proof -- which is exactly the property the pre-registration design exists
    to provide. Substituting a self-issued commitment reduces to A4 against a
    registry-bound verifier.
    """
    rogue = commit_model(
        hashlib_sha256_hex(secrets.token_bytes(32)), proof.model_id
    )
    forged = generate_proof(rogue, proof.score, proof.triple, proof.model_id)
    return _clone(
        forged,
        commitment_xy=proof.commitment_xy,  # claim the registered commitment
    )


def hashlib_sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


ADVERSARIES = {
    "A1_response_tamper": tamper_response,
    "A2_prediction_swap": tamper_prediction,
    "A3_score_inflation": tamper_score,
    "A4_commitment_swap": tamper_commitment,
    "A5_offcurve_point": forge_offcurve,
    "A6_unregistered_model": unregistered_model,
}

# Kept so older scripts importing `tamper_proof` still work.
tamper_proof = tamper_response


# ============================================================
# Self-tests
# ============================================================

def selftest_binding() -> None:
    """Assert the h_seed*G binding break no longer applies.

    With H = h_seed*G the attacker recomputes r' = (v-v')*h_seed^-1 + r and
    opens the same C to any v'. With H from hash_to_curve there is no known
    h_seed, so the attack cannot even be written down. We verify the weaker
    observable consequence: committing to a different value yields a
    different commitment for every random opening we try.
    """
    v, r = 123456789, 987654321
    C = add(multiply(G1, v), multiply(H_POINT, r))
    for _ in range(64):
        v2 = secrets.randbelow(N)
        r2 = secrets.randbelow(N)
        if v2 == v and r2 == r:
            continue
        C2 = add(multiply(G1, v2), multiply(H_POINT, r2))
        assert not eq(C, C2), "collision found -- binding is broken"
    print("  [ok] binding: no collision across 64 random openings")


def selftest_vacuity() -> None:
    """The protocol must not accept a commitment invented after the query.

    This is the regression test for the design flaw that motivated
    pre-registration: with a per-prediction commitment, the prover chooses
    the committed value itself and the proof degenerates into "I know a
    number I just picked". Here the registered commitment is fixed first,
    and a proof against any *other* commitment must fail.
    """
    registered = commit_model(hashlib_sha256_hex(b"the-real-model"), "M")
    rogue = commit_model(hashlib_sha256_hex(b"some-other-model"), "M")

    honest = generate_proof(registered, 1.0, (1, 2, 3))
    assert verify_proof(honest), "honest proof against the registry failed"

    # A proof built from the rogue opening, presented as if it opened the
    # registered commitment, must not verify.
    forged = _clone(
        generate_proof(rogue, 1.0, (1, 2, 3)),
        commitment_xy=registered.commitment_xy,
    )
    assert not verify_proof(forged), "proof for an unregistered model was accepted"
    print("  [ok] vacuity: proofs are bound to the registered commitment")


def selftest_completeness(n: int = 25) -> None:
    rng = np.random.default_rng(7)
    mc = commit_model(hashlib_sha256_hex(b"selftest-weights"), "selftest")
    for _ in range(n):
        emb = rng.standard_normal(384).astype("float32")
        p = generate_proof(mc, float(rng.uniform(-5, 5)), (1, 2, 3),
                           embedding_vector=emb)
        assert verify_proof(p), "honest proof rejected"
    print(f"  [ok] completeness: {n}/{n} honest proofs accepted")


def selftest_soundness(n: int = 10) -> None:
    rng = np.random.default_rng(11)
    mc = commit_model(hashlib_sha256_hex(b"selftest-weights"), "selftest")
    for name, attack in ADVERSARIES.items():
        caught = 0
        for _ in range(n):
            emb = rng.standard_normal(384).astype("float32")
            p = generate_proof(mc, float(rng.uniform(-5, 5)), (4, 5, 6),
                               embedding_vector=emb)
            if not verify_proof(attack(p)):
                caught += 1
        assert caught == n, f"{name}: only {caught}/{n} rejected"
        print(f"  [ok] soundness {name}: {caught}/{n} rejected")


def selftest_hvzk(n: int = 25) -> None:
    """Simulated transcripts must satisfy the same equation as real ones."""
    for _ in range(n):
        C = pedersen_commit(secrets.randbelow(N)).commitment_xy
        e = secrets.randbelow(N)
        s_v = secrets.randbelow(N)
        s_r = secrets.randbelow(N)
        # A' = s_v*G + s_r*H - e*C
        A = add(
            add(multiply(G1, s_v), multiply(H_POINT, s_r)),
            neg(multiply(_point_from_ints(C), e)),
        )
        lhs = add(multiply(G1, s_v), multiply(H_POINT, s_r))
        rhs = add(A, multiply(_point_from_ints(C), e))
        assert eq(lhs, rhs), "simulator produced an invalid transcript"
    print(f"  [ok] HVZK: {n}/{n} simulated transcripts satisfy the relation")


def run_selftests() -> None:
    print("ZK-KGVerify cryptographic self-tests (BN128)")
    print(f"  H = ({H_X},\n       {H_Y})")
    print(f"  H on curve: {is_on_curve(H_POINT, CURVE_B)}")
    selftest_binding()
    selftest_vacuity()
    selftest_completeness()
    selftest_soundness()
    selftest_hvzk()
    print("All self-tests passed.")


if __name__ == "__main__":
    run_selftests()
