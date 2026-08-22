# ZK-KGVerify — Theory Notes

Reference notes for the cryptographic construction: what it proves, what it
does not, and why each design choice is the way it is. Written to be readable
alongside `src/zkp_module.py` and `contracts/ZKKGVerifyV2.sol`, and to answer
the questions a reviewer will ask.

---

## 1. Setting

| Symbol | Meaning |
|---|---|
| $\mathbb{G}_1$ | BN128 (alt_bn128) group, $y^2 = x^3 + 3$ over $\mathbb{F}_p$ |
| $p$ | base field characteristic, 254 bits, $p \equiv 3 \pmod 4$ |
| $n$ | group order, prime, 254 bits; scalars live in $\mathbb{Z}_n$ |
| $G$ | standard generator, $(1, 2)$ |
| $H$ | second generator, `HashToCurve("ZK-KGVerify/BN128/NUMS-H/v2")` |
| $\theta$ | trained model parameters (secret) |
| $\mathsf{wd}$ | $\mathrm{SHA256}(\theta)$, the weight digest |
| $C_\theta$ | registered Pedersen commitment to the model |
| $\mathsf{H}(\cdot)$ | Keccak-256, reduced mod $n$ where a scalar is needed |

BN128 is chosen for one reason: Ethereum prices its group operations in
precompiles (`0x06` ECADD at 150 gas, `0x07` ECMUL at 6 000 gas). Any other
curve would have to be implemented in EVM opcodes, which is what makes
on-chain verification impractical elsewhere. The cost is a security margin of
roughly 110–128 bits rather than a clean 128 — see §7.

Because $n$ is prime and equals the full group order, **on-curve implies
in-subgroup**. No cofactor clearing is needed, and a single curve-equation
check suffices to validate an incoming point.

---

## 2. The protocol

### 2.1 Setup (public, once ever)

$H$ is derived by try-and-increment:

```
for ctr = 0, 1, 2, ...:
    x  = Keccak(seed ‖ ctr) mod p
    y² = x³ + 3
    if y² is a QR mod p:            # Euler: (y²)^((p-1)/2) == 1
        y = (y²)^((p+1)/4)          # valid because p ≡ 3 (mod 4)
        return (x, min(y, p-y))     # canonical: even root
```

Nobody knows $\log_G H$. §3 explains why this is not optional.

### 2.2 Registration (once per model, on-chain)

$$v = \mathsf{H}(\mathsf{wd} \,\|\, \mathsf{id}), \qquad
r \xleftarrow{\$} \mathbb{Z}_n, \qquad
C_\theta = vG + rH$$

The owner publishes $C_\theta$ via `registerModel`. The contract stores it
with the registering address and block number, and **refuses to overwrite**.
The opening $(v, r)$ never leaves the owner.

### 2.3 Proving (per prediction)

Given $(h, r, ?)$, compute $\hat t = \arg\max_{t'} f_\theta(h,r,t')$ and
$s = f_\theta(h,r,\hat t)$. Then:

$$
\begin{aligned}
\mathsf{aux} &= \mathsf{H}(\mathbf{e}_h \,\|\, \mathbf{e}_r \,\|\, \mathbf{e}_{\hat t}) \\
\mathsf{ph}  &= \mathsf{H}(\texttt{tag} \,\|\, h \,\|\, r \,\|\, \hat t \,\|\, s \,\|\, \mathsf{aux}) \\
A            &= k_v G + k_r H, \qquad k_v, k_r \xleftarrow{\$} \mathbb{Z}_n \\
e            &= \mathsf{H}(C_\theta \,\|\, A \,\|\, \mathsf{ph} \,\|\, \mathsf{id}) \bmod n \\
s_v          &= k_v + e v \bmod n, \qquad s_r = k_r + e r \bmod n
\end{aligned}
$$

Transmit $\pi = (A, s_v, s_r, \mathsf{aux})$ plus the public answer.

**What is deliberately not transmitted:** $C_\theta$, $e$, and $\mathsf{ph}$.
All three are re-derived by the verifier. A verifier that accepted any of them
from the prover would forfeit the guarantee — see §5, adversaries A3 and A4.

### 2.4 Verification

$$s_v G + s_r H \;\stackrel{?}{=}\; A + e\,C_\theta$$

with $C_\theta$ read from the registry, $\mathsf{ph}$ recomputed from the
fields being stored, and $e$ recomputed from the transcript.

---

## 3. Why $H$ must be hashed to the curve

**Proposition 1 (binding fails for $H = hG$ with public $h$).**
Let $H = hG$ with $h \in \mathbb{Z}_n$ public, and $C = vG + rH$. For any
target $v'$, setting

$$r' = (v - v')\,h^{-1} + r \pmod n$$

gives $\mathrm{Commit}(v', r') = C$.

*Proof.* $C = vG + rhG = (v + rh)G$, so $C$ depends on $(v,r)$ only through
$v + rh$. Then $v' + r'h = v' + (v - v') + rh = v + rh$. ∎

The attack costs one modular inverse. It is not asymptotic, not
quantum, not a reduction to some hard problem — it is arithmetic.

### Why this is easy to get wrong

$h = \mathsf{H}(\text{"some tag"})$ *looks* like a nothing-up-my-sleeve
derivation. The seed is public, the choice is not adversarial, and the
resulting $H$ is a perfectly good group element. The flaw is not in how $h$
was chosen; it is that $\log_G H$ **exists in closed form at all**.

An earlier version of this project used exactly this construction, and its
test suite passed. That is the instructive part:

| Property | Passes under $H = hG$? | Why |
|---|---|---|
| Completeness | yes | honest proofs verify regardless of how $H$ was built |
| HVZK | yes | simulator never touches $\log_G H$ |
| Tamper detection | yes | mutating $s_v$ still breaks the equation |
| **Binding** | **no** | the only property that probes $\log_G H$ |

None of the usual tests exercise binding, so the break is invisible until
someone writes the reopening attack down. `selftest_binding()` now does.

**Rule of thumb:** if you can write an expression for the discrete log of your
second generator, you do not have a Pedersen commitment.

---

## 4. Why the commitment must pre-date the query

**Proposition 2 (per-query commitments are vacuous).**
If the prover generates $C = \mathrm{Commit}(v, r)$ at query time and proves
knowledge of $(v,r)$, then it succeeds for *every* $v$ — including values
unrelated to any model.

*Proof.* The prover chooses $(v,r)$, hence knows the opening by construction;
completeness does the rest. ∎

This is the more consequential of the two flaws, because it is not a
cryptographic bug but a **modelling** one: every individual primitive is
correct, and the composed statement is still empty. A system built this way
generates proofs, verifies them, logs them, and reports 100% success, while
proving nothing whatsoever about the model.

Pre-registration removes the prover's freedom. Once $C_\theta$ is on-chain at
block $b$:

- every later transcript must open **that** commitment;
- an adversary without the opening faces Theorem 1;
- the owner cannot backdate, because consensus ordered the registration.

That last point is the one property for which a **blockchain** is genuinely
required rather than merely convenient. Attribution and integrity could be had
from a signed log. Ordering that neither party can rewrite could not.

`selftest_vacuity()` is the regression test.

---

## 5. Security properties

### Completeness (perfect)

$$s_v G + s_r H = (k_v + ev)G + (k_r + er)H = \underbrace{(k_vG + k_rH)}_{A} + e\underbrace{(vG + rH)}_{C_\theta}$$

Holds for every $(v,r)$ and every nonce choice.

### Soundness

**Theorem 1 (2-special soundness).** From two accepting transcripts
$(A, e, s_v, s_r)$ and $(A, e', s_v', s_r')$ with $e \neq e'$:

$$v = \frac{s_v - s_v'}{e - e'}, \qquad r = \frac{s_r - s_r'}{e - e'} \pmod n$$

*Proof.* Subtracting the verification equations gives
$(s_v - s_v')G + (s_r - s_r')H = (e-e')C_\theta$. $n$ is prime and
$e \neq e'$, so $(e-e')^{-1}$ exists. ∎

Under Fiat-Shamir this yields soundness in the ROM. Binding (which needs §3)
makes the extracted opening unique. A forger therefore either solves ECDLP on
BN128 or finds a Keccak-256 collision.

### Honest-verifier zero-knowledge

Simulator: draw $e', s_v', s_r' \xleftarrow{\$} \mathbb{Z}_n$, set
$A' = s_v'G + s_r'H - e'C_\theta$. The tuple satisfies the verification
equation and is distributed identically to a real transcript, because
$k_v, k_r$ are uniform and independent, hence so are $s_v, s_r$.

**Caveat worth stating plainly:** ZK covers the *transcript*. A long series of
published predictions leaks information about a model no matter how those
predictions are attested. The proof layer is not a defence against model
extraction via query access.

### Adversary classes

Implemented in `zkp_module.ADVERSARIES`, evaluated against both verifiers.

| ID | Attack | Off-chain | On-chain |
|---|---|---|---|
| A1 | perturb response scalar $s_v$ | rejected | rejected |
| A2 | swap the answered entity, keep the proof | rejected | rejected |
| A3 | inflate the reported score | rejected | rejected |
| A4 | substitute the commitment | rejected | **inert** — read from registry |
| A5 | off-curve announcement | rejected | rejected |
| A6 | honest proof, unregistered model | rejected | rejected |

**A4 is inert, not undetected.** The contract never reads a caller-supplied
commitment, so the substitution has no effect; the underlying honest proof
still verifies against the registered $C_\theta$. The same applies to
challenge substitution. Two of the values a naive verifier would have to
check are simply not attack surface on-chain — a consequence of deriving
rather than accepting them.

**A6 is the interesting one.** It runs the protocol perfectly, with a valid
opening, for a model nobody registered. Its transcript is internally
consistent, so the group equation *cannot* distinguish it. Only the registry
lookup does. A6 exists to demonstrate that §4 is doing real work.

**A5 is why the curve check is load-bearing.** Without validating $A$, a
forger can hand the precompiles a point outside the intended group. The
check on line 3 of the on-chain algorithm is not defensive boilerplate.

---

## 6. What this does and does not prove

### Guaranteed

> An accepted on-chain record establishes that the party that registered model
> $\mathsf{id}$ at block $b$ produced the answer $(h, r, \hat t, s)$, that the
> answer has not been altered since, and that neither the registration nor the
> record can subsequently be revised or denied.

Formally, under ECDLP on BN128 and Keccak-256 as a random oracle: an adversary
that has not registered a valid opening of $C_\theta$ cannot produce an
accepting transcript for any $(h, r, \hat t, s)$, except with negligible
probability.

### Not guaranteed

> That $\hat t = \arg\max_{t'} f_\theta(h, r, t')$.

An owner who registers honestly and then returns a deliberately wrong answer
produces a **valid proof of a wrong prediction**. What it cannot do is deny
having produced it, or attribute it to some other model.

This is the exact boundary of the guarantee, and the single most important
thing to state accurately. Detecting that class of misbehaviour requires
proving the forward pass — the province of circuit-based zkML, at seconds to
minutes per inference rather than tens of milliseconds.

The claim being made is therefore **accountable attribution and
non-repudiation**, not verified computation.

### Also not guaranteed

- **That $\mathsf{wd}$ corresponds to a genuinely trained model.** It commits
  to *some* weights; nothing forces those weights to be the product of
  training rather than fabrication. Binding registration to a training
  transcript would close this.
- **Anything about $\mathsf{aux}$.** The verifier cannot check it. It exists
  so a later voluntary disclosure can be checked against an immutable record —
  useful for audit, but no part of the security argument rests on it.

---

## 7. Practical notes

**Curve margin.** BN128 gives ~110–128 bits against known attacks (the
Kim–Barbulescu improvements to NFS in the embedding field pushed the estimate
down from the original 128). Adequate now; a deployment with a long
confidentiality horizon should plan migration to BLS12-381 once comparable
precompiles exist.

**Cost is constant in model size.** The proof is over a commitment, not over
the computation, so proving and verification cost do not depend on the number
of parameters or the size of the graph. This is precisely the trade against
circuit-based zkML, where cost scales with the network. It also means the
cryptographic measurements in the paper transfer unchanged to larger models.

**Nonce reuse is fatal.** Reusing $(k_v, k_r)$ across two different challenges
leaks the opening by Theorem 1 — the same failure that has repeatedly broken
ECDSA deployments. `secrets.randbelow` is used, not `random`.

**Fixed-point scores.** Scores are stored as `int256` scaled by $10^6$, and
`hash_prediction` binds the *scaled* value, so off-chain and on-chain agree
without float serialisation entering the digest.

**Float determinism.** Embeddings are serialised as `%.6e` before hashing so
$\mathsf{aux}$ is stable across CPU/GPU and platforms, where the last bits of
a float32 can differ.

---

## 8. Open directions

1. **Compose with a circuit proof of the scoring step only.** Proving
   $\hat t = \arg\max f_\theta(h,r,\cdot)$ over a *committed* embedding table
   is far smaller than proving a whole GNN, and would close the gap in §6
   without paying full zkML cost.
2. **Bind registration to a training transcript**, removing the assumption
   that $\mathsf{wd}$ came from real training.
3. **Batch verification.** Several Schnorr proofs against the same $C_\theta$
   can share a randomised linear combination, amortising the scalar
   multiplications — the dominant on-chain crypto cost.
4. **L2 deployment.** Storage dominates a record; on a rollup where calldata
   dominates instead, the economics change qualitatively.

---

## 9. Map to the code

| Concept | Location |
|---|---|
| $H$ derivation | `zkp_module.hash_to_curve` |
| Registration | `zkp_module.commit_model`, `ZKKGVerifyV2.registerModel` |
| Proving | `zkp_module.generate_proof` |
| Off-chain verification | `zkp_module.verify_proof` |
| On-chain verification | `ZKKGVerifyV2.verifyProof` / `verifyAndLog` |
| Prop. 1 regression test | `zkp_module.selftest_binding` |
| Prop. 2 regression test | `zkp_module.selftest_vacuity` |
| Adversaries | `zkp_module.ADVERSARIES` |
| Off/on-chain parity | `tools/test_onchain_verifier.py` |
| Parity vs live chain | `tools/check_onchain_parity.py` |
