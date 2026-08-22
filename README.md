# ZK-KGVerify

On-chain verification of knowledge-graph link predictions under model
confidentiality.

A model owner registers a Pedersen commitment to a digest of its trained
weights, once and in public. Every subsequent prediction carries a Schnorr
proof of knowledge of the opening of *that registered commitment*, with the
prediction bound into the Fiat-Shamir challenge. An Ethereum contract
re-derives the challenge and checks the group equation itself using the
alt_bn128 precompiles, and reverts on failure — so a stored record is an
attestation by consensus, not a claim by whoever wrote it.

**Cost of moving verification into consensus: about 4% more gas than logging
an unchecked boolean.**

---

## Quick start

```bash
pip install -r requirements.txt

# no key, no money — proves the whole path works
python run_all.py --profile smoke --stages selftest,local-evm

# the real thing
python run_all.py --private-key 0xYOUR_SEPOLIA_TESTNET_KEY
```

Use a throwaway account with free Sepolia testnet ETH only. See
[`HOW_TO_RUN.txt`](HOW_TO_RUN.txt) for profiles, stages, splitting training
across machines, and troubleshooting.

---

## What is and is not proved

**Guaranteed.** An accepted record establishes that the party that registered
model `id` at block *b* produced the answer `(h, r, t̂, s)`, that the answer
has not been altered since, and that neither the registration nor the record
can later be revised or denied — *accountable attribution and
non-repudiation*.

**Not guaranteed.** That `t̂ = argmax f_θ(h, r, ·)`. An owner who registers
honestly and then returns a deliberately wrong answer produces a valid proof
of a wrong prediction; what it cannot do is deny having produced it, or
attribute it to a different model. Proving the forward pass is the province
of circuit-based zkML, at seconds to minutes per inference rather than tens
of milliseconds.

Full statement, with proofs: [`docs/THEORY.md`](docs/THEORY.md).

---

## Two design points that are easy to get wrong

Both are documented as results because our own earlier version got both
wrong, and its test suite passed anyway.

**1. The second generator must be hashed to the curve.**
Deriving `H = h·G` for a public scalar `h` destroys binding. Since
`C = vG + rH = (v + rh)G`, any commitment reopens to any value `v'` by
setting `r' = (v − v')·h⁻¹ + r`. One modular inverse. Completeness, HVZK,
and tamper detection all still pass under the broken construction, because
none of them exercises binding. `selftest_binding()` now runs the attack.

**2. The commitment must pre-date the query.**
A proof of knowledge of the opening of a commitment created at query time is
vacuous — the prover picked the value, so of course it knows the opening.
Pre-registration, timestamped by consensus, is what turns "I know some
opening" into "I am the party that registered this model".
`selftest_vacuity()` is the regression test.

---

## Layout

```
run_all.py                  single entry point; all stages
contracts/
  ZKKGVerify.sol            V1: logs a caller-asserted boolean (baseline)
  ZKKGVerifyV2.sol          V2: registry + on-chain Schnorr verification
src/
  zkp_module.py             BN128 crypto, commitments, proofs, adversaries
  sepolia_module.py         deploy, register, submit, measure
  solidity.py               single compiler entry point
  models.py                 TransE, RotatE, CompGCN, R-GCN
  data_loader.py            FB15k-237, WN18RR
  trainer.py                training and filtered-ranking evaluation
  reporting.py              figures, tables, paper macros
tools/
  test_onchain_verifier.py  full contract test on an in-memory EVM
  check_onchain_parity.py   audit a deployed contract, read-only
  fetch_onchain_log.py      rebuild the audit log from the chain
  check_paper_numbers.py    fail if the manuscript drifted from results
  make_figures.py           schematic figures
docs/THEORY.md              construction, proofs, and boundaries
```

---

## Verifying without trusting us

Everything below is read-only and needs no key.

```bash
# rebuild the complete audit log straight from the chain
python tools/fetch_onchain_log.py --address 0xCONTRACT

# confirm the deployed contract matches this repo's cryptography,
# and that it rejects forged proofs
python tools/check_onchain_parity.py --address 0xCONTRACT --model-id CompGCN

# run the contract against every adversary class on a local EVM
python tools/test_onchain_verifier.py
```

---

## Adversary classes

| ID | Attack | Off-chain | On-chain |
|---|---|---|---|
| A1 | perturb response scalar `s_v` | rejected | rejected |
| A2 | swap the answered entity, keep the proof | rejected | rejected |
| A3 | inflate the reported score | rejected | rejected |
| A4 | substitute the commitment | rejected | **inert** |
| A5 | off-curve announcement | rejected | rejected |
| A6 | honest proof, unregistered model | rejected | rejected |

A4 is *inert* rather than undetected: the contract reads the commitment from
its registry and ignores anything the caller sends, so the substitution has
no effect. The same holds for challenge substitution. Two of the values a
naive verifier would have to check are simply not attack surface on-chain.

A6 is the interesting one — it runs the protocol correctly with a valid
opening, for a model nobody registered. Its transcript is internally
consistent, so the group equation cannot distinguish it; only the registry
lookup does.

---

## Reproducibility

Every number in the paper is emitted by this code as a LaTeX macro
(`results/paper_macros.tex`) and referenced by name in the manuscript. None
is transcribed by hand, and `tools/check_paper_numbers.py` fails the build if
a literal appears where a macro belongs, or if an address or transaction hash
is cited that the recorded results do not contain.

Seeds are pinned across Python, NumPy, PyTorch, and cuDNN. The in-memory EVM
reproduces the Sepolia deployment gas exactly (1,325,277 for V2), which is a
useful check that local measurements transfer to the public chain.

Measured numbers, the reproduction command, and the on-chain evidence are
collected in [`EVIDENCE.md`](EVIDENCE.md).

---

## Deployed instance

| | |
|---|---|
| Contract | [`0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3`](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3) |
| Version | `ZKKGVerifyV2` — verification runs inside the EVM |
| Network | Ethereum Sepolia (chainId 11155111) |
| Records | 30, each accepted only after the chain checked the proof |
| Gas per record | 246,952 ± 8 steady state; 56,568 of that is verification |
| Deployment | 1,325,277 gas |

Headline measurements from the run that produced the manuscript
(RTX 3050 laptop, 1,000 proofs, both benchmarks at full test-set size):

| | |
|---|---|
| Proof generation | 23.05 ± 0.72 ms |
| Proof verification | 34.86 ± 0.98 ms |
| Proof size | 160 bytes |
| Honest proofs accepted | 1000 / 1000 |
| Adversarial proofs rejected | 6000 / 6000 across six classes |
| Cost of on-chain verification | +4.13% gas per record |

---

## Citation

Chattopadhyay, S. and Howlader, J. *ZK-KGVerify: On-Chain Verification of
Knowledge-Graph Link Predictions under Model Confidentiality.* Manuscript
under submission.
