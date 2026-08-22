# Evidence for the results reported in the paper

Everything below is either reproducible from this repository with one command,
or readable directly from the Ethereum Sepolia testnet without our involvement.
Nothing here requires trusting the authors.

Run: 22 August 2026. Machine: one NVIDIA GeForce RTX 3050 Laptop GPU (6 GB),
Intel Core i5-13450HX, 24 GB RAM, Windows 11, PyTorch 2.11.0+cu128.

---

## The live deployment

| | |
|---|---|
| Contract | [`0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3`](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3) |
| Version | `ZKKGVerifyV2.sol` — verification runs **inside** the EVM |
| Deployment gas | 1,325,277 |
| Model registration | 138,684 gas, single-shot |
| Records | 30, every one accepted only after the chain checked the proof |
| Gas per record | 246,952 ± 8 (steady state); 281,162 first call |
| On-chain verification | 56,568 gas |
| Median latency | 9.2 s |
| Total cost | 0.01927 testnet ETH |

The deploy gas is the quickest way to tell the two contracts apart: V1 (which
does not verify) costs 545,765; V2 costs 1,325,277.

### Independently exported from Etherscan

[`docs/etherscan_export_sepolia.csv`](docs/etherscan_export_sepolia.csv) is
the transaction list for this contract, exported from Etherscan rather than
produced by our code. It is worth reading as a cross-check, because nothing in
it passed through our tooling.

| | |
|---|---|
| Transactions | 32 — 30 `verifyAndLog`, 1 `registerModel`, 1 contract creation |
| Status | **Success on all 32.** No reverts, no retries |
| Blocks | 11,542,020 → 11,542,051 — 32 consecutive blocks, one transaction each |
| Window | 2026-08-22 09:03:00 → 09:09:48 UTC (6 min 48 s) |
| Total fees | 0.02298617 ETH (testnet) |

Etherscan shows the method column as raw selectors because the source is not
verified on its site. They decode against `ZKKGVerifyV2.sol` as:

| Selector | Function |
|---|---|
| `0xb43e985b` | `verifyAndLog(string,uint256[3],int256,uint256[2],uint256,uint256,bytes32)` |
| `0x20a954a8` | `registerModel(string,uint256[2],bytes32)` |
| `0x60806040` | contract creation (EVM initialisation prefix, not a selector) |

Reproduce that mapping yourself — it is the first four bytes of the Keccak-256
hash of each signature:

```bash
python -c "from eth_utils import keccak; print('0x'+keccak(text='verifyAndLog(string,uint256[3],int256,uint256[2],uint256,uint256,bytes32)')[:4].hex())"
```

That 30 records landed in 30 consecutive blocks with zero failures is the
practical claim behind the latency figure: every proof the prover submitted
was accepted by the EVM on first presentation.

### Verify it yourself, without us

```bash
python tools/fetch_onchain_log.py --address 0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3
```

Rebuilds the complete audit log from the chain alone.

```bash
python tools/check_onchain_parity.py --address 0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3 --model-id CompGCN
```

Confirms the deployed bytecode derives the same second generator `H` and the
same prediction digests as the Python prover in this repository, and that it
rejects forged proofs.

---

## What was measured

### 1. Link prediction — 3 models × 2 datasets, full test sets, no subsampling

| Dataset | Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| FB15k-237 | TransE | 0.1726 | 0.0928 | 0.1910 | 0.3432 |
| | RotatE | 0.2529 | 0.1753 | 0.2794 | 0.4061 |
| | **CompGCN** | **0.3939** | **0.2965** | **0.4366** | **0.5868** |
| WN18RR | TransE | 0.1786 | 0.0590 | 0.2581 | 0.3947 |
| | **RotatE** | **0.2609** | **0.1857** | 0.2888 | 0.4056 |
| | CompGCN | 0.2360 | 0.0600 | **0.3660** | **0.5530** |

All 20,466 FB15k-237 and 3,134 WN18RR test triples, filtered protocol.

### 2. Proof cost — 1,000 proofs on real CompGCN predictions

| | |
|---|---|
| Generation | 23.05 ± 0.72 ms |
| Verification | 34.86 ± 0.98 ms |
| Proof size | 160 bytes |
| Accepted | 1000 / 1000 |

Pure-Python BN128 (`py_ecc`), single core, machine otherwise idle. These are
an upper bound: a native backend or the Ethereum precompiles would be roughly
an order of magnitude faster.

### 3. Soundness — six adversary classes × 1,000 trials each

| Adversary | Rejected |
|---|---|
| A1 response scalar perturbed | 1000/1000 |
| A2 answered entity swapped after proving | 1000/1000 |
| A3 reported confidence inflated | 1000/1000 |
| A4 commitment replaced with an unopenable one | 1000/1000 |
| A5 announcement forced off the curve | 1000/1000 |
| A6 honest proof for an unregistered model | 1000/1000 |

Replayed against the deployed contract as well: an invalid proof reverts and
leaves no record.

### 4. The cost of trustlessness — in-memory EVM, V1 vs V2

| Operation | V1 (unverified) | V2 (chain-verified) |
|---|---|---|
| Contract deployment | 545,765 | 1,325,277 |
| Model registration | — | 138,696 |
| Proof verification (call only) | — | 68,869 |
| Record, first call | 271,361 | 281,162 |
| Record, steady state | 237,031 | 246,828 |

**+9,797 gas, +4.13% per record.** Measured on `eth-tester`/`py-evm`, which
runs the same bytecode as Sepolia without fee-market noise.

---

## Reproducing the whole thing

```bash
python -m venv .venv && .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python run_step.py --label repro --stages selftest,local-evm,train,zkp,report --profile paper-local
```

About 50 minutes on the machine described above, no key and no money needed.
Only the `sepolia` stage requires a funded testnet account.

Every invocation writes a full log to `results/logs/` and snapshots every
artefact into `results/runs/<timestamp>__<label>/`, so any number in the paper
can be traced back to the run that produced it.

## Scope of the guarantee

The proof establishes that whoever produced a record knew the opening of the
commitment registered under that model identifier, and bound this specific
triple and score to it. It does **not** certify that the score is what
evaluating the committed model on that input would produce. See
§"What the Proof Does and Does Not Establish" in the paper — this is a
deliberate trade that buys a ~1000× reduction in proof cost against
circuit-based zkML.
