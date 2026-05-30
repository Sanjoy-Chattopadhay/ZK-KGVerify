# ZK-KGVerify

**Privacy-Preserving Verification of Knowledge Graph Reasoning using Zero-Knowledge Proofs and Blockchain**

Companion code for the *Information Systems Frontiers* submission *"ZK-KGVerify: Privacy-Preserving Verification Infrastructure for Knowledge Graph Reasoning in Intelligent Information Systems"* by Sanjoy Chattopadhyay and Jaydeep Howlader (NIT Durgapur).

---

## What this repository proves

ZK-KGVerify combines three independently-verifiable layers into one end-to-end pipeline:

1. **KG Reasoning Layer** — TransE, RotatE, and CompGCN trained on FB15k-237.
2. **ZKP Verification Layer** — Pedersen commitments and Schnorr-Fiat-Shamir proofs on the **real BN128 elliptic curve** (via `py_ecc`), the same curve used by Ethereum's ZK precompiles.
3. **Blockchain Audit Layer** — `ZKKGVerify.sol` deployed on **Ethereum Sepolia** at [`0xdE79F732C74701D168dfdc912a9e42FC8958e56A`](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A) (**source verified** ✓, compiler `v0.8.20+commit.a1b79de6`, optimizer 200 runs), exercised with 30 real on-chain verification records. The contract has 31 confirmed transactions on Sepolia: one contract creation and 30 `logVerification` calls.

Everything is reproducible from a Colab T4 + a Sepolia-funded wallet.

---

## On-chain artefacts (live, public)

> *Every claim about gas cost or on-chain verifiability in the paper can be checked at these URLs by anyone with a browser. No login or wallet required to read.*

| Artefact | Value | Sepolia Etherscan |
|---|---|---|
| Contract name | `ZKKGVerify` (verified ✓) | [Contract page](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A) |
| Contract address | `0xdE79F732C74701D168dfdc912a9e42FC8958e56A` | [↗](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A) |
| Deployer | `0xA2c65C1F28A20bF1acbdFF54c4DFEea669183b5B` | [↗](https://sepolia.etherscan.io/address/0xA2c65C1F28A20bF1acbdFF54c4DFEea669183b5B) |
| Creation tx | `0xef0027...91e2` (block 10,952,286, 30 May 2026) | [↗](https://sepolia.etherscan.io/tx/0xef0027065c000a3cde353c99c3c131e8d07a4297e312aebaa0e76db8347791e2) |
| Deployment gas | 545,765 (21.83% of limit) | — |
| Deployment fee | 0.00195505 ETH @ 3.58 gwei | — |
| Solidity compiler | `v0.8.20+commit.a1b79de6`, optimizer 200 runs | [Source tab](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A#code) |
| Total transactions | 31 (deploy + 30 `logVerification`) | — |
| Sample `logVerification` tx | `0x7af5a2...7688` — 236,813 gas, `_modelId="CompGCN"` | [↗](https://sepolia.etherscan.io/tx/0x7af5a2810ca7693f693115826108529dea632cf1d2cd393455d1af11f4677688) |
| `VerificationLogged` events | 30 emitted | [Events tab](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A#events) |
| Interactive query | `recordCount` returns 30; `getRecord(n)` returns the stored record | [Read Contract tab](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A#readContract) |

### Verify the on-chain claims yourself (no setup, ~60 s)

1. Open the [contract page](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A) and confirm **31 transactions** and the green **Verified** badge.
2. Open the [Read Contract tab](https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A#readContract). Click `recordCount` → it returns `30`. Click `getRecord(0)` → see the first verification record, with its model id, triple, score, commitment hash, verified flag, proof hash, and block timestamp — all immutable.
3. Open the [sample logVerification tx](https://sepolia.etherscan.io/tx/0x7af5a2810ca7693f693115826108529dea632cf1d2cd393455d1af11f4677688) and click *"Decode Input Data"*. You will see the eight named parameters the contract was called with, and the `VerificationLogged` event in the **Logs** section.

That is the entire trust chain for the blockchain layer of the paper, observable to anyone.

### Screenshot — deployed contract
*(shows: balance, all 30 + 1 transactions, contract creation date)*

<!-- SCREENSHOT_1: contract_overview.png -->
![Sepolia contract overview](docs/screenshots/contract_overview.png)

🔗 **Verify live:** <https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A>

### Screenshot — deployment transaction
*(shows: 545,765 gas, contract creation, deployer address)*

<!-- SCREENSHOT_2: deployment_tx.png -->
![Deployment transaction](docs/screenshots/deployment_tx.png)

🔗 **Verify live:** <https://sepolia.etherscan.io/tx/0xef0027065c000a3cde353c99c3c131e8d07a4297e312aebaa0e76db8347791e2>

### Screenshot — representative `logVerification` call
*(shows: decoded input fields — model_id, head/relation/tail, commitment hash, verified=true, proofHash; emitted event)*

<!-- SCREENSHOT_3: log_verification_tx.png -->
![Sample logVerification transaction](docs/screenshots/log_verification_tx.png)

🔗 **Verify live:** <https://sepolia.etherscan.io/tx/0x7af5a2810ca7693f693115826108529dea632cf1d2cd393455d1af11f4677688>

### Screenshot — on-chain audit log
*(shows: all 30 `VerificationLogged` events in chronological order)*

<!-- SCREENSHOT_4: events_tab.png -->
![VerificationLogged events](docs/screenshots/events_tab.png)

🔗 **Verify live:** <https://sepolia.etherscan.io/address/0xdE79F732C74701D168dfdc912a9e42FC8958e56A#events>

### Every single one of the 30 `logVerification` transactions

<details>
<summary><b>Click to expand — direct Etherscan link for each on-chain record</b></summary>

Each link below opens the individual transaction on Sepolia Etherscan. Click *"Decode Input Data"* on any of them to see the eight named parameters (`_modelId`, `_head`, `_relation`, `_tail`, `_score`, `_commitment`, `_verified`, `_proofHash`) that were written immutably to the chain.

**Contract creation**
- [`0xef0027...91e2`](https://sepolia.etherscan.io/tx/0xef0027065c000a3cde353c99c3c131e8d07a4297e312aebaa0e76db8347791e2) — deploys `ZKKGVerify.sol`, 545,765 gas

**Run 2 (15 calls, attached to existing contract — most recent first)**
1. [`0xd9a2cf7b...41e8`](https://sepolia.etherscan.io/tx/0xd9a2cf7b0f1ef1f89ef6765760a91fa31f50e420bca3acdfdc9f59f3178b41e8)
2. [`0x5db83bae...4124`](https://sepolia.etherscan.io/tx/0x5db83baeb2862e6e327d8fd9846370da65220f7095c5ec11cd377f32e8f84124)
3. [`0xeb3c1f81...7699`](https://sepolia.etherscan.io/tx/0xeb3c1f81adaf153a24b12d9659d1478d32e3d69718f60ed6483cec56a0977699)
4. [`0x6bfd1491...05a7`](https://sepolia.etherscan.io/tx/0x6bfd1491a438971a47e3266a11695d3953a0aacc424d37494a3b55b51b4b05a7)
5. [`0x0800e6f2...8297`](https://sepolia.etherscan.io/tx/0x0800e6f2177f3986d5688017e4a406ca17dffd061ea35f9d63afe6b567b58297)
6. [`0xa51e13ea...2f87`](https://sepolia.etherscan.io/tx/0xa51e13ea361af2e884866ddedc68eb3536ae933603278f9b99d3790111c02f87)
7. [`0x1429c557...3fb1`](https://sepolia.etherscan.io/tx/0x1429c5575d4ff1bad3fb979528308452c1551abb7964a3724a8d454ab3383fb1)
8. [`0x452b4a30...1fdd`](https://sepolia.etherscan.io/tx/0x452b4a30d9e10417540eab1a910ea4624f35a26094ee86fc4d72c2b1f2721fdd)
9. [`0x57bf5ea8...fe83`](https://sepolia.etherscan.io/tx/0x57bf5ea810a3e7f60bcf2c73cf6aa8b2bdc66fbed28f3e8ea39487a47735fe83)
10. [`0x4064cbdc...f207`](https://sepolia.etherscan.io/tx/0x4064cbdc5888807cdb35e0a1740ce7129e24a0ffd95ab03bc8ddcf7598e0f207)
11. [`0xf89d778d...250b`](https://sepolia.etherscan.io/tx/0xf89d778d70fcf622d9741ffc1a9c24fbb3fd0999932bad7e54c57b3dff8b250b)
12. [`0x062c62b3...1e3f`](https://sepolia.etherscan.io/tx/0x062c62b359903a41fa109b62895d8010b53ca2c51a737a0d19d63f8d271b1e3f)
13. [`0x6f0af3c1...349b`](https://sepolia.etherscan.io/tx/0x6f0af3c1c2238fd3392354b76d8a47bd5964bfdacd8e1f6d362a8c411f6f349b)
14. [`0x7af5a281...7688`](https://sepolia.etherscan.io/tx/0x7af5a2810ca7693f693115826108529dea632cf1d2cd393455d1af11f4677688) — sample shown in Screenshot 3
15. [`0x7d1666ae...1500`](https://sepolia.etherscan.io/tx/0x7d1666aed603834be81bb8cd68001a015cd27cb0bdad94c49e415182af7b1500)

**Run 1 (15 calls, also visible from the contract Transactions tab — page 2)**
16. [`0x32539868...3033`](https://sepolia.etherscan.io/tx/0x32539868e4c44508cfa7c25eb11d82dd0bbc5ebdf9cba62314d8e7742e5a3033)
17. [`0xf89cdbf8...6c9d`](https://sepolia.etherscan.io/tx/0xf89cdbf80cadc99578a8e4b1d5ee4c34166b7c149e7f845c4c919f3d27936c9d)
18. [`0x1192cf6a...5634`](https://sepolia.etherscan.io/tx/0x1192cf6adcb7c7844c828afbd41e4be7f492a305ff9822921d87b72a64505634)
19. [`0xce179b7f...ad63`](https://sepolia.etherscan.io/tx/0xce179b7ff684ce83ce853eedee1fae03d29688f2908100be141d819aae77ad63)
20. [`0xab62e754...a38a`](https://sepolia.etherscan.io/tx/0xab62e754a5f91852ab56f7441dc15a17209d992b48d87c5c31db34a0ace0a38a)
21. [`0x66b7d152...2ec9`](https://sepolia.etherscan.io/tx/0x66b7d152bbb748cd024e3fcb26193762ea335d0fc97c292db3f7c8085f6f2ec9)
22. [`0x481b457c...4428`](https://sepolia.etherscan.io/tx/0x481b457c58674c95130dd2f62cec12d201fb4489c6eab64ecd87a4ff2c4d4428)
23. [`0x8dfc2a37...bce8`](https://sepolia.etherscan.io/tx/0x8dfc2a370aa9d7d0f2dea36aa8ed25eeeb95705193e19ab231d237302708bce8)
24. [`0x63c34505...221e`](https://sepolia.etherscan.io/tx/0x63c345050b6a38ca26fbcba0f30465e6e06d932e1d900c4a7fa588c0ba6b221e)
25. The remaining 6 (the oldest from Run 1, blocks 10,952,287–10,952,292) are visible on [page 2 of the contract's Transactions tab](https://sepolia.etherscan.io/txs?a=0xdE79F732C74701D168dfdc912a9e42FC8958e56A&p=2).

</details>

---

## Project structure

```
ZK-KGVerify/
├── run.py                          # End-to-end local pipeline (training → ZKP → blockchain → plots)
├── run_sepolia.py                  # Real Sepolia deploy + on-chain logging
├── requirements.txt
│
├── configs/config.py               # Hyperparameters, seed, early stopping
├── contracts/ZKKGVerify.sol        # Solidity audit contract
│
├── src/
│   ├── data_loader.py              # FB15k-237 loader
│   ├── models.py                   # TransE, RotatE, CompGCN, R-GCN
│   ├── trainer.py                  # Training loop + early stopping
│   ├── zkp_module.py               # BN128 Pedersen + Schnorr-Fiat-Shamir
│   ├── blockchain_module.py        # Local Python-blockchain simulation
│   ├── sepolia_module.py           # Real Sepolia deployment + tx logging
│   ├── pipeline.py                 # Orchestrator
│   └── visualization.py            # Figures + LaTeX tables
│
├── results/
│   ├── all_results.json            # Metrics, ZKP stats, simulated blockchain stats
│   ├── sepolia_log.json            # REAL Sepolia tx records with measured gas
│   ├── training_curves.png
│   ├── metrics_comparison.png
│   ├── zkp_overhead.png
│   ├── blockchain_stats.png
│   └── pipeline_timing.png
│
└── ZK_KGVerify_colab_v2.ipynb      # One-click Colab T4 reproduction
```

---

## Quick start

### Option A — One-click Colab T4 (recommended)

1. Open [`ZK_KGVerify_colab_v2.ipynb`](ZK_KGVerify_colab_v2.ipynb) in Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. Wall-clock: ~45–90 min.
4. Outputs are copied to your Google Drive.

### Option B — Local

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

### Option C — Real Sepolia deployment

You will need a throwaway Metamask account with ~0.05 Sepolia ETH (free from [sepoliafaucet.com](https://sepoliafaucet.com)) and an RPC URL (free tier of Alchemy/Infura).

```powershell
$env:SEPOLIA_RPC_URL = "https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY"
$env:SEPOLIA_PRIVATE_KEY = "0xYOUR_TESTNET_KEY"
python run_sepolia.py --num-tx 30
```

Deploys the contract, logs 30 proofs on chain, writes measured gas/latency/cost to `results/sepolia_log.json`. Takes ~8 minutes.

---

## Headline results

### Link prediction on FB15k-237 (filtered, 20,466 test triples)

| Model    | MRR | Hits@1 | Hits@3 | Hits@10 |
|----------|-----|--------|--------|---------|
| TransE   | 0.1694 | 0.0945 | 0.1808 | 0.3341 |
| RotatE   | 0.2547 | 0.1763 | 0.2802 | 0.4098 |
| **CompGCN** | **0.3910** | **0.2957** | **0.4316** | **0.5820** |

### Screenshot — end-to-end pipeline run (Colab T4)

*(shows: training summary, link-prediction metrics, BN128 ZKP "valid 1000/1000",
local-blockchain mining stats, and final pipeline timing breakdown)*

<!-- SCREENSHOT_5: pipeline_run.png -->
![End-to-end pipeline output on Colab T4](docs/screenshots/pipeline_run.png)

🔗 **Reproduce live:** [open `ZK_KGVerify_colab_v2.ipynb` in Colab](https://colab.research.google.com/github/Sanjoy-Chattopadhay/ZK-KGVerify/blob/v2-real-bn128/ZK_KGVerify_colab_v2.ipynb), set runtime to **T4 GPU**, Run All.

### BN128 ZKP overhead (1,000 proofs, measured on Colab T4)

| Metric | Value |
|---|---|
| Avg. proof generation time | 107.35 ms (sd 26.59) |
| Avg. proof verification time | 79.16 ms (sd 18.91) |
| Avg. proof size | 854 bytes |
| Verification success rate | 100% (1000/1000) |
| Tamper detection rate | 100% (100/100) |

> *These are pure-Python `py_ecc` numbers, designed for cross-platform reproducibility.
> A native bn254/mcl backend or the Ethereum BN128 precompile would cut both
> numbers by roughly 10–100×; on the user's local Windows machine outside Colab,
> the same code measures ~46 ms / ~34 ms per proof.*

### Real Sepolia on-chain costs (30 records)

| Metric | Value |
|---|---|
| Deployment gas | 545,765 |
| Deployment cost | 0.00196 Sepolia ETH |
| Avg. `logVerification` gas | 236,955 ± 180 |
| Avg. on-chain latency | 11.4 s |
| Total gas (30 calls) | 7,142,682 |
| Total cost (30 calls) | 0.0376 Sepolia ETH |

### Screenshot — Sepolia run completed

<!-- SCREENSHOT_6: sepolia_run.png -->
![Sepolia run terminal output](docs/screenshots/sepolia_run.png)

🔗 **Audit the resulting log:** [`results/sepolia_log.json`](results/sepolia_log.json) — every measured gas value, effective gas price, latency, and tx hash for the 15 calls in this run (the other 15 are in the earlier batch on the same contract).

---

## Reproducing the paper's numbers

All randomness is seeded via `RANDOM_SEED = 42` in `configs/config.py`, covering Python `random`, NumPy, PyTorch CPU, and CUDA. Re-running the pipeline on the same hardware should reproduce the metrics in the headline tables within floating-point noise.

Generated artefacts that match the paper figures and tables:

| Paper element | Generated file |
|---|---|
| Fig. 4 (training curves) | `results/training_curves.png` |
| Fig. 5 (metrics bar chart) | `results/metrics_comparison.png` |
| Fig. 6 (ZKP overhead distributions) | `results/zkp_overhead.png` |
| Fig. 7 (blockchain stats) | `results/blockchain_stats.png` |
| Fig. 8 (pipeline timing) | `results/pipeline_timing.png` |
| Tables 2–5 | `results/tables.tex` (LaTeX, paste-ready) |

---

## External references (verify every dependency)

Every third-party building block of this work is publicly inspectable at the URLs below — useful for any reviewer who wants to confirm we are not running custom or weakened crypto.

| Component | Authoritative source |
|---|---|
| FB15k-237 dataset | <https://www.microsoft.com/en-us/download/details.aspx?id=52312> · mirror: <https://github.com/villmow/datasets_knowledge_embedding/tree/master/FB15k-237> |
| BN128 elliptic curve (EIP-196 / EIP-197) | <https://eips.ethereum.org/EIPS/eip-196> |
| `py_ecc` reference implementation | <https://github.com/ethereum/py_ecc> (PyPI: <https://pypi.org/project/py-ecc/>) |
| Ethereum Sepolia testnet | <https://sepolia.dev/> · explorer: <https://sepolia.etherscan.io/> |
| Sepolia ETH faucets used | <https://sepoliafaucet.com> · <https://faucets.chain.link/sepolia> |
| Alchemy RPC (Sepolia) | <https://www.alchemy.com/sepolia> |
| Solidity compiler `solc 0.8.20` | <https://github.com/ethereum/solidity/releases/tag/v0.8.20> |
| TransE (Bordes et al., 2013) | <https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html> |
| RotatE (Sun et al., 2019) | <https://arxiv.org/abs/1902.10197> |
| CompGCN (Vashishth et al., 2020) | <https://arxiv.org/abs/1911.03082> |
| Pedersen commitments (Pedersen, 1991) | <https://link.springer.com/chapter/10.1007/3-540-46766-1_9> |
| Schnorr identification (Schnorr, 1991) | <https://link.springer.com/article/10.1007/BF00196725> |
| Fiat-Shamir transform (Fiat & Shamir, 1986) | <https://link.springer.com/chapter/10.1007/3-540-47721-7_12> |

---

## Citation

A journal version of this work is under review. While that is in progress,
please cite the manuscript as:

```bibtex
@misc{chattopadhyay2026zkkgverify,
  title        = {{ZK-KGVerify}: Privacy-Preserving Verification Infrastructure
                  for Knowledge Graph Reasoning in Intelligent Information Systems},
  author       = {Chattopadhyay, Sanjoy and Howlader, Jaydeep},
  year         = {2026},
  howpublished = {Manuscript, National Institute of Technology Durgapur},
  url          = {https://github.com/Sanjoy-Chattopadhay/ZK-KGVerify}
}
```

This BibTeX entry will be replaced with the formal journal record (with
DOI) once the paper is accepted and published.

---

## License

MIT
