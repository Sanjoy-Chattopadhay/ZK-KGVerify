# How to run this — FGCS submission run

Everything runs on **this laptop**. The RTX 3050 is enough; Colab is no longer
part of the path.

Every step writes a log to `results/logs/` and a full snapshot to
`results/runs/<timestamp>__<label>/`, so nothing is lost by a later re-run.

---

## Why this changed

The earlier plan sent the GPU work to Colab, and the run there never
finished — it crawled, then the runtime disconnected. Colab was not the
problem. Three things in this code were:

1. **The data loader drew negatives one triple at a time.** For each of the
   272,115 training triples it made a separate `torch.randint` call, and then
   `default_collate` restacked 1,024 single-element tensors into every batch.
   That is ~150 ms of Python per batch against ~2 ms of arithmetic. Sampling
   the whole batch at once — same distribution, same reshuffling — cut TransE
   from 2.3 h to 0.7 h per 200 epochs.
2. **The GCN models re-ran message passing for every scored triple.**
   `CompGCN.predict` calls `encode`, which propagates over all 272,115 edges;
   evaluation calls `predict` once per test triple, so a full test set ran the
   graph convolution 20,466 times, and the proof stage another 2,000. The
   encoding cannot change while the weights are frozen, so it is now computed
   once and cached, and dropped the moment training resumes.
3. **Evaluation scored one triple per GPU launch.** Those launches are
   latency-bound — the dispatch costs far more than the arithmetic. Ranking
   now goes 32 triples at a time, with one device sync per block instead of
   one per triple.

All three are pure speed: the ranks, the metrics and the proofs are computed
the same way, from the same arithmetic, in the same order. An A/B against the
original code gave metrics identical to within 1e-9 on all three models.

The fourth thing was the profile itself. `--profile paper` trains **R-GCN**,
which no table, figure or claim in the manuscript uses — it appears once, in
the limitations section, as a model the method has *not* been tried on. It
costs 440 s/epoch on FB15k-237 against CompGCN's 37 s, so it alone was ~24 h
of the run. Use `--profile paper-local`, which is the same protocol at the
same scale without it.

---

## Setup — once

A CUDA build of torch, in its own environment on `D:` so it cannot disturb the
CPU-only torch in your system Python:

```bash
python -m venv "D:\Study\PSIT\Journal Paper\ISF _August\.venv"
```

```bash
"D:\Study\PSIT\Journal Paper\ISF _August\.venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
"D:\Study\PSIT\Journal Paper\ISF _August\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Confirm the card is visible and get a timing estimate for each profile:

```bash
"D:\Study\PSIT\Journal Paper\ISF _August\.venv\Scripts\python.exe" tools/preflight.py
```

Everything below uses that interpreter. Set it as the workspace interpreter in
VS Code (`Ctrl+Shift+P` → Python: Select Interpreter) and plain `python` will
resolve to it in the integrated terminal.

---

## Step 1 — self-tests and on-chain gas (~2 min, no key, no money)

```bash
python run_step.py --label 01-selftest-evm --stages selftest,local-evm --profile paper-local
```

Deploys both contracts to an in-memory EVM, runs all six adversaries against
the verifier, and measures the gas gap between the unverified V1 and the
chain-verified V2. Expect `ALL CHECKS PASSED` and a *cost of trustlessness*
near **+4.1%** — that is the paper's headline number.

## Step 2 — train, prove, report (~4 h, no key)

```bash
python run_step.py --label 02-train-zkp-report --stages train,zkp,report --profile paper-local
```

Trains TransE, RotatE and CompGCN on FB15k-237 and WN18RR, evaluates on the
**full** test sets, generates 1,000 proofs from real model predictions, runs
the off-chain soundness battery, and regenerates every figure, table and
macro.

Checkpoints are written as each model finishes and reused on the next run, so
this is safe to interrupt — restarting picks up at the model that was in
flight. Watch it with:

```bash
tail -f results/logs/*02-train*.log
```

## Step 3 — Sepolia deployment (~10 min, needs your testnet key)

**Read this before running.**

- Use a **throwaway MetaMask account** holding only free Sepolia testnet ETH.
  Never an account with real funds on any network.
- Fund it free at <https://sepoliafaucet.com> or
  <https://www.alchemy.com/faucets/ethereum-sepolia>. About 0.05 test ETH is
  ample for 30 records.
- Set the key as an environment variable rather than typing it on the command
  line, so it never lands in your shell history. It is never printed, never
  logged, and never written to disk.

In PowerShell, for this session only:

```bash
$env:SEPOLIA_PRIVATE_KEY = "0xYOUR_THROWAWAY_TESTNET_KEY"
```

Then:

```bash
python run_step.py --label 03-sepolia --stages train,zkp,sepolia,report --profile paper-local
```

`train` resumes instantly from the Step 2 checkpoints; `zkp` regenerates the
proof objects the Sepolia stage needs in memory, then it deploys and submits.

When it finishes, clear the key:

```bash
Remove-Item Env:SEPOLIA_PRIVATE_KEY
```

## Step 4 — verify the paper matches the results

```bash
python tools/check_paper_numbers.py --tex ../manuscript_fgcs/zkkgverify-fgcs.tex
```

Re-reads `results/paper_macros.tex` against the manuscript and fails if any
quoted number has drifted from what the code produced. **Expect this to fail
the first time**: the manuscript still quotes the Colab T4 timings
(≈107 ms generation, ≈79 ms verification), and this machine is slower at
pure-Python BN128 — the preflight measured ≈129 ms and ≈183 ms. Those
sentences, and the hardware paragraph in the experimental setup, need
updating to this machine.

---

## If something breaks

Send me `results/logs/<the newest log>`. Every step is captured in full,
including the traceback, and the log now flushes per line — so an empty tail
means genuinely stuck, not merely quiet.

Re-running any step is safe. Snapshots are additive: `results/runs/` keeps one
directory per invocation, so two runs can always be diffed against each other.

## What each artefact is for

| File | Feeds |
|---|---|
| `results/paper_macros.tex` | every number in the manuscript, via `\input` |
| `results/table_onchain.tex` | Table: V1 vs V2 gas — the headline result |
| `results/table_soundness.tex` | Table: adversaries A1–A6 |
| `results/table_link_prediction.tex` | Table: MRR / Hits@k |
| `results/table_datasets.tex` | Table: dataset statistics |
| `results/gas_comparison.png` | Figure: cost of trustlessness |
| `results/zkp_overhead.png` | Figure: proof generation / verification cost |
| `results/sepolia_log.json` | every transaction hash, for the reproducibility appendix |
