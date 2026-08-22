#!/usr/bin/env python
"""
ZK-KGVerify -- one command runs the whole paper.

    python run_all.py --private-key 0xYOUR_SEPOLIA_TESTNET_KEY

That is the only input required. An RPC endpoint is chosen automatically from
a list of public Sepolia nodes, the contract is compiled and deployed, models
are trained (or reloaded from checkpoints), proofs are generated from real
model embeddings, every proof is verified inside the EVM, and all numbers
quoted in the manuscript are regenerated into results/.

SAFETY
------
Use a throwaway account holding only free Sepolia testnet ETH. The key is
never printed, never logged, and never written to disk. Get testnet funds at
https://sepoliafaucet.com or https://www.alchemy.com/faucets/ethereum-sepolia.

PROFILES
--------
  --profile smoke   ~10 min, CPU. Proves the whole path works end to end.
  --profile fast    ~1-2 h, CPU. Reduced epochs; real numbers, not paper-grade.
  --profile paper   Full protocol incl. R-GCN. Needs a GPU; ~24 h even on one.
  --profile paper-local
                    Everything the manuscript reports, at full scale, minus
                    R-GCN (which the manuscript does not report). ~4 h on an
                    RTX 3050; this is the local configuration.

STAGES
------
Run everything, or pick up where a previous run stopped:

  python run_all.py --private-key 0x... --stages selftest,local-evm
  python run_all.py --private-key 0x... --stages sepolia,report

  selftest   cryptographic self-tests (binding, completeness, soundness, HVZK)
  local-evm  deploy + verify on an in-memory EVM; no key or network needed
  train      train and evaluate the KG embedding models
  zkp        proof generation/verification benchmark on real embeddings
  sepolia    deploy V2 to Sepolia and submit chain-verified records
  report     figures, LaTeX tables, and paper macros

Everything except `sepolia` runs without a private key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

# ----------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------

PROFILES = {
    "smoke": {
        "datasets": ["FB15k-237"],
        "models": ["TransE"],
        "epochs": 3,
        "eval_max": 500,
        "num_proofs": 25,
        "num_onchain": 3,
        "embedding_dim": 64,
    },
    "fast": {
        "datasets": ["FB15k-237", "WN18RR"],
        "models": ["TransE", "RotatE"],
        "epochs": 30,
        "eval_max": 3000,
        "num_proofs": 200,
        "num_onchain": 15,
        "embedding_dim": 128,
    },
    # Publication-grade numbers on a CPU-only machine. The expensive part is
    # training, and training quality is not what this paper claims -- the
    # claim is about verification cost. So the embedding models get enough
    # epochs to land in the normal published range for FB15k-237 (early
    # stopping usually ends it sooner), while the parts the paper actually
    # measures -- proof cost, on-chain gas, soundness -- run at full scale.
    "paper-cpu": {
        "datasets": ["FB15k-237", "WN18RR"],
        "models": ["TransE", "RotatE", "CompGCN"],
        "epochs": 100,
        "eval_max": 5000,
        "num_proofs": 1000,
        "num_onchain": 30,
        "embedding_dim": 128,
    },
    "paper": {
        "datasets": ["FB15k-237", "WN18RR"],
        "models": ["TransE", "RotatE", "CompGCN", "RGCN"],
        "epochs": 200,
        "eval_max": None,
        "num_proofs": 1000,
        "num_onchain": 30,
        "embedding_dim": 128,
    },
    # What the manuscript actually reports, sized for one consumer GPU.
    #
    # 'paper' trains R-GCN, which no table, figure or claim in the paper
    # uses -- it appears once, in the limitations section, as a model the
    # method has NOT been tried on. It also costs 440 s/epoch on FB15k-237
    # against CompGCN's 37 s, so it alone is ~24 h of the run and was the
    # single biggest reason the full profile never finished. Everything the
    # paper measures -- link prediction for the three reported models, the
    # 1,000-proof benchmark, the soundness battery, on-chain gas -- runs
    # here at full scale, on the full test sets.
    "paper-local": {
        "datasets": ["FB15k-237", "WN18RR"],
        "models": ["TransE", "RotatE", "CompGCN"],
        "epochs": 200,
        "eval_max": None,
        "num_proofs": 1000,
        "num_onchain": 30,
        "embedding_dim": 128,
    },
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def model_fingerprint(model) -> str:
    """Stable digest of a model's weights.

    Recorded alongside every on-chain batch so a reader can tell which
    trained model produced the committed embeddings. Without it, "these
    records came from CompGCN" would be an unfalsifiable claim.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def environment_info() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - not a git checkout, or git missing
        info["git_commit"] = None
    return info


# ----------------------------------------------------------------------
# Stage: self-tests
# ----------------------------------------------------------------------

def stage_selftest(state: dict) -> None:
    banner("STAGE 1/6  Cryptographic self-tests")
    from src.zkp_module import run_selftests

    t0 = time.time()
    run_selftests()
    state["timings"]["selftest"] = time.time() - t0


# ----------------------------------------------------------------------
# Stage: local EVM
# ----------------------------------------------------------------------

def stage_local_evm(state: dict, cfg: dict) -> None:
    banner("STAGE 2/6  On-chain verifier on an in-memory EVM")
    t0 = time.time()
    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "tools" / "test_onchain_verifier.py"),
            "--num", str(min(12, max(4, cfg["num_onchain"]))),
            "--json", str(RESULTS / "onchain_verifier_test.json"),
        ],
        cwd=str(ROOT),
    )
    if rc != 0:
        raise SystemExit(
            "Local EVM verification failed. Fix this before spending testnet ETH."
        )
    state["timings"]["local_evm"] = time.time() - t0
    state["onchain_verifier_test"] = load_json(RESULTS / "onchain_verifier_test.json")


# ----------------------------------------------------------------------
# Stage: train
# ----------------------------------------------------------------------

def stage_train(state: dict, cfg: dict, resume: bool) -> dict:
    banner("STAGE 3/6  Train and evaluate KG embedding models")
    import torch

    import configs.config as C
    from src.data_loader import KGDataset, get_data_loaders
    from src.models import get_model
    from src.trainer import evaluate_model, train_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cpu" and cfg["epochs"] >= 200:
        print(
            "  NOTE: the 'paper' profile on CPU will take many hours.\n"
            "        Run it on a GPU (Colab T4 reproduces the published numbers)."
        )

    # Apply profile to the shared config object the trainer reads.
    C.NUM_EPOCHS = cfg["epochs"]
    C.EMBEDDING_DIM = cfg["embedding_dim"]
    C.EVAL_MAX = cfg["eval_max"]

    CHECKPOINTS.mkdir(exist_ok=True)
    all_results: dict = {}
    trained: dict = {}
    datasets: dict = {}

    for ds_name in cfg["datasets"]:
        print(f"\n  --- dataset: {ds_name} ---")
        ds = KGDataset(ds_name, data_dir=str(ROOT / "data"))
        datasets[ds_name] = ds.stats()
        loader = get_data_loaders(
            ds, batch_size=C.BATCH_SIZE, negative_sample_size=C.NEGATIVE_SAMPLE_SIZE,
            device=device,
        )
        all_results[ds_name] = {}

        for model_name in cfg["models"]:
            tag = f"{ds_name}__{model_name}__d{cfg['embedding_dim']}__e{cfg['epochs']}"
            ckpt = CHECKPOINTS / f"{tag}.pt"
            meta = CHECKPOINTS / f"{tag}.json"

            model = get_model(
                model_name,
                num_entities=ds.num_entities,
                num_relations=ds.num_relations,
                embedding_dim=cfg["embedding_dim"],
                margin=C.MARGIN,
            )
            if hasattr(model, "set_graph"):
                model.set_graph(
                    ds.train_triples[:, [0, 2]].t().to(device),
                    ds.train_triples[:, 1].to(device),
                )
            model = model.to(device)

            if resume and ckpt.exists() and meta.exists():
                try:
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                    saved = json.loads(meta.read_text(encoding="utf-8"))
                    print(f"  [resume] {tag}  (MRR={saved['metrics']['MRR']:.4f})")
                    all_results[ds_name][model_name] = saved
                    trained[(ds_name, model_name)] = model
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"  [resume] checkpoint unusable ({exc}); retraining")

            print(f"\n  === {ds_name} / {model_name} ===")
            history = train_model(model, loader, ds, C, device=device)
            metrics = evaluate_model(model, ds, C, device=device, max_eval=cfg["eval_max"])

            entry = {
                "metrics": metrics,
                "history": history,
                "fingerprint": model_fingerprint(model),
                "config": {
                    "epochs": cfg["epochs"],
                    "embedding_dim": cfg["embedding_dim"],
                    "eval_max": cfg["eval_max"],
                },
            }
            torch.save(model.state_dict(), ckpt)
            meta.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
            all_results[ds_name][model_name] = entry
            trained[(ds_name, model_name)] = model

    state["datasets"] = datasets
    state["link_prediction"] = all_results
    save_json(RESULTS / "link_prediction.json",
              {"datasets": datasets, "results": all_results})
    return trained


# ----------------------------------------------------------------------
# Stage: ZKP benchmark
# ----------------------------------------------------------------------

def stage_zkp(state: dict, cfg: dict, trained: dict) -> list:
    banner("STAGE 4/6  ZKP benchmark on real model embeddings")
    import torch

    from src.data_loader import KGDataset
    from src.zkp_module import (
        ADVERSARIES,
        batch_generate_proofs,
        batch_verify_proofs,
        commit_model,
        verify_proof,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pick the best model by MRR on the primary dataset. Proofs must come
    # from a real trained model -- committing to random vectors would make
    # the on-chain records unrelated to any actual prediction.
    primary = cfg["datasets"][0]
    lp = state["link_prediction"][primary]
    best_name = max(lp, key=lambda k: lp[k]["metrics"]["MRR"])
    model = trained[(primary, best_name)]
    model.eval()
    fingerprint = lp[best_name]["fingerprint"]
    print(f"  model: {best_name} on {primary} "
          f"(MRR={lp[best_name]['metrics']['MRR']:.4f})")
    print(f"  weights fingerprint: {fingerprint[:16]}...")

    # Commit to the trained weights once. Every proof below opens this same
    # commitment; it is what gets registered on-chain in the next stage.
    model_commitment = commit_model(fingerprint, best_name)
    state["model_commitment"] = model_commitment.public()
    print(f"  model commitment   : ({str(model_commitment.commitment_xy[0])[:20]}..., "
          f"{str(model_commitment.commitment_xy[1])[:20]}...)")

    ds = KGDataset(primary, data_dir=str(ROOT / "data"))
    n = min(cfg["num_proofs"], len(ds.test_triples))
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(ds.test_triples), generator=g)[:n]
    sample = ds.test_triples[idx]

    embeddings, scores, triples = [], [], []
    with torch.no_grad():
        for i in range(n):
            h, r, t = sample[i].tolist()
            hi = torch.tensor([h], device=device)
            ri = torch.tensor([r], device=device)
            ti = torch.tensor([t], device=device)
            embeddings.append(
                model.get_embedding_vector(hi, ri, ti).cpu().numpy().flatten()
            )
            scores.append(float(model.predict(hi, ri)[0, t].item()))
            triples.append((h, r, t))

    print(f"  generating {n} proofs ...")
    proofs, gen_stats = batch_generate_proofs(
        model_commitment, scores, triples, best_name, embedding_vectors=embeddings
    )
    state["_model_commitment_obj"] = model_commitment
    print(f"    gen  {gen_stats['avg_gen_time']*1000:7.2f} ms  "
          f"(sd {gen_stats['std_gen_time']*1000:.2f})")
    print(f"    size {gen_stats['avg_proof_size_bytes']:.0f} bytes")

    results, ver_stats = batch_verify_proofs(proofs)
    print(f"    vfy  {ver_stats['avg_verify_time']*1000:7.2f} ms  "
          f"(sd {ver_stats['std_verify_time']*1000:.2f})")
    print(f"    accepted {ver_stats['num_valid']}/{ver_stats['num_verified']}")

    # Soundness: every adversary against every proof.
    print("\n  soundness (off-chain verifier):")
    soundness = {}
    for name, attack in ADVERSARIES.items():
        rejected = sum(1 for p in proofs if not verify_proof(attack(p)))
        soundness[name] = {"trials": len(proofs), "rejected": rejected,
                           "rate": rejected / max(len(proofs), 1)}
        flag = "ok " if rejected == len(proofs) else "FAIL"
        print(f"    [{flag}] {name}: {rejected}/{len(proofs)}")

    zkp = {
        **gen_stats, **ver_stats,
        "model": best_name,
        "dataset": primary,
        "model_fingerprint": fingerprint,
        "model_commitment": model_commitment.public(),
        "embedding_source": "real trained model predictions (not synthetic)",
        "legitimate_verification_rate": ver_stats["verification_rate"],
        "soundness": soundness,
        "tamper_detection_rate": min(s["rate"] for s in soundness.values()),
    }
    state["zkp"] = zkp
    save_json(RESULTS / "zkp_benchmark.json", zkp)
    return proofs


# ----------------------------------------------------------------------
# Stage: Sepolia
# ----------------------------------------------------------------------

def stage_sepolia(state: dict, cfg: dict, proofs: list, args) -> None:
    banner("STAGE 5/6  Sepolia: deploy V2 and submit chain-verified records")
    from src.sepolia_module import SepoliaBackend

    sep = SepoliaBackend(
        rpc_url=args.rpc,
        private_key=args.private_key,
        version=args.contract_version,
    )

    if args.attach:
        sep.attach(args.attach)
    else:
        sep.deploy()

    sep.assert_generator_parity()

    # Register the model commitment before submitting anything. The
    # registration block is what timestamps the owner's commitment to a
    # specific set of weights, and proofs are refused until it exists.
    registration = {}
    mc = state.get("_model_commitment_obj")
    if mc is not None and args.contract_version == "v2":
        registration = sep.register_model(mc)

    if proofs:
        sep.assert_digest_parity(proofs[0])

    n = min(cfg["num_onchain"], len(proofs))
    print(f"\n  submitting {n} records "
          f"({'chain-verified' if args.contract_version == 'v2' else 'log-only'})\n")

    verify_gas = []
    for i in range(n):
        p = proofs[i]
        if args.contract_version == "v2":
            try:
                verify_gas.append(sep.estimate_verify_gas(p))
            except Exception as exc:  # noqa: BLE001
                print(f"    [warn] verify-gas estimate failed: {exc}")
        tx = sep.log_verification(p, verified=True)
        print(f"    [{i+1:>3}/{n}] gas={tx.gas_used:,}  block={tx.block_number:,}  "
              f"{tx.latency_s:5.1f}s  tx={tx.tx_hash[:12]}...")

    context = {
        "contract_version": args.contract_version,
        "proof_source": state.get("zkp", {}).get("embedding_source"),
        "model": state.get("zkp", {}).get("model"),
        "model_fingerprint": state.get("zkp", {}).get("model_fingerprint"),
        "model_commitment": state.get("model_commitment"),
        "registration": registration,
        "dataset": state.get("zkp", {}).get("dataset"),
        "verify_only_gas_estimates": verify_gas,
        "verify_only_gas_mean": float(np.mean(verify_gas)) if verify_gas else None,
    }
    sep.save_log(str(RESULTS / "sepolia_log.json"), extra=context)

    s = sep.summary()
    state["sepolia"] = {**s, "context": context}

    print("\n  " + "-" * 60)
    print(f"  contract        : {s['contract_address']}")
    print(f"  records         : {s['num_tx']}   all succeeded: {s['all_tx_succeeded']}")
    print(f"  first tx gas    : {s['first_tx_gas']:,}  (cold storage slots)")
    print(f"  steady-state gas: {s['steady_state_avg_gas']:,.0f} "
          f"+/- {s['steady_state_std_gas']:,.0f}")
    if context["verify_only_gas_mean"]:
        print(f"  verify-only gas : {context['verify_only_gas_mean']:,.0f}")
    print(f"  total cost      : {s['total_cost_eth']:.8f} ETH (testnet)")
    print(f"  median latency  : {s['median_latency_s']:.1f} s")
    print(f"  etherscan       : https://sepolia.etherscan.io/address/{s['contract_address']}")
    print("  " + "-" * 60)


# ----------------------------------------------------------------------
# Stage: report
# ----------------------------------------------------------------------

def stage_report(state: dict) -> None:
    banner("STAGE 6/6  Figures, tables, and paper macros")
    from src.reporting import build_all

    build_all(state, RESULTS)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

ALL_STAGES = ["selftest", "local-evm", "train", "zkp", "sepolia", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full ZK-KGVerify experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--private-key", default=os.environ.get("SEPOLIA_PRIVATE_KEY"),
                    help="Sepolia testnet key. Throwaway accounts only.")
    ap.add_argument("--rpc", default=os.environ.get("SEPOLIA_RPC_URL"),
                    help="Sepolia RPC URL. Omit to auto-select a public node.")
    ap.add_argument("--profile", choices=list(PROFILES), default="fast")
    ap.add_argument("--stages", default="all",
                    help=f"Comma-separated subset of: {','.join(ALL_STAGES)}")
    ap.add_argument("--contract-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--attach", default=None,
                    help="Reuse an already-deployed contract at this address.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing checkpoints and retrain.")
    ap.add_argument("--num-onchain", type=int, default=None,
                    help="Override the profile's on-chain record count.")
    # Checkpoint filenames encode dataset, model, dim and epochs. Overriding
    # the model or dataset list therefore reuses everything already trained at
    # the same epoch count, whereas switching profile changes the epoch count
    # and silently orphans those checkpoints.
    ap.add_argument("--models", default=None,
                    help="Comma-separated model subset, e.g. TransE,RotatE,CompGCN")
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated dataset subset, e.g. FB15k-237,WN18RR")
    args = ap.parse_args()

    stages = ALL_STAGES if args.stages == "all" else [
        s.strip() for s in args.stages.split(",") if s.strip()
    ]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stage(s): {sorted(unknown)}")

    cfg = dict(PROFILES[args.profile])
    if args.num_onchain is not None:
        cfg["num_onchain"] = args.num_onchain
    for field, raw in (("models", args.models), ("datasets", args.datasets)):
        if raw is None:
            continue
        picked = [x.strip() for x in raw.split(",") if x.strip()]
        unknown = set(picked) - set(cfg[field])
        if unknown:
            ap.error(f"--{field}: {sorted(unknown)} not in profile "
                     f"'{args.profile}' ({cfg[field]})")
        cfg[field] = picked

    if "sepolia" in stages and not args.private_key:
        ap.error(
            "The 'sepolia' stage needs --private-key (or SEPOLIA_PRIVATE_KEY).\n"
            "Run without it using: --stages selftest,local-evm,train,zkp,report"
        )

    banner("ZK-KGVerify -- full experimental pipeline")
    env = environment_info()
    print(f"  profile : {args.profile}  {cfg}")
    print(f"  stages  : {', '.join(stages)}")
    print(f"  device  : {env['device']}" + (f" ({env.get('gpu')})" if env.get("gpu") else ""))
    print(f"  commit  : {env['git_commit']}")

    RESULTS.mkdir(exist_ok=True)

    # Stages are commonly run across separate invocations -- train on a GPU
    # box, then submit to Sepolia from a laptop. Seed state from the last run
    # so `--stages report` can still see results the current process did not
    # produce, and so a partial rerun does not silently blank earlier work.
    state = {}
    previous = load_json(RESULTS / "all_results.json")
    if previous and set(stages) != set(ALL_STAGES):
        carried = [k for k in ("datasets", "link_prediction", "zkp", "sepolia",
                               "onchain_verifier_test", "model_commitment")
                   if k in previous]
        if carried:
            state.update({k: previous[k] for k in carried})
            print(f"  carried : {', '.join(carried)} (from the previous run)")
        state["timings"] = dict(previous.get("timings") or {})
        state["timings"].pop("total", None)

    state.update({
        "profile": args.profile,
        "config": cfg,
        "environment": env,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    state.setdefault("timings", {})

    trained: dict = {}
    proofs: list = []
    t_start = time.time()

    if "selftest" in stages:
        stage_selftest(state)
    if "local-evm" in stages:
        stage_local_evm(state, cfg)
    if "train" in stages:
        t0 = time.time()
        trained = stage_train(state, cfg, resume=not args.no_resume)
        state["timings"]["train"] = time.time() - t0
    if "zkp" in stages:
        if not trained:
            raise SystemExit("Stage 'zkp' needs 'train'; add it to --stages.")
        t0 = time.time()
        proofs = stage_zkp(state, cfg, trained)
        state["timings"]["zkp"] = time.time() - t0
    if "sepolia" in stages:
        if not proofs:
            raise SystemExit("Stage 'sepolia' needs 'zkp'; add it to --stages.")
        t0 = time.time()
        stage_sepolia(state, cfg, proofs, args)
        state["timings"]["sepolia"] = time.time() - t0
    if "report" in stages:
        stage_report(state)

    state["timings"]["total"] = time.time() - t_start
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # The ModelCommitment object carries the secret opening (v, r). Publishing
    # it would let anyone forge proofs for this model, so it is dropped before
    # anything touches disk. Only the public commitment point is persisted.
    state.pop("_model_commitment_obj", None)
    save_json(RESULTS / "all_results.json", state)

    banner("DONE")
    for k, v in state["timings"].items():
        print(f"  {k:<12} {v:>10.1f} s")
    print(f"\n  Results in {RESULTS}")
    print("  Paper macros: results/paper_macros.tex "
          "(\\input this to keep every number in sync)")


if __name__ == "__main__":
    main()
