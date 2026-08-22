#!/usr/bin/env python
"""
Preflight: prove the machine can run the paper before committing hours to it.

    python tools/preflight.py

Checks every dependency the pipeline actually imports, makes sure the Solidity
compiler is present (py-solc-x downloads it on first use, which needs network
and is the most common first-run failure), confirms the datasets are on disk,
then times a handful of real training steps and converts that into a wall-clock
estimate for each profile.

Nothing here spends testnet ETH and nothing needs a private key.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "
_failures: list[str] = []
_warnings: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<34} {detail}")


def fail(label: str, detail: str) -> None:
    line(BAD, label, detail)
    _failures.append(f"{label}: {detail}")


def warn(label: str, detail: str) -> None:
    line(WARN, label, detail)
    _warnings.append(f"{label}: {detail}")


# ----------------------------------------------------------------------
# 1. interpreter and packages
# ----------------------------------------------------------------------

def check_packages() -> dict:
    print("\n--- interpreter and packages " + "-" * 43)
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    line(OK, "python", info["python"])

    if sys.version_info < (3, 9):
        fail("python version", "3.9+ required")

    required = [
        ("torch", "machine learning"),
        ("numpy", "arrays"),
        ("matplotlib", "figures"),
        ("tqdm", "progress bars"),
        ("py_ecc", "BN128 curve arithmetic"),
        ("web3", "Ethereum RPC"),
        ("eth_account", "transaction signing"),
        ("solcx", "Solidity compiler driver"),
        ("eth_tester", "in-memory EVM"),
    ]
    for mod, why in required:
        try:
            m = importlib.import_module(mod)
            line(OK, mod, f"{getattr(m, '__version__', '')}  ({why})")
        except Exception as exc:  # noqa: BLE001
            fail(mod, f"{type(exc).__name__} -- pip install -r requirements.txt")
    return info


# ----------------------------------------------------------------------
# 2. compute device
# ----------------------------------------------------------------------

def check_device() -> dict:
    print("\n--- compute " + "-" * 59)
    import torch

    dev = {"device": "cuda" if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        dev["gpu"] = props.name
        dev["vram_gb"] = round(props.total_memory / 1024**3, 1)
        line(OK, "device", f"cuda -- {props.name}  ({dev['vram_gb']} GB)")
    else:
        dev["gpu"] = None
        # A CPU-only wheel on a CUDA machine is the usual cause, and it is
        # worth calling out loudly: it is the difference between a 20-minute
        # run and a 3-hour one.
        cpu_wheel = "+cpu" in torch.__version__
        detail = "CPU only"
        if cpu_wheel:
            detail += " -- this is a CPU-only torch build; see README_RUN.md if you have an NVIDIA GPU"
        warn("device", detail)
    line(OK, "torch threads", str(torch.get_num_threads()))
    return dev


# ----------------------------------------------------------------------
# 3. solidity compiler
# ----------------------------------------------------------------------

def check_solc() -> dict:
    print("\n--- solidity compiler " + "-" * 49)
    out: dict = {}
    try:
        import solcx
    except Exception as exc:  # noqa: BLE001
        fail("py-solc-x", str(exc))
        return out

    try:
        installed = solcx.get_installed_solc_versions()
    except Exception:  # noqa: BLE001
        installed = []

    if installed:
        out["solc"] = str(max(installed))
        line(OK, "solc installed", out["solc"])
        return out

    line(WARN, "solc", "not installed -- downloading now (needs network, ~10 MB)")
    try:
        ver = solcx.install_solc("0.8.20")
        out["solc"] = str(ver)
        line(OK, "solc downloaded", out["solc"])
    except Exception as exc:  # noqa: BLE001
        fail(
            "solc download",
            f"{exc}\n         Offline? Download solc-windows-amd64-v0.8.20 manually and\n"
            "         place it in %USERPROFILE%\\.solcx\\",
        )
    return out


# ----------------------------------------------------------------------
# 4. datasets and contracts
# ----------------------------------------------------------------------

def check_data() -> dict:
    print("\n--- datasets and contracts " + "-" * 44)
    stats: dict = {}
    for ds in ("FB15k-237", "WN18RR"):
        d = ROOT / "data" / ds
        missing = [f for f in ("train.txt", "valid.txt", "test.txt")
                   if not (d / f).exists()]
        if missing:
            fail(f"data/{ds}", f"missing {missing}")
            continue
        n = sum(1 for _ in (d / "train.txt").open(encoding="utf-8"))
        stats[ds] = n
        line(OK, f"data/{ds}", f"{n:,} training triples")

    for sol in ("ZKKGVerify.sol", "ZKKGVerifyV2.sol"):
        p = ROOT / "contracts" / sol
        if p.exists():
            line(OK, f"contracts/{sol}", f"{p.stat().st_size:,} bytes")
        else:
            fail(f"contracts/{sol}", "missing")
    return stats


# ----------------------------------------------------------------------
# 5. proof timing
# ----------------------------------------------------------------------

def probe_proof_speed() -> dict:
    print("\n--- zero-knowledge proof speed " + "-" * 40)
    try:
        from src.zkp_module import commit_model, generate_proof, verify_proof
    except Exception as exc:  # noqa: BLE001
        fail("zkp_module import", str(exc))
        return {}

    mc = commit_model("0" * 64, "PreflightProbe")
    n = 5
    t0 = time.time()
    proofs = [generate_proof(mc, 0.5 + i / 100, (i, 0, i + 1), "PreflightProbe")
              for i in range(n)]
    gen_ms = (time.time() - t0) / n * 1000

    t0 = time.time()
    accepted = sum(1 for p in proofs if verify_proof(p))
    ver_ms = (time.time() - t0) / n * 1000

    line(OK, "proof generation", f"{gen_ms:.1f} ms each")
    line(OK, "proof verification", f"{ver_ms:.1f} ms each")
    if accepted != n:
        fail("proof verification", f"only {accepted}/{n} accepted")
    return {"gen_ms": gen_ms, "verify_ms": ver_ms}


# ----------------------------------------------------------------------
# 6. training speed  ->  profile estimates
# ----------------------------------------------------------------------

def probe_training_speed() -> dict:
    """Time a few real optimiser steps and extrapolate to whole profiles."""
    print("\n--- training speed probe " + "-" * 46)
    import torch

    import configs.config as C
    from src.data_loader import KGDataset, get_data_loaders
    from src.models import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = KGDataset("FB15k-237", data_dir=str(ROOT / "data"))
    loader = get_data_loaders(ds, batch_size=C.BATCH_SIZE,
                              negative_sample_size=C.NEGATIVE_SAMPLE_SIZE)
    model = get_model("TransE", num_entities=ds.num_entities,
                      num_relations=ds.num_relations,
                      embedding_dim=128, margin=C.MARGIN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    from src.trainer import MarginRankingLoss

    criterion = MarginRankingLoss(margin=C.MARGIN)

    # Same step the real trainer takes, so the timing transfers directly.
    n_steps, timed = 20, []
    it = iter(loader)
    for i in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t0 = time.time()
        head, relation, tail, neg = [b.to(device) for b in batch]
        pos_score, neg_score = model(head, relation, tail, neg)
        loss = criterion(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= 5:  # discard warm-up
            timed.append(time.time() - t0)

    per_step = sum(timed) / len(timed)
    steps_per_epoch = max(1, len(loader))
    epoch_s = per_step * steps_per_epoch
    line(OK, "batches per epoch", f"{steps_per_epoch:,}")
    line(OK, "seconds per epoch", f"{epoch_s:.1f} s  (TransE, dim 128)")
    return {"per_step_s": per_step, "steps_per_epoch": steps_per_epoch,
            "epoch_s": epoch_s}


def estimate_profiles(epoch_s: float, on_gpu: bool = False) -> None:
    """Wall-clock estimate per profile, accounting for early stopping."""
    from run_all import PROFILES

    recommended = "paper" if on_gpu else "paper-cpu"
    print("\n--- estimated training time per profile " + "-" * 31)
    print(f"  {'profile':<12} {'models x datasets':<20} {'epochs':>8} {'estimate':>14}")
    print("  " + "-" * 58)
    for name, cfg in PROFILES.items():
        runs = len(cfg["models"]) * len(cfg["datasets"])
        # Early stopping (patience 20, floor 30) usually bites well before the
        # cap on this data, so quote the realistic case, not the worst case.
        eff = min(cfg["epochs"], 70) if cfg["epochs"] > 70 else cfg["epochs"]
        secs = epoch_s * eff * runs
        # RotatE/CompGCN are heavier than the TransE we just timed.
        if any(m in cfg["models"] for m in ("RotatE", "CompGCN", "RGCN")):
            secs *= 1.8
        h, m = divmod(int(secs) // 60, 60)
        est = f"{h}h {m:02d}m" if h else f"{m} min"
        flag = "  <-- recommended" if name == recommended else ""
        print(f"  {name:<12} {runs:<20} {cfg['epochs']:>8} {est:>14}{flag}")
    print("\n  Estimates cover training only. Proofs, the EVM tests and reporting")
    print("  add a few minutes; the Sepolia stage is dominated by block times.")


# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  ZK-KGVerify preflight")
    print("=" * 72)

    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    report["env"] = check_packages()
    if _failures:
        print("\nFix the failures above before continuing.")
        sys.exit(1)

    report["device"] = check_device()
    report["solc"] = check_solc()
    report["data"] = check_data()
    report["zkp"] = probe_proof_speed()

    try:
        report["training"] = probe_training_speed()
        estimate_profiles(report["training"]["epoch_s"],
                          on_gpu=report["device"]["device"] == "cuda")
    except Exception as exc:  # noqa: BLE001
        warn("training probe", f"{type(exc).__name__}: {exc}")

    out = ROOT / "results" / "preflight.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S) -- do not start the run yet")
        for f in _failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  READY" + (f"  ({len(_warnings)} warning(s))" if _warnings else ""))
    for w in _warnings:
        print(f"    - {w}")
    print(f"  saved -> {out.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
