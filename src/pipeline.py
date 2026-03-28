"""
ZK-KGVerify: End-to-End Pipeline

This is the main orchestration script that runs the full experiment:
1. Load FB15k-237 dataset
2. Train 4 KG embedding models (TransE, RotatE, CompGCN, R-GCN)
3. Evaluate link prediction (MRR, Hits@1/3/10)
4. Generate ZK proofs for predictions
5. Verify proofs and test tamper detection
6. Store verification results on blockchain
7. Generate figures and tables for the paper
"""

import sys
import os
import time
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from src.data_loader import FB15k237Dataset, get_data_loaders
from src.models import get_model
from src.trainer import train_model, evaluate_model
from src.zkp_module import (
    generate_proof, verify_proof, batch_generate_proofs,
    batch_verify_proofs, tamper_proof
)
from src.blockchain_module import create_blockchain
from src.visualization import (
    plot_training_curves, plot_metrics_comparison, plot_zkp_overhead,
    plot_blockchain_stats, plot_end_to_end_pipeline, generate_latex_tables,
    save_all_results
)


def run_full_pipeline():
    """Run the complete ZK-KGVerify experiment."""

    print("=" * 70)
    print("  ZK-KGVerify: Privacy-Preserving Verification of")
    print("  Knowledge Graph Reasoning using ZKPs and Blockchain")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()

    pipeline_times = {}

    # ============================================================
    # STEP 1: Load Dataset
    # ============================================================
    print("=" * 50)
    print("STEP 1: Loading FB15k-237 Dataset")
    print("=" * 50)

    step_start = time.time()
    dataset = FB15k237Dataset(data_dir=DATA_DIR)
    pipeline_times["1. Data Loading"] = time.time() - step_start
    print()

    # Create data loader
    train_loader = get_data_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        negative_sample_size=NEGATIVE_SAMPLE_SIZE
    )

    # ============================================================
    # STEP 2 & 3: Train and Evaluate Models
    # ============================================================
    print("=" * 50)
    print("STEP 2-3: Training and Evaluating KG Models")
    print("=" * 50)

    import configs.config as config_module

    all_histories = {}
    all_metrics = {}
    trained_models = {}

    step_start = time.time()

    for model_name in MODELS:
        print(f"\n--- {model_name} ---")

        model = get_model(
            model_name,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=EMBEDDING_DIM,
            margin=MARGIN
        )

        # Train
        history = train_model(model, train_loader, dataset, config_module, device=DEVICE)
        all_histories[model_name] = history

        # Evaluate
        print(f"  Evaluating {model_name}...")
        eval_max = getattr(config_module, 'EVAL_MAX', None)
        metrics = evaluate_model(model, dataset, config_module, device=DEVICE, max_eval=eval_max)
        all_metrics[model_name] = metrics

        trained_models[model_name] = model

    pipeline_times["2-3. Training + Evaluation"] = time.time() - step_start

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<12} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}")
    print("-" * 60)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<12} {metrics['MRR']:>8.4f} {metrics['Hits@1']:>8.4f} "
              f"{metrics['Hits@3']:>8.4f} {metrics['Hits@10']:>8.4f}")
    print("=" * 60)

    # ============================================================
    # STEP 4: Generate ZK Proofs
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 4: Generating Zero-Knowledge Proofs")
    print("=" * 50)

    step_start = time.time()

    # Use the best model for ZKP demonstration
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["MRR"])
    best_model = trained_models[best_model_name]
    print(f"\n  Using best model: {best_model_name} (MRR={all_metrics[best_model_name]['MRR']:.4f})")

    best_model.eval()
    best_model = best_model.to(DEVICE)

    # Set graph for GCN models
    if hasattr(best_model, 'set_graph'):
        edge_index = dataset.train_triples[:, [0, 2]].t().to(DEVICE)
        edge_type = dataset.train_triples[:, 1].to(DEVICE)
        best_model.set_graph(edge_index, edge_type)

    # Sample test triples for ZKP generation
    num_samples = min(NUM_ZKP_SAMPLES, len(dataset.test_triples))
    sample_indices = torch.randperm(len(dataset.test_triples))[:num_samples]
    sample_triples = dataset.test_triples[sample_indices]

    embedding_vectors = []
    prediction_scores = []
    triples_list = []

    with torch.no_grad():
        for i in range(num_samples):
            h, r, t = sample_triples[i].tolist()
            h_idx = torch.tensor([h], device=DEVICE)
            r_idx = torch.tensor([r], device=DEVICE)
            t_idx = torch.tensor([t], device=DEVICE)

            # Get embedding vector
            emb_vec = best_model.get_embedding_vector(h_idx, r_idx, t_idx)
            embedding_vectors.append(emb_vec.cpu().numpy().flatten())

            # Get prediction score
            scores = best_model.predict(h_idx, r_idx)
            prediction_scores.append(scores[0, t].item())
            triples_list.append((h, r, t))

    # Generate proofs
    print(f"\n  Generating {num_samples} ZK proofs...")
    proofs, gen_stats = batch_generate_proofs(
        embedding_vectors, prediction_scores, triples_list, best_model_name
    )

    print(f"  Avg generation time: {gen_stats['avg_gen_time']*1000:.2f} ms")
    print(f"  Avg proof size: {gen_stats['avg_proof_size_bytes']:.0f} bytes")

    # Verify proofs
    print(f"\n  Verifying {num_samples} proofs...")
    results, verify_stats = batch_verify_proofs(proofs)

    print(f"  Verification rate: {verify_stats['verification_rate']*100:.1f}%")
    print(f"  Avg verification time: {verify_stats['avg_verify_time']*1000:.2f} ms")

    # Test tamper detection
    print(f"\n  Testing tamper detection...")
    tampered_proofs = [tamper_proof(p) for p in proofs[:100]]
    tampered_results, tampered_stats = batch_verify_proofs(tampered_proofs)
    tampered_detected = tampered_stats["num_invalid"]
    print(f"  Tampered proofs detected: {tampered_detected}/{len(tampered_proofs)} "
          f"({tampered_detected/len(tampered_proofs)*100:.1f}%)")

    # Collect detailed ZKP stats for visualization
    gen_times = []
    verify_times = []
    proof_sizes = []
    for p in proofs:
        proof_sizes.append(p.proof_size_bytes)
    # Re-time for detailed stats
    for i in range(min(200, len(proofs))):
        start = time.time()
        generate_proof(embedding_vectors[i], prediction_scores[i], triples_list[i], best_model_name)
        gen_times.append(time.time() - start)

        start = time.time()
        verify_proof(proofs[i])
        verify_times.append(time.time() - start)

    zkp_full_stats = {
        **gen_stats,
        **verify_stats,
        "gen_times": gen_times,
        "verify_times": verify_times,
        "proof_sizes": proof_sizes,
        "legitimate_verification_rate": verify_stats["verification_rate"],
        "tamper_detection_rate": tampered_detected / len(tampered_proofs),
    }

    pipeline_times["4. ZKP Generation + Verification"] = time.time() - step_start

    # ============================================================
    # STEP 5: Store on Blockchain
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 5: Storing Verification Results on Blockchain")
    print("=" * 50)

    step_start = time.time()
    blockchain = create_blockchain(mode=BLOCKCHAIN_MODE)

    print(f"\n  Logging {num_samples} verification records...")
    for i, proof in enumerate(proofs):
        blockchain.add_verification_record(
            model_id=proof.model_id,
            triple=proof.triple,
            prediction_score=proof.score,
            zkp_commitment=proof.commitment,
            zkp_verified=results[i],
            proof_hash=proof.prediction_hash
        )

    # Mine remaining transactions
    blockchain.mine_pending()
    bc_stats = blockchain.get_stats()

    print(f"  Blocks mined: {bc_stats['num_blocks']}")
    print(f"  Total transactions: {bc_stats['total_transactions']}")
    print(f"  Total gas used: {bc_stats['total_gas_used']:,}")
    print(f"  Avg gas per TX: {bc_stats['avg_gas_per_tx']:,.0f}")
    print(f"  Chain valid: {bc_stats['chain_valid']}")

    pipeline_times["5. Blockchain Storage"] = time.time() - step_start

    # ============================================================
    # STEP 6: Generate Figures and Tables
    # ============================================================
    print("\n" + "=" * 50)
    print("STEP 6: Generating Figures and Tables")
    print("=" * 50)

    step_start = time.time()

    plot_training_curves(all_histories, save_dir=RESULTS_DIR)
    plot_metrics_comparison(all_metrics, save_dir=RESULTS_DIR)
    plot_zkp_overhead(zkp_full_stats, save_dir=RESULTS_DIR)
    plot_blockchain_stats(bc_stats, save_dir=RESULTS_DIR)
    plot_end_to_end_pipeline(pipeline_times, save_dir=RESULTS_DIR)
    generate_latex_tables(all_metrics, zkp_full_stats, bc_stats, save_dir=RESULTS_DIR)
    save_all_results(all_metrics, all_histories, zkp_full_stats, bc_stats, pipeline_times, save_dir=RESULTS_DIR)

    pipeline_times["6. Visualization"] = time.time() - step_start

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Total pipeline time: {sum(pipeline_times.values()):.2f}s")
    print(f"\n  Timing breakdown:")
    for stage, t in pipeline_times.items():
        print(f"    {stage}: {t:.2f}s")

    print(f"\n  Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"  Files generated:")
    for f in os.listdir(RESULTS_DIR):
        print(f"    - {f}")

    print("\n  Key findings:")
    print(f"    Best model: {best_model_name} (MRR={all_metrics[best_model_name]['MRR']:.4f})")
    print(f"    ZKP overhead: {zkp_full_stats['avg_gen_time']*1000:.2f}ms per proof")
    print(f"    Tamper detection: {zkp_full_stats['tamper_detection_rate']*100:.1f}%")
    print(f"    Blockchain cost: {bc_stats['avg_gas_per_tx']:,.0f} gas/tx")
    print("=" * 70)

    return {
        "metrics": all_metrics,
        "histories": all_histories,
        "zkp_stats": zkp_full_stats,
        "blockchain_stats": bc_stats,
        "pipeline_times": pipeline_times,
    }


if __name__ == "__main__":
    run_full_pipeline()
