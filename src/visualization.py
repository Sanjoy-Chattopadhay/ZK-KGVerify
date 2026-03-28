"""
Visualization module for ZK-KGVerify paper figures and tables.
"""

import os
import json
import numpy as np
import matplotlib
import sys
# Use Agg backend only if NOT in a notebook (Colab/Jupyter use inline)
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curves(all_histories, save_dir="./results"):
    """Plot training loss curves for all models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {"TransE": "#1f77b4", "RotatE": "#ff7f0e", "CompGCN": "#2ca02c", "RGCN": "#d62728"}

    for model_name, history in all_histories.items():
        epochs = list(range(1, len(history["loss"]) + 1))
        color = colors.get(model_name, "#333333")
        ax.plot(epochs, history["loss"], label=model_name, color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Training Loss Curves — KG Embedding Models", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved training_curves.png")


def plot_metrics_comparison(all_metrics, save_dir="./results"):
    """Plot bar chart comparing metrics across models."""
    os.makedirs(save_dir, exist_ok=True)

    models = list(all_metrics.keys())
    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for i, model_name in enumerate(models):
        values = [all_metrics[model_name].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Metric", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Link Prediction Performance — FB15k-237", fontsize=16)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved metrics_comparison.png")


def plot_zkp_overhead(zkp_stats, save_dir="./results"):
    """Plot ZKP overhead analysis."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Proof generation time distribution
    if "gen_times" in zkp_stats:
        axes[0].hist(zkp_stats["gen_times"], bins=30, color="#2ca02c", alpha=0.7, edgecolor="black")
        axes[0].axvline(np.mean(zkp_stats["gen_times"]), color="red", linestyle="--",
                        label=f'Mean: {np.mean(zkp_stats["gen_times"])*1000:.2f}ms')
        axes[0].set_xlabel("Time (seconds)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Proof Generation Time Distribution", fontsize=14)
        axes[0].legend(fontsize=11)

    # 2. Proof size distribution
    if "proof_sizes" in zkp_stats:
        axes[1].hist(zkp_stats["proof_sizes"], bins=20, color="#ff7f0e", alpha=0.7, edgecolor="black")
        axes[1].axvline(np.mean(zkp_stats["proof_sizes"]), color="red", linestyle="--",
                        label=f'Mean: {np.mean(zkp_stats["proof_sizes"]):.0f} bytes')
        axes[1].set_xlabel("Size (bytes)", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Proof Size Distribution", fontsize=14)
        axes[1].legend(fontsize=11)

    # 3. Verification time
    if "verify_times" in zkp_stats:
        axes[2].hist(zkp_stats["verify_times"], bins=30, color="#1f77b4", alpha=0.7, edgecolor="black")
        axes[2].axvline(np.mean(zkp_stats["verify_times"]), color="red", linestyle="--",
                        label=f'Mean: {np.mean(zkp_stats["verify_times"])*1000:.2f}ms')
        axes[2].set_xlabel("Time (seconds)", fontsize=12)
        axes[2].set_ylabel("Frequency", fontsize=12)
        axes[2].set_title("Proof Verification Time Distribution", fontsize=14)
        axes[2].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "zkp_overhead.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved zkp_overhead.png")


def plot_blockchain_stats(bc_stats, save_dir="./results"):
    """Plot blockchain statistics."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Gas cost breakdown
    labels = ["Base TX Cost", "Data Storage", "Total per TX"]
    base_cost = 21000
    data_cost = bc_stats.get("avg_gas_per_tx", 0) - base_cost
    total = bc_stats.get("avg_gas_per_tx", 0)
    values = [base_cost, max(0, data_cost), total]
    colors_gas = ["#3498db", "#e74c3c", "#2ecc71"]

    bars = axes[0].bar(labels, values, color=colors_gas, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 500,
                     f'{val:,.0f}', ha='center', va='bottom', fontsize=11)
    axes[0].set_ylabel("Gas Units", fontsize=12)
    axes[0].set_title("Average Gas Cost per Transaction", fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Summary pie chart
    labels_pie = ["Verified", "Stored on Chain"]
    sizes = [bc_stats.get("total_transactions", 0), bc_stats.get("total_transactions", 0)]
    colors_pie = ["#2ecc71", "#3498db"]
    axes[1].pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[1].set_title(f"Total Records: {bc_stats.get('total_transactions', 0)}", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "blockchain_stats.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved blockchain_stats.png")


def plot_end_to_end_pipeline(pipeline_times, save_dir="./results"):
    """Plot end-to-end pipeline timing breakdown."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    stages = list(pipeline_times.keys())
    times = list(pipeline_times.values())
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    bars = ax.barh(stages, times, color=colors[:len(stages)], edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                f'{val:.2f}s', ha='left', va='center', fontsize=11)

    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_title("ZK-KGVerify Pipeline — Time Breakdown", fontsize=16)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pipeline_timing.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pipeline_timing.png")


def generate_latex_tables(all_metrics, zkp_stats, bc_stats, save_dir="./results"):
    """Generate LaTeX tables for the paper."""
    os.makedirs(save_dir, exist_ok=True)

    # Table 1: Link Prediction Results
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Link Prediction Results on FB15k-237}")
    lines.append("\\label{tab:link_prediction}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Model & MRR & Hits@1 & Hits@3 & Hits@10 \\\\")
    lines.append("\\hline")

    for model_name, metrics in all_metrics.items():
        lines.append(f"{model_name} & {metrics['MRR']:.4f} & {metrics['Hits@1']:.4f} & "
                     f"{metrics['Hits@3']:.4f} & {metrics['Hits@10']:.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Table 2: ZKP Overhead
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{ZKP Overhead Analysis}")
    lines.append("\\label{tab:zkp_overhead}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("Metric & Value \\\\")
    lines.append("\\hline")
    lines.append(f"Avg. Proof Generation Time & {zkp_stats.get('avg_gen_time', 0)*1000:.2f} ms \\\\")
    lines.append(f"Avg. Proof Verification Time & {zkp_stats.get('avg_verify_time', 0)*1000:.2f} ms \\\\")
    lines.append(f"Avg. Proof Size & {zkp_stats.get('avg_proof_size_bytes', 0):.0f} bytes \\\\")
    lines.append(f"Verification Rate & {zkp_stats.get('verification_rate', 0)*100:.1f}\\% \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Table 3: Blockchain Costs
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Blockchain Storage Costs}")
    lines.append("\\label{tab:blockchain}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("Metric & Value \\\\")
    lines.append("\\hline")
    lines.append(f"Total Transactions & {bc_stats.get('total_transactions', 0)} \\\\")
    lines.append(f"Total Gas Used & {bc_stats.get('total_gas_used', 0):,} \\\\")
    lines.append(f"Avg. Gas per TX & {bc_stats.get('avg_gas_per_tx', 0):,.0f} \\\\")
    lines.append(f"Avg. Mining Time & {bc_stats.get('avg_mining_time', 0)*1000:.2f} ms \\\\")
    lines.append(f"Chain Valid & {bc_stats.get('chain_valid', False)} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)
    with open(os.path.join(save_dir, "tables.tex"), "w") as f:
        f.write(latex_content)

    print(f"  Saved tables.tex")
    return latex_content


def save_all_results(all_metrics, all_histories, zkp_stats, bc_stats, pipeline_times, save_dir="./results"):
    """Save all experimental results to JSON for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "link_prediction_metrics": all_metrics,
        "training_histories": {k: {"loss": v["loss"], "total_time": v.get("total_time", 0)} for k, v in all_histories.items()},
        "zkp_statistics": {k: v for k, v in zkp_stats.items() if k not in ["gen_times", "verify_times", "proof_sizes"]},
        "blockchain_statistics": bc_stats,
        "pipeline_timing": pipeline_times,
    }

    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved all_results.json")
