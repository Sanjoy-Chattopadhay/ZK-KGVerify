# ZK-KGVerify

**Privacy-Preserving Verification of Knowledge Graph Reasoning using Zero-Knowledge Proofs and Blockchain**

## Overview

ZK-KGVerify is a framework that combines knowledge graph embedding models with zero-knowledge proofs (ZKPs) and blockchain technology to enable **privacy-preserving, verifiable link prediction**. The system allows a prover to demonstrate the correctness of KG predictions without revealing the underlying embeddings, while recording verification results on an immutable ledger.

### Key Components

- **Knowledge Graph Embedding Models**: TransE, RotatE, and CompGCN trained on the FB15k-237 benchmark
- **Zero-Knowledge Proofs**: Pedersen commitment-based ZKP scheme for embedding verification
- **Blockchain Ledger**: Smart contract (Solidity) for immutable on-chain logging of verification records
- **End-to-End Pipeline**: Automated training, evaluation, proof generation, and result visualization

## Project Structure

```
ZK-KGVerify/
├── run.py                          # Quick-run entry point
├── requirements.txt                # Python dependencies
├── HOW_TO_RUN.txt                  # Detailed setup instructions
│
├── configs/
│   └── config.py                   # All hyperparameters
│
├── src/
│   ├── data_loader.py              # FB15k-237 dataset loading
│   ├── models.py                   # TransE, RotatE, CompGCN, R-GCN
│   ├── trainer.py                  # Training & evaluation loop
│   ├── zkp_module.py               # Pedersen commitments & ZK proofs
│   ├── blockchain_module.py        # Local blockchain simulation
│   ├── visualization.py            # Paper-ready figures & LaTeX tables
│   └── pipeline.py                 # End-to-end orchestration
│
├── contracts/
│   └── ZKKGVerify.sol              # Solidity smart contract
│
├── results/                        # Generated outputs
│   ├── all_results.json            # Full numerical results
│   ├── tables.tex                  # LaTeX tables for the paper
│   ├── training_curves.png         # Loss curves for all models
│   ├── metrics_comparison.png      # MRR, Hits@1/3/10 bar chart
│   ├── zkp_overhead.png            # ZKP timing & size analysis
│   ├── blockchain_stats.png        # Gas costs & chain statistics
│   └── pipeline_timing.png         # End-to-end timing breakdown
│
└── ZK_KGVerify_*.ipynb             # Jupyter notebooks (Colab-ready)
```

## Quick Start

### Option A: Google Colab (Recommended)

1. Upload this folder to Google Drive
2. Open `ZK_KGVerify_only code.ipynb` in Colab
3. Set runtime to **T4 GPU** (Runtime > Change runtime type)
4. Run all cells -- results appear in `./results/`

### Option B: Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run the full pipeline
python run.py
```

Results are saved to `./results/`.

## Configuration

Edit `configs/config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_DIM` | 128 | Embedding dimensionality |
| `NUM_EPOCHS` | 200 | Training epochs |
| `BATCH_SIZE` | 1024 | Training batch size |
| `NUM_ZKP_SAMPLES` | 1000 | Number of ZK proofs to generate |
| `BLOCKCHAIN_MODE` | `"local"` | `"local"` or `"ganache"` |

For a quick test, set `NUM_EPOCHS = 5` and `NUM_ZKP_SAMPLES = 100`.

## Generated Outputs

| Output | Description |
|--------|-------------|
| `training_curves.png` | Loss curves for all models |
| `metrics_comparison.png` | Bar chart comparing MRR, Hits@1/3/10 |
| `zkp_overhead.png` | Proof generation time, size, verification time |
| `blockchain_stats.png` | Gas costs and chain statistics |
| `pipeline_timing.png` | End-to-end timing breakdown |
| `tables.tex` | LaTeX tables ready to paste into the paper |
| `all_results.json` | Complete numerical results in JSON |

## Smart Contract

The `ZKKGVerify.sol` contract stores on-chain verification records containing:
- Model ID and predicted triple (head, relation, tail)
- Prediction score and Pedersen commitment hash
- ZKP verification result and proof hash
- Timestamp for auditability

## Estimated Runtimes

| Hardware | Time |
|----------|------|
| Colab T4 GPU | ~30-40 min |
| Local GPU (RTX) | ~20-30 min |
| CPU only | ~1-2 hours |

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NVIDIA GPU with CUDA (optional, CPU works but slower)
- See `requirements.txt` for full list

## License

MIT
