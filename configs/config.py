"""
Configuration for ZK-KGVerify experiments.
"""

# Dataset
DATASET_NAME = "FB15k-237"
DATA_DIR = "./data"

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_EPOCHS = 200
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NEGATIVE_SAMPLE_SIZE = 64
MARGIN = 6.0  # For margin-based losses (TransE)

# Models to train
MODELS = ["TransE", "RotatE", "CompGCN"]

# Evaluation
EVAL_BATCH_SIZE = 256
EVAL_MAX = None  # None = full test set (20,466 triples). Override to a small int for smoke tests.
METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

# Reproducibility -- pinned across torch / numpy / Python random and CUDA.
RANDOM_SEED = 42

# Early stopping on training-loss plateau.
# Stops when no model has improved its loss by at least EARLY_STOP_MIN_DELTA
# for EARLY_STOP_PATIENCE consecutive epochs. Set EARLY_STOPPING=False to
# force the old behaviour of always running NUM_EPOCHS.
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-3
EARLY_STOP_MIN_EPOCHS = 30   # never stop before this epoch (lets warm-up finish)

# ZKP
NUM_ZKP_SAMPLES = 1000  # Number of predictions to generate proofs for
ZKP_CURVE = "bn128"  # Elliptic curve for Pedersen commitments

# Blockchain
BLOCKCHAIN_MODE = "local"  # "local" uses Python-based chain, "ganache" uses eth-tester
GAS_LIMIT = 3000000

# Results
RESULTS_DIR = "./results"

# Checkpointing -- survives Colab disconnects when pointed at a Drive folder.
# On Colab, set CHECKPOINT_DIR to a Google Drive path (e.g.
# /content/drive/MyDrive/ZK-KGVerify-checkpoints) so trained model weights
# and per-step results persist across sessions. Locally, ./checkpoints is fine.
# Leave RESUME=True so a restart skips already-trained / already-evaluated models.
CHECKPOINT_DIR = "./checkpoints"
RESUME = True

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
