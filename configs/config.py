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
EVAL_MAX = None  # None = evaluate on full test set (20,466 triples)
METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

# ZKP
NUM_ZKP_SAMPLES = 1000  # Number of predictions to generate proofs for
ZKP_CURVE = "bn128"  # Elliptic curve for Pedersen commitments

# Blockchain
BLOCKCHAIN_MODE = "local"  # "local" uses Python-based chain, "ganache" uses eth-tester
GAS_LIMIT = 3000000

# Results
RESULTS_DIR = "./results"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
