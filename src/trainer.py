"""
Training and evaluation pipeline for KG embedding models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm


class MarginRankingLoss(nn.Module):
    """Margin-based ranking loss for KG embeddings."""

    def __init__(self, margin=6.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_score, neg_score):
        # pos_score: (batch,), neg_score: (batch, neg_size)
        # We want pos_score > neg_score by at least margin
        target = torch.ones_like(neg_score)
        loss = F.margin_ranking_loss(
            pos_score.unsqueeze(1).expand_as(neg_score),
            neg_score,
            target,
            margin=self.margin
        )
        return loss


class BinaryCrossEntropyLoss(nn.Module):
    """Self-adversarial negative sampling loss (used by RotatE)."""

    def __init__(self, adversarial_temperature=1.0):
        super().__init__()
        self.adversarial_temperature = adversarial_temperature

    def forward(self, pos_score, neg_score):
        # Positive loss
        pos_loss = -F.logsigmoid(pos_score).mean()

        # Self-adversarial weights
        with torch.no_grad():
            neg_weights = F.softmax(neg_score * self.adversarial_temperature, dim=-1)

        # Negative loss with adversarial weighting
        neg_loss = -(neg_weights * F.logsigmoid(-neg_score)).sum(dim=-1).mean()

        return (pos_loss + neg_loss) / 2


def train_model(model, train_loader, dataset, config, device="cpu"):
    """
    Train a KG embedding model.

    Returns: training history (losses per epoch, training time)
    """
    model = model.to(device)

    # Set graph structure for GCN-based models
    if hasattr(model, 'set_graph'):
        edge_index = dataset.train_triples[:, [0, 2]].t().to(device)
        edge_type = dataset.train_triples[:, 1].to(device)
        model.set_graph(edge_index, edge_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Choose loss function based on model type
    model_name = model.__class__.__name__
    if model_name == "RotatE":
        criterion = BinaryCrossEntropyLoss(adversarial_temperature=1.0)
    else:
        criterion = MarginRankingLoss(margin=config.MARGIN)

    history = {"loss": [], "epoch_time": []}
    total_start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            head, relation, tail, neg_samples = [b.to(device) for b in batch]

            pos_score, neg_score = model(head, relation, tail, neg_samples)
            loss = criterion(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        history["loss"].append(avg_loss)
        history["epoch_time"].append(epoch_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config.NUM_EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

    total_time = time.time() - total_start
    history["total_time"] = total_time
    print(f"  Training completed in {total_time:.2f}s")

    return history


@torch.no_grad()
def evaluate_model(model, dataset, config, device="cpu", max_eval=None):
    """
    Evaluate a KG embedding model on the test set.
    Uses filtered ranking protocol.

    Args:
        max_eval: Maximum number of test triples to evaluate (for speed).

    Returns: dict with MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    model = model.to(device)

    # Set graph for GCN models
    if hasattr(model, 'set_graph'):
        edge_index = dataset.train_triples[:, [0, 2]].t().to(device)
        edge_type = dataset.train_triples[:, 1].to(device)
        model.set_graph(edge_index, edge_type)

    test_triples = dataset.test_triples
    true_triples = dataset.get_all_true_triples()

    # Subsample test set for speed (if max_eval is set)
    if max_eval is not None and len(test_triples) > max_eval:
        indices = torch.randperm(len(test_triples))[:max_eval]
        test_triples = test_triples[indices]

    ranks = []
    eval_start = time.time()

    for i in tqdm(range(len(test_triples)), desc="  Evaluating", leave=False):
        h, r, t = test_triples[i].tolist()

        head_idx = torch.tensor([h], device=device)
        rel_idx = torch.tensor([r], device=device)

        # Get scores for all entities as tail
        scores = model.predict(head_idx, rel_idx).squeeze(0)

        # Filter: set scores of other true triples to -inf
        for ent in range(dataset.num_entities):
            if (h, r, ent) in true_triples and ent != t:
                scores[ent] = float('-inf')

        # Rank of the true tail
        rank = (scores >= scores[t]).sum().item()
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float32)
    eval_time = time.time() - eval_start

    metrics = {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "Mean_Rank": float(np.mean(ranks)),
        "eval_time": eval_time,
        "num_evaluated": len(test_triples),
    }

    print(f"  MRR: {metrics['MRR']:.4f} | Hits@1: {metrics['Hits@1']:.4f} | "
          f"Hits@3: {metrics['Hits@3']:.4f} | Hits@10: {metrics['Hits@10']:.4f}")

    return metrics
