"""
Training and evaluation pipeline for KG embedding models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

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

    # Early-stopping bookkeeping (config flags optional; safe defaults applied).
    es_enabled    = bool(getattr(config, "EARLY_STOPPING", False))
    es_patience   = int(getattr(config, "EARLY_STOP_PATIENCE", 20))
    es_min_delta  = float(getattr(config, "EARLY_STOP_MIN_DELTA", 1e-3))
    es_min_epochs = int(getattr(config, "EARLY_STOP_MIN_EPOCHS", 30))
    best_loss = float("inf")
    epochs_without_improvement = 0

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

        # Early stopping: count how many consecutive epochs have failed to
        # improve best_loss by at least es_min_delta. Only check after the
        # min-epochs warm-up so we don't kill RotatE/CompGCN during ramp.
        if es_enabled and (epoch + 1) >= es_min_epochs:
            # Compare against the best loss seen so far, and always lower the
            # bar when the loss drops. Previously best_loss only moved when an
            # epoch beat it by the full min_delta, so a slow steady decline
            # kept accumulating until it crossed the threshold and reset the
            # counter -- a model creeping down by 1e-4 an epoch never
            # plateaued and never stopped. That is what made RGCN run all 200
            # epochs at ~108 s each.
            improved = best_loss - avg_loss > es_min_delta
            if avg_loss < best_loss:
                best_loss = avg_loss
            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= es_patience:
                    print(f"  [early stop] no improvement >= {es_min_delta} "
                          f"for {es_patience} epochs (best={best_loss:.4f}); "
                          f"stopping at epoch {epoch+1}/{config.NUM_EPOCHS}")
                    break
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss

    total_time = time.time() - total_start
    history["total_time"] = total_time
    history["epochs_run"] = len(history["loss"])
    print(f"  Training completed in {total_time:.2f}s ({history['epochs_run']} epochs)")

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

    # Filtered protocol needs, for each (h, r), the set of tails known to be
    # true. Indexing by (h, r) turns the inner filter into a lookup over the
    # handful of true tails instead of a scan over all |E| entities -- the
    # original form cost |T_test| x |E| Python set probes (~3e8 on FB15k-237)
    # and dominated total runtime.
    hr2t = dataset.get_hr_to_tails()

    # Subsample test set for speed (if max_eval is set)
    if max_eval is not None and len(test_triples) > max_eval:
        indices = torch.randperm(len(test_triples))[:max_eval]
        test_triples = test_triples[indices]

    ranks = []
    eval_start = time.time()

    # Colab and CI capture stderr to a file, not a terminal. tqdm then emits a
    # NEW LINE per refresh instead of rewriting one -- ~7,000 lines per
    # evaluation, which floods the notebook frontend until it stops responding
    # and the run looks hung. Refresh sparsely when nobody is watching a TTY.
    _bar_interval = 1.0 if sys.stderr.isatty() else float(
        os.environ.get("ZKKG_PROGRESS_SECONDS", "30")
    )

    # Score a block of test triples per call rather than one. predict()
    # already accepts a batch of (head, relation) pairs and returns
    # (batch, |E|), so this is the same arithmetic in the same order -- but
    # 20,466 single-row GPU launches per dataset are latency-bound, and each
    # one costs far more in dispatch than in floating point. The block size
    # is what bounds memory: TransE materialises (batch, |E|, dim) floats
    # internally, which is ~1.2 GB per 32 triples on FB15k-237 at dim 128,
    # so 32 is chosen to fit a 6 GB card alongside the model. Override with
    # ZKKG_EVAL_BATCH on a larger GPU.
    block = int(os.environ.get("ZKKG_EVAL_BATCH", "32"))
    heads_all = test_triples[:, 0].to(device)
    rels_all = test_triples[:, 1].to(device)
    tails_all = test_triples[:, 2].tolist()
    hr_pairs = test_triples[:, [0, 1]].tolist()  # keyed into hr2t as tuples

    for start in tqdm(range(0, len(test_triples), block), desc="  Evaluating",
                      leave=False, mininterval=_bar_interval):
        stop = min(start + block, len(test_triples))
        rows = stop - start

        # (block, |E|) scores for every candidate tail.
        # contiguous() so the flat index_fill_ below writes through to the
        # same storage the ranks are then read from.
        block_scores = model.predict(heads_all[start:stop], rels_all[start:stop]).contiguous()
        num_entities = block_scores.size(1)

        targets = torch.as_tensor(tails_all[start:stop], dtype=torch.long,
                                  device=block_scores.device).unsqueeze(1)
        target_scores = block_scores.gather(1, targets)

        # Filter: mask every other known-true tail for each (h, r). Flatten
        # the per-row masks into one index list so the whole block costs a
        # single index_fill_ rather than one kernel per test triple.
        flat = []
        for row in range(rows):
            others = hr2t.get(tuple(hr_pairs[start + row]))
            if others:
                base = row * num_entities
                flat.extend(base + o for o in others)
        if flat:
            block_scores.view(-1).index_fill_(
                0, torch.as_tensor(flat, dtype=torch.long, device=block_scores.device),
                float("-inf"),
            )
        block_scores.scatter_(1, targets, target_scores)

        # Rank of the true tail (1-based; ties counted pessimistically).
        # One transfer per block, instead of a device sync per triple.
        block_ranks = (block_scores >= target_scores).sum(dim=1)
        ranks.extend(block_ranks.tolist())

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
