"""
Knowledge Graph Embedding Models for link prediction.

Models implemented:
1. TransE  - Translational distance model (Bordes et al., 2013)
2. RotatE  - Rotation-based model (Sun et al., 2019)
3. CompGCN - Composition-based GCN (Vashishth et al., 2020)
4. R-GCN   - Relational Graph Convolutional Network (Schlichtkrull et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.
    Score function: ||h + r - t||
    """

    def __init__(self, num_entities, num_relations, embedding_dim=128, margin=6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize with uniform distribution
        nn.init.uniform_(self.entity_embeddings.weight, -6.0/math.sqrt(embedding_dim), 6.0/math.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6.0/math.sqrt(embedding_dim), 6.0/math.sqrt(embedding_dim))

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=-1)

    def score(self, head, relation, tail):
        """Compute TransE score: -||h + r - t||_2"""
        return -torch.norm(head + relation - tail, p=2, dim=-1)

    def forward(self, head_idx, relation_idx, tail_idx, negative_idx):
        """
        Forward pass with negative sampling.
        Returns positive scores and negative scores.
        """
        head = self.entity_embeddings(head_idx)
        relation = self.relation_embeddings(relation_idx)
        tail = self.entity_embeddings(tail_idx)
        neg_entities = self.entity_embeddings(negative_idx)

        # Normalize entity embeddings
        head = F.normalize(head, p=2, dim=-1)
        tail = F.normalize(tail, p=2, dim=-1)
        neg_entities = F.normalize(neg_entities, p=2, dim=-1)

        # Positive score
        pos_score = self.score(head, relation, tail)

        # Negative scores (corrupt tail)
        # head: (batch,dim), relation: (batch,dim), neg: (batch,neg_size,dim)
        neg_score = -torch.norm(
            head.unsqueeze(1) + relation.unsqueeze(1) - neg_entities,
            p=2, dim=-1
        )

        return pos_score, neg_score

    def predict(self, head_idx, relation_idx):
        """Predict scores for all possible tails given (head, relation)."""
        head = F.normalize(self.entity_embeddings(head_idx), p=2, dim=-1)
        relation = self.relation_embeddings(relation_idx)
        all_entities = F.normalize(self.entity_embeddings.weight, p=2, dim=-1)

        # Score all possible tails: -||h + r - t||
        scores = -torch.norm(
            (head + relation).unsqueeze(1) - all_entities.unsqueeze(0),
            p=2, dim=-1
        )
        return scores

    def get_embedding_vector(self, head_idx, relation_idx, tail_idx):
        """Get the concatenated embedding vector for ZKP commitment."""
        head = self.entity_embeddings(head_idx)
        relation = self.relation_embeddings(relation_idx)
        tail = self.entity_embeddings(tail_idx)
        return torch.cat([head, relation, tail], dim=-1)


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
    Score function: -||h ∘ r - t|| where ∘ is Hadamard product in complex space.
    """

    def __init__(self, num_entities, num_relations, embedding_dim=128, margin=6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Complex embeddings: real and imaginary parts
        self.entity_embeddings_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_im = nn.Embedding(num_entities, embedding_dim)
        # Relations are phase angles
        self.relation_phases = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embeddings_re.weight, -bound, bound)
        nn.init.uniform_(self.entity_embeddings_im.weight, -bound, bound)
        nn.init.uniform_(self.relation_phases.weight, -math.pi, math.pi)

    def _get_entity_complex(self, idx):
        return self.entity_embeddings_re(idx), self.entity_embeddings_im(idx)

    def _get_relation_complex(self, idx):
        phase = self.relation_phases(idx)
        return torch.cos(phase), torch.sin(phase)

    def score(self, h_re, h_im, r_re, r_im, t_re, t_im):
        """Compute RotatE score: -||h ∘ r - t||"""
        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        diff_re = hr_re - t_re
        diff_im = hr_im - t_im

        return -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)

    def forward(self, head_idx, relation_idx, tail_idx, negative_idx):
        h_re, h_im = self._get_entity_complex(head_idx)
        r_re, r_im = self._get_relation_complex(relation_idx)
        t_re, t_im = self._get_entity_complex(tail_idx)

        pos_score = self.score(h_re, h_im, r_re, r_im, t_re, t_im)

        # Negative sampling (corrupt tail)
        neg_re = self.entity_embeddings_re(negative_idx)
        neg_im = self.entity_embeddings_im(negative_idx)

        neg_score = self.score(
            h_re.unsqueeze(1), h_im.unsqueeze(1),
            r_re.unsqueeze(1), r_im.unsqueeze(1),
            neg_re, neg_im
        )

        return pos_score, neg_score

    def predict(self, head_idx, relation_idx):
        h_re, h_im = self._get_entity_complex(head_idx)
        r_re, r_im = self._get_relation_complex(relation_idx)

        all_re = self.entity_embeddings_re.weight
        all_im = self.entity_embeddings_im.weight

        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        diff_re = hr_re.unsqueeze(1) - all_re.unsqueeze(0)
        diff_im = hr_im.unsqueeze(1) - all_im.unsqueeze(0)

        return -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)

    def get_embedding_vector(self, head_idx, relation_idx, tail_idx):
        h_re, h_im = self._get_entity_complex(head_idx)
        t_re, t_im = self._get_entity_complex(tail_idx)
        phase = self.relation_phases(relation_idx)
        return torch.cat([h_re, h_im, phase, t_re, t_im], dim=-1)


class CompGCNConv(nn.Module):
    """Single CompGCN convolution layer."""

    def __init__(self, in_dim, out_dim, num_relations, composition="sub"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.composition = composition

        self.W_O = nn.Linear(in_dim, out_dim, bias=False)  # Original direction
        self.W_I = nn.Linear(in_dim, out_dim, bias=False)  # Inverse direction
        self.W_S = nn.Linear(in_dim, out_dim, bias=False)  # Self-loop

        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)  # Relation update

        self.bn = nn.BatchNorm1d(out_dim)

    def compose(self, entity_emb, relation_emb):
        if self.composition == "sub":
            return entity_emb - relation_emb
        elif self.composition == "mult":
            return entity_emb * relation_emb
        else:
            return entity_emb - relation_emb

    def forward(self, entity_emb, relation_emb, edge_index, edge_type):
        """
        entity_emb: (num_entities, in_dim)
        relation_emb: (num_relations, in_dim)
        edge_index: (2, num_edges)
        edge_type: (num_edges,)
        """
        num_entities = entity_emb.size(0)
        out = torch.zeros(num_entities, self.out_dim, device=entity_emb.device)

        # Self-loop
        out += self.W_S(entity_emb)

        # Message passing for original edges
        src, dst = edge_index[0], edge_index[1]
        rel_emb = relation_emb[edge_type]
        msg = self.compose(entity_emb[src], rel_emb)
        msg = self.W_O(msg)

        # Aggregate messages
        out.index_add_(0, dst, msg)

        # Inverse edges
        inv_msg = self.compose(entity_emb[dst], rel_emb)
        inv_msg = self.W_I(inv_msg)
        out.index_add_(0, src, inv_msg)

        # Normalize by degree
        deg = torch.zeros(num_entities, device=entity_emb.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=entity_emb.device))
        deg.index_add_(0, src, torch.ones(src.size(0), device=entity_emb.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        out = out / deg

        out = self.bn(out)
        out = F.relu(out)

        # Update relation embeddings
        new_rel_emb = self.W_rel(relation_emb)

        return out, new_rel_emb


class CompGCN(nn.Module):
    """
    CompGCN: Composition-Based Multi-Relational Graph Convolutional Networks.
    Uses GCN layers with composition operations over entity-relation pairs.
    """

    def __init__(self, num_entities, num_relations, embedding_dim=128, num_layers=2, margin=6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CompGCNConv(embedding_dim, embedding_dim, num_relations))

        self.edge_index = None
        self.edge_type = None

    def set_graph(self, edge_index, edge_type):
        """Set the graph structure for message passing."""
        self.edge_index = edge_index
        self.edge_type = edge_type

    def encode(self):
        """Run GCN encoding to get entity and relation embeddings."""
        entity_emb = self.entity_embeddings.weight
        relation_emb = self.relation_embeddings.weight

        for layer in self.layers:
            entity_emb, relation_emb = layer(entity_emb, relation_emb, self.edge_index, self.edge_type)

        return entity_emb, relation_emb

    def score_fn(self, head, relation, tail):
        """TransE-style scoring after GCN encoding."""
        return -torch.norm(head + relation - tail, p=2, dim=-1)

    def forward(self, head_idx, relation_idx, tail_idx, negative_idx):
        entity_emb, relation_emb = self.encode()

        head = entity_emb[head_idx]
        relation = relation_emb[relation_idx]
        tail = entity_emb[tail_idx]
        neg = entity_emb[negative_idx]

        pos_score = self.score_fn(head, relation, tail)
        neg_score = self.score_fn(
            head.unsqueeze(1), relation.unsqueeze(1), neg
        )

        return pos_score, neg_score

    def predict(self, head_idx, relation_idx):
        entity_emb, relation_emb = self.encode()

        head = entity_emb[head_idx]
        relation = relation_emb[relation_idx]
        all_entities = entity_emb

        scores = -torch.norm(
            (head + relation).unsqueeze(1) - all_entities.unsqueeze(0),
            p=2, dim=-1
        )
        return scores

    def get_embedding_vector(self, head_idx, relation_idx, tail_idx):
        entity_emb, relation_emb = self.encode()
        head = entity_emb[head_idx]
        relation = relation_emb[relation_idx]
        tail = entity_emb[tail_idx]
        return torch.cat([head, relation, tail], dim=-1)


class RGCNConv(nn.Module):
    """Single R-GCN convolution layer with basis decomposition."""

    def __init__(self, in_dim, out_dim, num_relations, num_bases=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis decomposition
        self.bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)

        self.bn = nn.BatchNorm1d(out_dim)

        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, entity_emb, edge_index, edge_type):
        num_entities = entity_emb.size(0)
        out = self.self_loop(entity_emb)

        # Compute relation-specific weight matrices via basis decomposition
        # W_r = sum_b(a_rb * B_b)  — shape (num_relations, in_dim, out_dim)
        W = torch.einsum("rb,bij->rij", self.coefficients, self.bases)

        src, dst = edge_index[0], edge_index[1]

        # Process per relation type to avoid materializing (num_edges, in, out)
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            # (num_edges_r, in_dim) @ (in_dim, out_dim) -> (num_edges_r, out_dim)
            msg = entity_emb[src_r] @ W[r]
            out.index_add_(0, dst_r, msg)

        # Normalize
        deg = torch.zeros(num_entities, device=entity_emb.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=entity_emb.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        out = out / deg

        out = self.bn(out)
        out = F.relu(out)

        return out


class RGCN(nn.Module):
    """
    R-GCN: Relational Graph Convolutional Network.
    Uses basis decomposition for relation-specific weight matrices.
    """

    def __init__(self, num_entities, num_relations, embedding_dim=128, num_layers=2, num_bases=4, margin=6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases))

        self.edge_index = None
        self.edge_type = None

    def set_graph(self, edge_index, edge_type):
        self.edge_index = edge_index
        self.edge_type = edge_type

    def encode(self):
        entity_emb = self.entity_embeddings.weight
        for layer in self.layers:
            entity_emb = layer(entity_emb, self.edge_index, self.edge_type)
        return entity_emb

    def score_fn(self, head, relation, tail):
        return -torch.norm(head + relation - tail, p=2, dim=-1)

    def forward(self, head_idx, relation_idx, tail_idx, negative_idx):
        entity_emb = self.encode()
        relation_emb = self.relation_embeddings.weight

        head = entity_emb[head_idx]
        relation = relation_emb[relation_idx]
        tail = entity_emb[tail_idx]
        neg = entity_emb[negative_idx]

        pos_score = self.score_fn(head, relation, tail)
        neg_score = self.score_fn(
            head.unsqueeze(1), relation.unsqueeze(1), neg
        )

        return pos_score, neg_score

    def predict(self, head_idx, relation_idx):
        entity_emb = self.encode()
        relation_emb = self.relation_embeddings.weight

        head = entity_emb[head_idx]
        relation = relation_emb[relation_idx]

        scores = -torch.norm(
            (head + relation).unsqueeze(1) - entity_emb.unsqueeze(0),
            p=2, dim=-1
        )
        return scores

    def get_embedding_vector(self, head_idx, relation_idx, tail_idx):
        entity_emb = self.encode()
        head = entity_emb[head_idx]
        relation = self.relation_embeddings(relation_idx)
        tail = entity_emb[tail_idx]
        return torch.cat([head, relation, tail], dim=-1)


def get_model(model_name, num_entities, num_relations, embedding_dim=128, margin=6.0):
    """Factory function to create models by name."""
    models = {
        "TransE": TransE,
        "RotatE": RotatE,
        "CompGCN": CompGCN,
        "RGCN": RGCN,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        margin=margin
    )
