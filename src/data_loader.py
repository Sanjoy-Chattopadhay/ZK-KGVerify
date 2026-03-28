"""
Data loading and preprocessing for FB15k-237 knowledge graph dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class KGTriple:
    """Represents a knowledge graph triple (head, relation, tail)."""
    def __init__(self, head, relation, tail):
        self.head = head
        self.relation = relation
        self.tail = tail


class FB15k237Dataset:
    """
    FB15k-237 Knowledge Graph dataset loader.
    Downloads and processes the dataset from PyTorch Geometric or loads from local files.
    """

    DATASET_URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/"

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.num_entities = 0
        self.num_relations = 0

        self.train_triples = None
        self.valid_triples = None
        self.test_triples = None

        self._load_dataset()

    def _download_if_needed(self):
        """Download FB15k-237 dataset files if not present."""
        os.makedirs(self.data_dir, exist_ok=True)
        fb_dir = os.path.join(self.data_dir, "FB15k-237")
        os.makedirs(fb_dir, exist_ok=True)

        files = ["train.txt", "valid.txt", "test.txt"]
        for fname in files:
            fpath = os.path.join(fb_dir, fname)
            if not os.path.exists(fpath):
                import urllib.request
                url = self.DATASET_URL + fname
                print(f"Downloading {fname}...")
                urllib.request.urlretrieve(url, fpath)
                print(f"  Saved to {fpath}")

        return fb_dir

    def _load_dataset(self):
        """Load and process the dataset."""
        fb_dir = self._download_if_needed()

        # First pass: build entity and relation vocabularies from all splits
        all_entities = set()
        all_relations = set()

        for split in ["train", "valid", "test"]:
            fpath = os.path.join(fb_dir, f"{split}.txt")
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        h, r, t = parts
                        all_entities.add(h)
                        all_entities.add(t)
                        all_relations.add(r)

        # Create mappings
        self.entity2id = {e: i for i, e in enumerate(sorted(all_entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(all_relations))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        # Second pass: load triples as tensors
        self.train_triples = self._load_split(os.path.join(fb_dir, "train.txt"))
        self.valid_triples = self._load_split(os.path.join(fb_dir, "valid.txt"))
        self.test_triples = self._load_split(os.path.join(fb_dir, "test.txt"))

        print(f"Dataset loaded: {self.num_entities} entities, {self.num_relations} relations")
        print(f"  Train: {len(self.train_triples)} triples")
        print(f"  Valid: {len(self.valid_triples)} triples")
        print(f"  Test:  {len(self.test_triples)} triples")

    def _load_split(self, filepath):
        """Load a dataset split and return as tensor of shape (N, 3)."""
        triples = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = parts
                    triples.append([
                        self.entity2id[h],
                        self.relation2id[r],
                        self.entity2id[t]
                    ])
        return torch.tensor(triples, dtype=torch.long)

    def get_all_true_triples(self):
        """Return set of all true triples for filtered evaluation."""
        all_triples = torch.cat([self.train_triples, self.valid_triples, self.test_triples], dim=0)
        true_triples = set()
        for i in range(len(all_triples)):
            h, r, t = all_triples[i].tolist()
            true_triples.add((h, r, t))
        return true_triples


class KGTrainDataset(Dataset):
    """Dataset for training KG embeddings with negative sampling."""

    def __init__(self, triples, num_entities, negative_sample_size=64):
        self.triples = triples
        self.num_entities = num_entities
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        head, relation, tail = triple[0], triple[1], triple[2]

        # Generate negative samples by corrupting head or tail
        negative_samples = torch.randint(0, self.num_entities, (self.negative_sample_size,))

        return head, relation, tail, negative_samples


def get_data_loaders(dataset, batch_size=1024, negative_sample_size=64):
    """Create data loaders for training and evaluation."""
    train_dataset = KGTrainDataset(
        dataset.train_triples,
        dataset.num_entities,
        negative_sample_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    return train_loader
