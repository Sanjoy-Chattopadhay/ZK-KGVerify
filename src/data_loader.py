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


class KGDataset:
    """Loader for tab-separated link-prediction benchmarks.

    Handles any dataset published as train/valid/test.txt with one
    `head<TAB>relation<TAB>tail` triple per line. FB15k-237 is the primary
    benchmark; WN18RR is included because it has a very different profile
    (far fewer relations, much sparser, longer-tailed), which is what makes
    it a meaningful second point rather than a repeat of the first.
    """

    BASE_URL = (
        "https://raw.githubusercontent.com/villmow/"
        "datasets_knowledge_embedding/master/"
    )

    AVAILABLE = {
        "FB15k-237": "FB15k-237/",
        "WN18RR": "WN18RR/text/",
    }

    def __init__(self, name="FB15k-237", data_dir="./data"):
        if name not in self.AVAILABLE:
            raise ValueError(
                f"Unknown dataset {name!r}; choose from {sorted(self.AVAILABLE)}"
            )
        self.name = name
        self.data_dir = data_dir
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.num_entities = 0
        self.num_relations = 0
        self._hr2t = None

        self.train_triples = None
        self.valid_triples = None
        self.test_triples = None

        self._load_dataset()

    def _download_if_needed(self):
        """Fetch train/valid/test splits if they are not already on disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        ds_dir = os.path.join(self.data_dir, self.name)
        os.makedirs(ds_dir, exist_ok=True)

        base = self.BASE_URL + self.AVAILABLE[self.name]
        for fname in ["train.txt", "valid.txt", "test.txt"]:
            fpath = os.path.join(ds_dir, fname)
            if not os.path.exists(fpath):
                import urllib.request

                url = base + fname
                print(f"Downloading {self.name}/{fname} ...")
                urllib.request.urlretrieve(url, fpath)
                print(f"  Saved to {fpath}")

        return ds_dir

    def _load_dataset(self):
        """Load and process the dataset."""
        fb_dir = self._download_if_needed()
        self.dataset_dir = fb_dir

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

        print(f"{self.name}: {self.num_entities} entities, {self.num_relations} relations")
        print(f"  Train: {len(self.train_triples)} triples")
        print(f"  Valid: {len(self.valid_triples)} triples")
        print(f"  Test:  {len(self.test_triples)} triples")

    def stats(self):
        """Summary row used for the dataset table in the paper."""
        return {
            "name": self.name,
            "entities": self.num_entities,
            "relations": self.num_relations,
            "train": len(self.train_triples),
            "valid": len(self.valid_triples),
            "test": len(self.test_triples),
        }

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

    def get_hr_to_tails(self):
        """Map (head, relation) -> list of all known-true tails.

        This is the index form of `get_all_true_triples` and is what the
        filtered ranking protocol actually needs. Built once and cached,
        because evaluation queries it for every test triple.
        """
        if getattr(self, "_hr2t", None) is None:
            hr2t = {}
            all_triples = torch.cat(
                [self.train_triples, self.valid_triples, self.test_triples], dim=0
            )
            for h, r, t in all_triples.tolist():
                hr2t.setdefault((h, r), []).append(t)
            self._hr2t = hr2t
        return self._hr2t


class FB15k237Dataset(KGDataset):
    """Backwards-compatible alias kept so older scripts keep working."""

    def __init__(self, data_dir="./data"):
        super().__init__(name="FB15k-237", data_dir=data_dir)


class KGTrainDataset(Dataset):
    """Dataset for training KG embeddings with negative sampling.

    Kept for scripts that build a torch DataLoader themselves. Training goes
    through KGBatchLoader below, which is the same sampling done per batch
    instead of per triple.
    """

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


class KGBatchLoader:
    """Shuffled batches of (head, relation, tail, negatives), sampled in bulk.

    The per-triple Dataset above, wrapped in a DataLoader, drew its 64
    negatives with a separate ``torch.randint`` call for each of the 272,115
    training triples and then had default_collate stack 1024 single-element
    tensors back into every batch. That is ~150 ms of Python and memcpy per
    batch against ~2 ms of arithmetic, so training ran at roughly a
    fiftieth of what the GPU could do and the profile that should take an
    hour took most of a day.

    Sampling the whole batch in one call gives the same distribution --
    negatives are still i.i.d. uniform over entities, triples are still
    reshuffled every epoch -- with one kernel per batch instead of 1024.
    Holding the triples on the training device also removes a host-to-device
    copy per batch; the tensors are a few MB.
    """

    def __init__(self, triples, num_entities, batch_size=1024,
                 negative_sample_size=64, device=None):
        self.device = device or "cpu"
        self.triples = triples.to(self.device)
        self.num_entities = num_entities
        self.batch_size = batch_size
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return (len(self.triples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.triples), device=self.device)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = self.triples[idx]
            negatives = torch.randint(
                0, self.num_entities,
                (len(idx), self.negative_sample_size),
                device=self.device,
            )
            yield batch[:, 0], batch[:, 1], batch[:, 2], negatives


def get_data_loaders(dataset, batch_size=1024, negative_sample_size=64, device=None):
    """Create data loaders for training and evaluation."""
    return KGBatchLoader(
        dataset.train_triples,
        dataset.num_entities,
        batch_size=batch_size,
        negative_sample_size=negative_sample_size,
        device=device,
    )
