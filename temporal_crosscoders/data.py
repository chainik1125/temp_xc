"""
data.py — Toy model, synthetic data generators, and data iterators.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from typing import Any, Callable

from config import (
    NUM_FEATS, HIDDEN_DIM, FEAT_PROB, FEAT_MEAN, FEAT_STD,
    MARKOV_ALPHA, MARKOV_BETA, DEVICE,
)


# ─── Toy model ──────────────────────────────────────────────────────────────────

def orthogonalize(num_vectors: int, vector_len: int, target_cos_sim: float = 0) -> torch.Tensor:
    """Optimise embeddings so all pairwise cosine similarities ≈ target_cos_sim."""
    embeddings = torch.randn(num_vectors, vector_len)
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    embeddings.requires_grad_(True)
    optimizer = torch.optim.Adam([embeddings], lr=0.01)
    pbar = tqdm(range(1000), desc="Orthogonalising features", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        dot = embeddings @ embeddings.T
        diff = dot - target_cos_sim
        diff.fill_diagonal_(0)
        loss = diff.pow(2).sum() + num_vectors * (dot.diag() - 1).pow(2).sum()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Orthogonalising | loss={loss.item():.3f}")
    with torch.no_grad():
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    return embeddings.detach().clone()


class ToyModel(nn.Module):
    """Linear map from feature space to representation space."""

    def __init__(self, num_feats: int, hidden_dim: int, target_cos_sim: float = 0):
        super().__init__()
        self.embed = nn.Linear(num_feats, hidden_dim, bias=False)
        embeddings = orthogonalize(num_feats, hidden_dim, target_cos_sim)
        self.embed.weight.data = embeddings.T  # (hidden_dim, num_feats)

    def forward(self, x: torch.Tensor, **kwargs: Any):
        return self.embed(x)


def build_toy_model(seed: int = 42) -> ToyModel:
    """Build and freeze the shared toy model (deterministic)."""
    torch.manual_seed(seed)
    model = ToyModel(NUM_FEATS, HIDDEN_DIM).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_true_features(toy_model: ToyModel) -> torch.Tensor:
    """(hidden_dim, num_feats) — true feature directions."""
    return toy_model.embed.weight.detach()


# ─── Data generators ────────────────────────────────────────────────────────────

def generate_iid_sequences(
    num_seqs: int,
    T: int,
    p: float = FEAT_PROB,
    mu: float = FEAT_MEAN,
    sigma: float = FEAT_STD,
    num_feats: int = NUM_FEATS,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """(num_seqs, T, num_feats) — all i.i.d."""
    s = torch.bernoulli(torch.full((num_seqs, T, num_feats), p, device=device))
    m = (torch.randn(num_seqs, T, num_feats, device=device) * sigma + mu).abs()
    return s * m


def generate_markov_sequences(
    num_seqs: int,
    T: int,
    alpha: float = MARKOV_ALPHA,
    beta: float = MARKOV_BETA,
    mu: float = FEAT_MEAN,
    sigma: float = FEAT_STD,
    num_feats: int = NUM_FEATS,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Scheme C: 2-state Markov chain support. (num_seqs, T, num_feats)"""
    p_stat = beta / (1 - alpha + beta)
    s = torch.bernoulli(torch.full((num_seqs, num_feats), p_stat, device=device))
    support_list = [s.clone()]

    for _ in range(T - 1):
        p_next = torch.where(
            s == 1,
            torch.full_like(s, alpha),
            torch.full_like(s, beta),
        )
        s = torch.bernoulli(p_next)
        support_list.append(s.clone())

    support = torch.stack(support_list, dim=1)
    magnitudes = (torch.randn(num_seqs, T, num_feats, device=device) * sigma + mu).abs()
    return support * magnitudes


# Registry for easy lookup
DATASET_GENERATORS: dict[str, Callable] = {
    "iid": generate_iid_sequences,
    "markov": generate_markov_sequences,
}


def get_seq_gen_fn(dataset_name: str) -> Callable:
    """Return the sequence generator for a named dataset."""
    if dataset_name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_GENERATORS)}")
    return DATASET_GENERATORS[dataset_name]


# ─── Data iterators ─────────────────────────────────────────────────────────────

class ShuffledDataIterator:
    """Randomly sample ONE position from each sequence → marginal i.i.d."""

    def __init__(self, model: ToyModel, seq_gen_fn: Callable, batch_size: int, T: int):
        self.model = model
        self.seq_gen_fn = seq_gen_fn
        self.batch_size = batch_size
        self.T = T

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        feat_seqs = self.seq_gen_fn(self.batch_size, self.T)
        t_idx = torch.randint(0, self.T, (self.batch_size,))
        feats = feat_seqs[torch.arange(self.batch_size), t_idx]
        return self.model(feats)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


class SequentialWindowIterator:
    """Emit (B, T, d) activation windows for the crosscoder."""

    def __init__(self, model: ToyModel, seq_gen_fn: Callable, batch_size: int, T: int):
        self.model = model
        self.seq_gen_fn = seq_gen_fn
        self.batch_size = batch_size
        self.T = T

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        feat_seqs = self.seq_gen_fn(self.batch_size, self.T)
        return self.model(feat_seqs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
