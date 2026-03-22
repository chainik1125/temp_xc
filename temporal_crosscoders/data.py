"""
data.py — Toy model, synthetic data generators, and data iterators.

Data generation is parametrized by rho (lag-1 autocorrelation):
  rho = 0  → IID (no temporal structure)
  rho > 0  → 2-state Markov chain with temporal persistence
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from typing import Any

from config import (
    NUM_FEATS, HIDDEN_DIM, FEAT_PROB, FEAT_MEAN, FEAT_STD,
    DEVICE, markov_params,
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


# ─── Unified sequence generator ─────────────────────────────────────────────────

def generate_sequences(
    num_seqs: int,
    T: int,
    rho: float,
    p_stat: float = FEAT_PROB,
    mu: float = FEAT_MEAN,
    sigma: float = FEAT_STD,
    num_feats: int = NUM_FEATS,
    device: torch.device = DEVICE,
    p_A: float = 0.0,
    p_B: float = 1.0,
) -> torch.Tensor:
    """
    Generate (num_seqs, T, num_feats) feature sequences.

    rho = 0:   IID Bernoulli support (no temporal structure).
    rho > 0:   2-state Markov chain support with lag-1 autocorrelation = rho.

    The stationary firing probability is always p_stat regardless of rho.

    HMM emissions: the hidden chain state z_t emits s_t ~ Bernoulli(p_{z_t})
    where p_A is the emission probability in the OFF state and p_B in the ON
    state. With defaults (p_A=0, p_B=1), the observation equals the hidden
    state (deterministic emission, recovering the original MC).
    """
    alpha, beta = markov_params(rho, p_stat)

    # First timestep: draw hidden state from stationary distribution
    z = torch.bernoulli(torch.full((num_seqs, num_feats), p_stat, device=device))
    hidden_list = [z.clone()]

    for _ in range(T - 1):
        if rho == 0.0:
            # Pure IID — no dependence on previous state
            z = torch.bernoulli(torch.full_like(z, p_stat))
        else:
            p_next = torch.where(
                z == 1,
                torch.full_like(z, alpha),
                torch.full_like(z, beta),
            )
            z = torch.bernoulli(p_next)
        hidden_list.append(z.clone())

    hidden = torch.stack(hidden_list, dim=1)  # (N, T, F)

    # HMM emission step: sample observed support from hidden states
    if p_A == 0.0 and p_B == 1.0:
        support = hidden
    else:
        emission_probs = torch.where(
            hidden > 0.5,
            torch.tensor(p_B, device=device),
            torch.tensor(p_A, device=device),
        )
        support = torch.bernoulli(emission_probs)

    magnitudes = (torch.randn(num_seqs, T, num_feats, device=device) * sigma + mu).abs()
    return support * magnitudes


# ─── Cached data source ─────────────────────────────────────────────────────────

from config import NUM_CHAINS, CHAIN_LENGTH, CACHE_REFRESH_INTERVAL


class CachedDataSource:
    """
    Pre-generate long feature chains → toy model activations.
    Parametrized by rho (correlation level) and optional HMM emission
    probabilities (p_A, p_B).

    Stored as:
        feat_chains: (NUM_CHAINS, CHAIN_LENGTH, NUM_FEATS)
        act_chains:  (NUM_CHAINS, CHAIN_LENGTH, HIDDEN_DIM)
    """

    def __init__(
        self,
        rho: float,
        toy_model: ToyModel,
        num_chains: int = NUM_CHAINS,
        chain_length: int = CHAIN_LENGTH,
        device: torch.device = DEVICE,
        p_A: float = 0.0,
        p_B: float = 1.0,
    ):
        self.rho = rho
        self.p_A = p_A
        self.p_B = p_B
        self.toy_model = toy_model
        self.num_chains = num_chains
        self.chain_length = chain_length
        self.device = device

        self._generate()

    @torch.no_grad()
    def _generate(self):
        """Generate long chains and cache activations."""
        self.feat_chains = generate_sequences(
            self.num_chains, self.chain_length, self.rho,
            p_A=self.p_A, p_B=self.p_B,
        )

        chunk_size = 32
        act_chunks = []
        for i in range(0, self.num_chains, chunk_size):
            chunk = self.feat_chains[i : i + chunk_size].to(self.device)
            act_chunks.append(self.toy_model(chunk))
        self.act_chains = torch.cat(act_chunks, dim=0)  # (N, L, d)

    def refresh(self):
        """Regenerate all chains."""
        self._generate()

    def sample_windows(self, batch_size: int, T: int) -> torch.Tensor:
        """Sample random sliding windows → (B, T, d)."""
        max_start = self.chain_length - T
        chain_idx = torch.randint(0, self.num_chains, (batch_size,))
        start_idx = torch.randint(0, max_start + 1, (batch_size,))

        offsets = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        pos_idx = start_idx.to(self.device).unsqueeze(1) + offsets   # (B, T)
        chain_idx_dev = chain_idx.to(self.device)
        chain_exp = chain_idx_dev.unsqueeze(1).expand(-1, T)
        return self.act_chains[chain_exp, pos_idx]  # (B, T, d)


# ─── Iterator backed by cache ───────────────────────────────────────────────────

class CachedWindowIterator:
    """Yields (B, T, d) window batches from a CachedDataSource."""

    def __init__(self, cache: CachedDataSource, batch_size: int, T: int):
        self.cache = cache
        self.batch_size = batch_size
        self.T = T
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        self.step += 1
        if CACHE_REFRESH_INTERVAL > 0 and self.step % CACHE_REFRESH_INTERVAL == 0:
            self.cache.refresh()
        return self.cache.sample_windows(self.batch_size, self.T)
