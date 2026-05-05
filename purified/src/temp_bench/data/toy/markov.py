"""Markov-chain support generator for C1 (toy TopK sweep).

Two-state Markov support per feature with stationary-distribution
init. ``rho_levels`` map to per-feature rho, distributed in equal-size
groups (4 features per level by default at C1's n=20, 5 levels).

Ports the Phase 2 Scheme C generator from
``origin/han-phase7-unification @ 94119bc0:src/data/toy/markov.py`` +
``support.py`` into a single self-contained module.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch


class MarkovData(NamedTuple):
    """Output of :func:`markov_chain_support`.

    Attributes:
        x:               ``(n_seqs, T, d_in)`` observation vectors.
        features:        ``(n_features, d_in)`` orthogonal directions.
        support:         ``(n_seqs, n_features, T)`` binary support.
        per_feature_rho: ``(n_features,)`` rho assigned to each feature.
    """
    x: torch.Tensor
    features: torch.Tensor
    support: torch.Tensor
    per_feature_rho: torch.Tensor


def markov_chain_support(
    *,
    n_features: int = 20,
    d_in: int = 40,
    seq_len: int = 64,
    n_seqs: int = 4096,
    rho_levels: list[float] | None = None,
    pi: float = 0.5,
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> MarkovData:
    """Generate Markov-chain support data (C1 default).

    Args follow ``configs/datasources.yaml::toy_markov_n20_d40``.
    """
    if rho_levels is None:
        rho_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    # Per-feature rho assignment: equal-size groups across rho_levels.
    n_levels = len(rho_levels)
    if n_features % n_levels != 0:
        raise ValueError(
            f"n_features={n_features} must divide evenly across "
            f"{n_levels} rho levels."
        )
    per_level = n_features // n_levels
    rho_t = torch.cat([
        torch.full((per_level,), float(r)) for r in rho_levels
    ])
    pi_t = torch.full((n_features,), float(pi))

    # Orthogonal feature directions.
    g = torch.randn(d_in, n_features, generator=rng)
    q, _ = torch.linalg.qr(g, mode="reduced")
    features = q.T.contiguous()                       # (n_features, d_in)

    # Generate per-seq Markov supports.
    p01 = pi_t * (1.0 - rho_t)
    p10 = (1.0 - pi_t) * (1.0 - rho_t)
    p_stay_on = 1.0 - p10

    u = torch.rand(n_seqs, n_features, seq_len, generator=rng)
    s = torch.empty(n_seqs, n_features, seq_len)
    s[:, :, 0] = (u[:, :, 0] < pi_t.unsqueeze(0)).float()
    for t in range(1, seq_len):
        prev = s[:, :, t - 1]
        prob_on = prev * p_stay_on.unsqueeze(0) + (1 - prev) * p01.unsqueeze(0)
        s[:, :, t] = (u[:, :, t] < prob_on).float()

    # Magnitudes (folded normal, mean=1, std=0.15 — same convention as C2).
    raw = torch.randn(n_seqs, n_features, seq_len, generator=rng) * magnitude_std + magnitude_mean
    magnitudes = raw.abs()
    activations = s * magnitudes

    # Project: x_t = sum_i a_{i,t} f_i.
    x = torch.einsum("nft,fd->ntd", activations, features)

    return MarkovData(
        x=x.to(device_t),
        features=features.to(device_t),
        support=s.to(device_t),
        per_feature_rho=rho_t.to(device_t),
    )


def make_batch_iter(
    data: MarkovData,
    *,
    seed: int = 0,
) -> Callable[[int], torch.Tensor]:
    """Sampling-with-replacement iterator over ``data.x``."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n = data.x.shape[0]
    device = data.x.device

    def _iter(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, n, (batch_size,), generator=rng)
        return data.x[idx.to(device)].to(torch.float32)

    return _iter
