"""Markov-chain support generator for C1 (toy TopK sweep).

Two-state Markov support per feature with stationary-distribution
init. ``rho_levels`` map to per-feature rho, distributed in equal-size
groups (4 features per level by default at C1's n=20, 5 levels).

Ports the Phase 2 Scheme C generator from
``origin/wasteland-canonical @ 94119bc0:src/data/toy/markov.py`` +
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
                         For deterministic emissions this equals the
                         hidden Markov state. For noisy emissions
                         (``p_A``/``p_B`` != defaults), this is the
                         OBSERVED support (used to build ``x``); the
                         hidden Markov state is in ``hidden_support``.
        per_feature_rho: ``(n_features,)`` rho assigned to each feature.
        hidden_support:  ``(n_seqs, n_features, T)`` the underlying
                         hidden Markov state (= ``support`` when
                         deterministic; differs from ``support`` only
                         when ``p_A>0`` or ``p_B<1``).
    """
    x: torch.Tensor
    features: torch.Tensor
    support: torch.Tensor
    per_feature_rho: torch.Tensor
    hidden_support: torch.Tensor


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
    p_A: float = 0.0,
    p_B: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> MarkovData:
    """Generate Markov-chain support data (C1 default).

    Args follow ``configs/datasources.yaml::toy_markov_n20_d40``.

    Bernoulli emission noise: when the hidden Markov state is OFF, the
    observation fires with probability ``p_A``; when ON, the observation
    fires with probability ``p_B``. Defaults (``p_A=0, p_B=1``) reproduce
    the deterministic ``support = hidden_state`` behavior (Phase 2
    Scheme C). Setting ``p_A=0, p_B=0.625`` reproduces the wasteland
    Phase 2 Experiment 1c noisy-emission setup
    (``docs/han/research_logs/2026-03-30-experiment1c-noisy-emissions.md``)
    used to test denoising — q=0.8 → γ=(1-q)/q=0.25 → p_B=0.625.
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
    h = torch.empty(n_seqs, n_features, seq_len)
    h[:, :, 0] = (u[:, :, 0] < pi_t.unsqueeze(0)).float()
    for t in range(1, seq_len):
        prev = h[:, :, t - 1]
        prob_on = prev * p_stay_on.unsqueeze(0) + (1 - prev) * p01.unsqueeze(0)
        h[:, :, t] = (u[:, :, t] < prob_on).float()

    # Bernoulli emission noise: s = observed support, h = hidden state.
    if p_A == 0.0 and p_B == 1.0:
        s = h                                          # deterministic shortcut
    else:
        u_obs = torch.rand(n_seqs, n_features, seq_len, generator=rng)
        # p_emit = p_A when h=0, p_B when h=1.
        p_emit = h * float(p_B) + (1.0 - h) * float(p_A)
        s = (u_obs < p_emit).float()

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
        hidden_support=h.to(device_t),
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
