"""Setup aurora generator — coupled HMM with TEMPORALLY-CORRELATED noise.

Variant of Setup F (coupled + Gaussian obs noise). Instead of i.i.d.
Gaussian noise per token, the noise is an Ornstein-Uhlenbeck-style
process — auto-correlated over time. Hypothesis: TXC's window
averaging on the Setup F i.i.d. noise reduces noise by √T; on
auto-correlated noise the window-averaged noise still has structure.

Implementation: x(t) = x_clean(t) + ε(t), where
    ε(t) = α · ε(t-1) + sqrt(1 - α²) · σ · z(t)
with z(t) ~ N(0, I_d) i.i.d., α the noise auto-correlation.

α=0 → identical to Setup F (i.i.d. Gaussian noise).
α=0.9 → strongly auto-correlated noise.
"""

from __future__ import annotations

import math
import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _generate_coupling,
    _compute_hidden_features,
    _sample_magnitudes,
)


def aurora_features(
    *,
    K_hidden: int = 10,
    M_emissions: int = 20,
    n_parents: int = 2,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi: float = 0.05,
    rho: float = 0.7,
    obs_noise_sigma: float = 1.0,
    noise_alpha: float = 0.9,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Coupled HMM + auto-correlated Gaussian observation noise."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    emission_features = _orthogonalise(
        n_vectors=M_emissions, d_in=d_in, rng=rng,
    )
    coupling_matrix = _generate_coupling(
        K=K_hidden, M=M_emissions, n_parents=n_parents, rng=rng,
    )
    hidden_features = _compute_hidden_features(
        emission_features, coupling_matrix,
    )

    pi_t = torch.full((K_hidden,), float(pi))
    rho_t = torch.full((K_hidden,), float(rho))
    hidden_states = _markov_chain_batch(
        n_seqs=n_seqs, k=K_hidden, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )
    parent_sum = torch.einsum(
        "mk,nkt->nmt", coupling_matrix, hidden_states,
    )
    emission_support = (parent_sum >= 1).float()

    magnitudes = _sample_magnitudes(
        shape=(n_seqs, M_emissions, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std, rng=rng,
    )
    activations = emission_support * magnitudes
    x_clean = torch.einsum("nmt,md->ntd", activations, emission_features)

    # Auto-correlated noise (OU process)
    eps = torch.zeros(n_seqs, seq_len, d_in)
    z = torch.randn(n_seqs, seq_len, d_in, generator=rng) * float(obs_noise_sigma)
    alpha = float(noise_alpha)
    sqrt_1_minus_a2 = math.sqrt(max(1.0 - alpha * alpha, 0.0))
    eps[:, 0] = z[:, 0]
    for t in range(1, seq_len):
        eps[:, t] = alpha * eps[:, t - 1] + sqrt_1_minus_a2 * z[:, t]

    x = x_clean + eps

    return CoupledData(
        x=x.to(device_t),
        emission_features=emission_features.to(device_t),
        hidden_features=hidden_features.to(device_t),
        coupling_matrix=coupling_matrix.to(device_t),
        hidden_states=hidden_states.to(device_t),
        emission_support=emission_support.to(device_t),
    )
