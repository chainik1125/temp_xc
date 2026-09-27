"""Setup G generator — hierarchical bench (Setup E) + Gaussian obs noise.

Same K_g slow globals × K_l fast locals modulation as Setup E, but
the observation x(t) is corrupted by additive Gaussian noise σ. Tests
whether the global-vs-local divide (Setup E) and the denoising story
(Setup F) interact constructively.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def hierarchical_obs_noise_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi_g: float = 0.05,
    rho_g: float = 0.95,
    p_l_high: float = 0.8,
    p_l_low: float = 0.1,
    obs_noise_sigma: float = 1.0,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Hierarchical bench + observation noise. σ=0 → identical to Setup E."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    n_total = K_global + K_local
    if n_total > d_in:
        raise ValueError(f"K_global+K_local={n_total} > d_in={d_in}")
    all_features = _orthogonalise(n_vectors=n_total, d_in=d_in, rng=rng)
    f_g = all_features[:K_global].contiguous()
    f_l = all_features[K_global:].contiguous()

    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0

    pi_t = torch.full((K_global,), float(pi_g))
    rho_t = torch.full((K_global,), float(rho_g))
    h_g = _markov_chain_batch(
        n_seqs=n_seqs, k=K_global, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )
    parent_on_count = torch.einsum("lk,nkt->nlt", C, h_g)
    parent_on = (parent_on_count >= 1).float()
    p_local = parent_on * float(p_l_high) + (1 - parent_on) * float(p_l_low)
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < p_local).float()

    mag_l = _sample_magnitudes(
        shape=(n_seqs, K_local, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std, rng=rng,
    )
    mag_g = _sample_magnitudes(
        shape=(n_seqs, K_global, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std, rng=rng,
    )

    a_g = h_g * mag_g
    a_l = s_l * mag_l
    x_g = torch.einsum("nkt,kd->ntd", a_g, f_g)
    x_l = torch.einsum("nkt,kd->ntd", a_l, f_l)
    x = x_g + x_l
    if obs_noise_sigma > 0:
        x = x + torch.randn(*x.shape, generator=rng) * float(obs_noise_sigma)

    return CoupledData(
        x=x.to(device_t),
        emission_features=f_l.to(device_t),
        hidden_features=f_g.to(device_t),
        coupling_matrix=C.to(device_t),
        hidden_states=h_g.to(device_t),
        emission_support=s_l.to(device_t),
    )
