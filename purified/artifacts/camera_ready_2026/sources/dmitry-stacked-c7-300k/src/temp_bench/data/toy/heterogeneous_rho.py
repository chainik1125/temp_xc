"""Setup M generator — heterogeneous-ρ globals (slow + fast mixed).

Same hierarchical structure as Setup E (K_g globals modulating K_l
locals via OR-gate parents) BUT each global chain has its OWN ρ value.
By default: 5 SLOW (ρ=0.99) + 5 FAST (ρ=0.0, i.i.d. Bernoulli) globals.

Mechanism: TXC's window pooling over T tokens averages out the FAST
globals (random firings cancel) and PRESERVES the SLOW globals
(consistent firings reinforce). Per-token SAEs see all globals as
equally-firing single events and don't differentiate by timescale.

Hypothesis: TXC's gAUC on slow-half features is HIGHER than on
fast-half; per-token SAE shows no slow/fast preference.

Returned CoupledData:
  - hidden_features: (K_global, d) — global directions f_g (in order
    matching rho_g_list: slow features first, then fast).
  - emission_features: (K_local, d) — local directions f_l.

The eval pipeline measures gAUC across ALL globals; for paper plot
we'd want a "slow-half gAUC" stratification, but the global-AUC
trend itself is informative.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _orthogonalise,
    _sample_magnitudes,
)


def _markov_chain_per_feature_rho(
    *, n_seqs: int, K: int, T: int,
    pi: torch.Tensor, rho: torch.Tensor, rng,
):
    """K independent 2-state Markov chains, each with its own ρ.

    pi: (K,) per-chain stationary on-probability.
    rho: (K,) per-chain stay-on / stay-off probability.

    Steady state: pi_on = pi (we sample initial state ∼ Bernoulli(pi)).
    Transition: P(stay) = rho; P(switch) = (1-rho).

    Returns h: (n_seqs, K, T) ∈ {0, 1}.
    """
    h = torch.zeros(n_seqs, K, T)
    # Initial state ∼ Bernoulli(pi)
    init_u = torch.rand(n_seqs, K, generator=rng)
    h[:, :, 0] = (init_u < pi.unsqueeze(0)).float()
    # Step forward t=1..T-1
    for t in range(1, T):
        u = torch.rand(n_seqs, K, generator=rng)
        prev = h[:, :, t - 1]
        # Stay with prob rho, switch with prob (1-rho).
        # If prev=1 and stay: still 1; if prev=1 and switch: 0.
        # If prev=0 and stay: still 0; if prev=0 and switch: 1.
        stay = (u < rho.unsqueeze(0)).float()
        h[:, :, t] = prev * stay + (1.0 - prev) * (1.0 - stay)
    return h


def heterogeneous_rho_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi_g: float = 0.5,
    rho_g_list: list[float] | None = None,
    p_l_high: float = 0.8,
    p_l_low: float = 0.1,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Heterogeneous-ρ hierarchical bench.

    rho_g_list: list of length K_global with per-chain ρ values. If
    None, defaults to half slow (ρ=0.99) + half fast (ρ=0.0).

    pi_g: stationary on-probability per global chain (default 0.5 to
    keep slow + fast both at ~50% activity, isolating the ρ axis).
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    if rho_g_list is None:
        # Default: half slow (ρ=0.99) + half fast (ρ=0.0). Slow first.
        n_slow = K_global // 2
        rho_g_list = [0.99] * n_slow + [0.0] * (K_global - n_slow)
    if len(rho_g_list) != K_global:
        raise ValueError(
            f"rho_g_list length {len(rho_g_list)} != K_global={K_global}"
        )

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
    rho_t = torch.tensor([float(r) for r in rho_g_list])
    h_g = _markov_chain_per_feature_rho(
        n_seqs=n_seqs, K=K_global, T=seq_len,
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

    return CoupledData(
        x=x.to(device_t),
        emission_features=f_l.to(device_t),
        hidden_features=f_g.to(device_t),
        coupling_matrix=C.to(device_t),
        hidden_states=h_g.to(device_t),
        emission_support=s_l.to(device_t),
    )
