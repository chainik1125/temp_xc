"""Setup L generator — magnitude-modulated locals (no globals in obs space).

agent_pro mission 2026-05-07T01:30Z (Han override): pure Effect 2 test.

Setup:
  - K_g slow Markov global chains (default ρ_g=0.95, π_g=0.05).
  - K_l locals fire i.i.d. with constant probability p_l (default 0.5);
    firing is INDEPENDENT of global state.
  - Local MAGNITUDE depends on global state via additive coupling:
        mag(j, t) = (base_mag + alpha · Σ_k C[j, k] · h_g[k, t])
                   * folded_normal(μ=1, σ=0.15)
    so when local j's parent global is on, its magnitude is boosted.
  - Observation: x(t) = Σ_j s_l[j,t] · mag(j,t) · f_l[j].
    NO global-direction term: f_g · h_g is NOT in observation space.

What this isolates:
  - Per-token observation looks like noisy magnitude-jittered local
    firings — no direction tracks the globals. The ONLY way to
    recover globals is to detect the slow temporal modulation of
    local-firing magnitudes, which requires aggregating across
    multiple tokens. Pure Effect 2 (temporal pattern detection).
  - gAUC for TXC should be high (window pool reveals slow magnitude
    trend); per-token SAE gAUC ≈ chance (sees only single magnitudes).

Returned ``CoupledData``:
  - ``hidden_features``  (K_global, d) — the global directions f_g
    (returned for gAUC eval; NOT used in the observation x).
  - ``emission_features``(K_local,  d) — local directions f_l.
  - ``coupling_matrix``  (K_local, K_global) — local→global parent map.
  - ``hidden_states``    (n_seqs, K_global, T) — slow chains.
  - ``emission_support`` (n_seqs, K_local, T) — fast i.i.d. firings.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def magnitude_modulated_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi_g: float = 0.05,
    rho_g: float = 0.95,
    p_l: float = 0.5,
    alpha: float = 1.0,
    base_mag: float = 1.0,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate the Setup L dataset.

    Defaults match
    ``configs/datasources.yaml::toy_magmod_Kg10_Kl30_d256_alpha1``.

    ``alpha`` is the strength of the global modulation on local
    magnitudes. ``alpha=0`` recovers a pure-noise baseline (locals are
    iid magnitude-noise; globals are uncorrelated with the observation).
    Larger ``alpha`` makes the temporal magnitude trend stronger and
    therefore easier for TXC to detect.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    n_total = K_global + K_local
    if n_total > d_in:
        raise ValueError(
            f"K_global + K_local = {n_total} exceeds d_in={d_in}; "
            f"cannot orthogonalise."
        )
    all_features = _orthogonalise(
        n_vectors=n_total, d_in=d_in, rng=rng,
    )                                                # (K_global+K_local, d)
    f_g = all_features[:K_global].contiguous()       # (K_global, d) — UNUSED in x
    f_l = all_features[K_global:].contiguous()       # (K_local,  d)

    # Coupling matrix (each local has n_global_parents parents).
    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0

    # K_global slow Markov chains.
    pi_t = torch.full((K_global,), float(pi_g))
    rho_t = torch.full((K_global,), float(rho_g))
    h_g = _markov_chain_batch(
        n_seqs=n_seqs, k=K_global, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )                                                # (n_seqs, K_global, T)

    # Local firings — i.i.d. Bernoulli(p_l), independent of globals.
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < float(p_l)).float()

    # Local magnitude: base_mag + alpha · (parent global state),
    # multiplied by per-cell folded-normal noise.
    parent_on_count = torch.einsum(
        "lk,nkt->nlt", C, h_g,
    )                                                # (n_seqs, K_local, T)
    mag_modulation = float(base_mag) + float(alpha) * parent_on_count
    mag_noise = _sample_magnitudes(
        shape=(n_seqs, K_local, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )
    mag_l = mag_modulation * mag_noise

    # Observation: only locals contribute. Globals are NOT in obs space.
    a_l = s_l * mag_l                                # (n_seqs, K_local, T)
    x = torch.einsum("nkt,kd->ntd", a_l, f_l)        # (n_seqs, T, d)

    return CoupledData(
        x=x.to(device_t),
        emission_features=f_l.to(device_t),
        hidden_features=f_g.to(device_t),
        coupling_matrix=C.to(device_t),
        hidden_states=h_g.to(device_t),
        emission_support=s_l.to(device_t),
    )
