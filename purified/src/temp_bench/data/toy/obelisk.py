"""Setup OBELISK generator — sparse + amplified magnitude-mod (rare events).

agent_pro follow-up to Setup K + L (mission 2026-05-07). A Setup-L
variant where local firings are RARE (p_l = 0.05) but each firing is
DRAMATICALLY amplified by global state (α = 5-10×). Tests TXC on
rare-event temporal patterns: per-token signal is mostly zero; the
information content is concentrated in the few magnitude spikes.

Setup:
  - K_g slow Markov globals (ρ_g = 0.95, π_g = 0.05; same as L).
  - K_l locals fire iid p_l = 0.05 (rare; expected ~3 firings per
    seq_len=64 token sequence).
  - When local j fires AND its parent global is on, magnitude is
    amplified by ``alpha`` (default 5). Otherwise magnitude is
    folded_normal(μ=1, σ=0.15) ≈ 1.
  - Observation: x(t) = Σ_j s_l[j,t] · mag(j,t) · f_l[j].
    NO global-direction term — pure Effect 2.

Why this matters:
  - Setup L (p_l = 0.5, α = 1) has dense, gentle magnitude modulation:
    every token has ~15 active locals, each magnitude differs by ~50%.
    The modulation is detectable but spread thin.
  - OBELISK concentrates the signal: most tokens are pure noise; a
    few scattered tokens have huge amplitude bursts that encode global
    state. TXC must correctly attribute the amplitude bursts to a
    slow global rather than treating them as outlier locals.
  - Predicts: TXC gAUC depends on whether T contains enough firings
    to detect amplitude trend; per-token SAE gAUC ≈ chance.

Returned ``CoupledData``:
  - ``hidden_features``  (K_global, d) — global directions f_g
    (NOT in observation; returned for gAUC eval).
  - ``emission_features``(K_local,  d) — local directions f_l.
  - ``coupling_matrix``  (K_local, K_global) — local→global parent map.
  - ``hidden_states``    (n_seqs, K_global, T) — slow chains.
  - ``emission_support`` (n_seqs, K_local, T) — sparse iid firings.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def obelisk_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi_g: float = 0.05,
    rho_g: float = 0.95,
    p_l: float = 0.05,
    alpha: float = 5.0,
    base_mag: float = 1.0,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate the Setup OBELISK dataset.

    Defaults match
    ``configs/datasources.yaml::toy_obelisk_Kg10_Kl30_d256_alpha5``.
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
    )
    f_g = all_features[:K_global].contiguous()
    f_l = all_features[K_global:].contiguous()

    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0

    # Slow Markov globals (same primitive as Setup L).
    pi_t = torch.full((K_global,), float(pi_g))
    rho_t = torch.full((K_global,), float(rho_g))
    h_g = _markov_chain_batch(
        n_seqs=n_seqs, k=K_global, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )

    # Sparse iid local firings.
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < float(p_l)).float()

    # Magnitude: base + alpha · 1[parent on], times folded-normal noise.
    parent_on_count = torch.einsum("lk,nkt->nlt", C, h_g)
    mag_modulation = float(base_mag) + float(alpha) * parent_on_count
    mag_noise = _sample_magnitudes(
        shape=(n_seqs, K_local, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )
    mag_l = mag_modulation * mag_noise

    a_l = s_l * mag_l
    x = torch.einsum("nkt,kd->ntd", a_l, f_l)

    return CoupledData(
        x=x.to(device_t),
        emission_features=f_l.to(device_t),
        hidden_features=f_g.to(device_t),
        coupling_matrix=C.to(device_t),
        hidden_states=h_g.to(device_t),
        emission_support=s_l.to(device_t),
    )
