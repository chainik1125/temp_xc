"""Hierarchical features data generator for C2 Phase 3 (ENGINEER).

agent_synth mission 2026-05-06T23:00Z (Han override): build a generator
*engineered* for the global/local divide. K_g slow global chains
modulate K_l fast local features; the per-token signal is dominated by
locals (K_l > K_g), but the global structure is only recoverable by
averaging across the time axis (= what TXC does).

Setup:
  - K_g global slow chains, ρ_g (default 0.95), π_g (default 0.05).
  - K_l local features. Each local has ``n_global_parents`` global
    chains as parents (default 1). When ANY parent is on, the local
    fires with prob ``p_l_high`` (default 0.8); when all parents are
    off, the local fires with prob ``p_l_low`` (default 0.1). Locals
    are otherwise i.i.d. across tokens (no temporal correlation).
  - Observation: ``x(t) = Σ h_g[i](t) · f_g[i] + Σ s_l[j](t) · f_l[j]``
    where ``f_g, f_l`` are orthogonal directions in ``R^d``.

Two ground-truth direction sets returned via ``CoupledData``:
  - ``hidden_features`` (K_g, d) — the GLOBAL directions ``f_g``.
  - ``emission_features`` (K_l, d) — the LOCAL directions ``f_l``.

Why this favors TXC:
  - The per-token signal is dominated by the K_l local contributions
    (K_l > K_g). A per-token SAE optimizing reconstruction will allocate
    its budget to the locals (`f_l`). Its dictionary aligns with locals.
  - Globals contribute slowly + sparsely; only window-pooled views see
    consistent global activations across multiple tokens. TXC's
    encoder pools T tokens and finds the slow consistent direction —
    its dictionary aligns with globals.
  - gAUC (against `f_g`) → TXC > SAE. eAUC (against `f_l`) → SAE > TXC.

This is "Narrative D" from agent_filler's brainstorm — quantify the
global/local axis.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def hierarchical_features(
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
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    global_magnitude_scale: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate a hierarchical-features dataset (slow globals × fast locals).

    Returns a ``CoupledData`` namedtuple with:
      - ``emission_features``: (K_local, d) — local directions f_l.
      - ``hidden_features``:   (K_global, d) — global directions f_g.
      - ``coupling_matrix``:   (K_local, K_global) — local→global parent map.
      - ``hidden_states``:     (n_seqs, K_global, T) — slow chains.
      - ``emission_support``:  (n_seqs, K_local, T) — fast (modulated) firings.
      - ``x``:                 (n_seqs, T, d) — observation = globals + locals.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    # 1. Orthogonal directions for K_global + K_local features.
    n_total = K_global + K_local
    if n_total > d_in:
        raise ValueError(
            f"K_global + K_local = {n_total} exceeds d_in={d_in}; "
            f"cannot orthogonalise."
        )
    all_features = _orthogonalise(
        n_vectors=n_total, d_in=d_in, rng=rng,
    )                                                # (K_global+K_local, d)
    f_g = all_features[:K_global].contiguous()       # (K_global, d)
    f_l = all_features[K_global:].contiguous()       # (K_local, d)

    # 2. Coupling matrix C ∈ {0,1}^{K_local × K_global}: each local has
    # n_global_parents global parents (uniform random).
    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0

    # 3. K_global independent slow Markov chains.
    pi_t = torch.full((K_global,), float(pi_g))
    rho_t = torch.full((K_global,), float(rho_g))
    h_g = _markov_chain_batch(
        n_seqs=n_seqs, k=K_global, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )                                                # (n_seqs, K_global, T)

    # 4. Local firings — modulated Bernoulli per token.
    # parent_on[n, j, t] = 1 if any parent of local j is on at time t.
    parent_on_count = torch.einsum(
        "lk,nkt->nlt", C, h_g,
    )                                                # (n_seqs, K_local, T)
    parent_on = (parent_on_count >= 1).float()
    # p_local[n, j, t] = p_l_high if parent_on else p_l_low.
    p_local = parent_on * float(p_l_high) + (1 - parent_on) * float(p_l_low)
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < p_local).float()                      # (n_seqs, K_local, T)

    # 5. Magnitudes for the K_local locals + K_global globals.
    mag_l = _sample_magnitudes(
        shape=(n_seqs, K_local, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )
    mag_g = _sample_magnitudes(
        shape=(n_seqs, K_global, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )

    # 6. Observation: x(t) = Σ_i h_g[i] m_g[i] f_g[i] + Σ_j s_l[j] m_l[j] f_l[j].
    a_g = h_g * mag_g * float(global_magnitude_scale)  # (n_seqs, K_global, T)
    a_l = s_l * mag_l                                # (n_seqs, K_local, T)
    x_g = torch.einsum("nkt,kd->ntd", a_g, f_g)      # (n_seqs, T, d)
    x_l = torch.einsum("nkt,kd->ntd", a_l, f_l)      # (n_seqs, T, d)
    x = x_g + x_l

    return CoupledData(
        x=x.to(device_t),
        emission_features=f_l.to(device_t),
        hidden_features=f_g.to(device_t),
        coupling_matrix=C.to(device_t),
        hidden_states=h_g.to(device_t),
        emission_support=s_l.to(device_t),
    )
