"""Setup chord generator — phase-locked global groups.

K_global globals organised into ``n_groups`` groups of equal size.
Within each group, all members fire/silence TOGETHER (perfectly
correlated via shared Markov chain). Between groups, chains are
independent.

Per-token: at each t, members of group g all carry the same on/off
state, so x(t) accumulates the SUM of all group members in their
direction f_g[k]. The PER-TOKEN SAE sees the GROUP SUM direction
(\sum_{k in group g} f_g[k]) — it can only learn ONE direction per
group, not the individual member directions.

TXC's window pool sees the TEMPORAL pattern of each group's on/off
firing, but the per-token aggregation is the same. So gAUC for TXC
should also be limited to GROUP-SUM directions.

HYPOTHESIS: at d_sae = K_global, both TXC and SAE recover at most
``n_groups`` directions (the group-sum directions). gAUC should
DROP for both, but EQUALLY — this is a mechanism CONSTRAINT, not a
TXC win. **This setup might NOT show TXC dominance** — keep if the
result is interesting (e.g. TXC handles group-sum recovery better),
drop if it's a flat tie.

Hash: paper-impactful as an honest exploration of when TXC/SAE
TIE rather than diverge.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def chord_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_groups: int = 2,
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
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Phase-locked global groups (n_groups groups of K_global//n_groups members)."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    if K_global % n_groups != 0:
        raise ValueError(f"K_global={K_global} not divisible by n_groups={n_groups}")
    members_per_group = K_global // n_groups

    n_total = K_global + K_local
    if n_total > d_in:
        raise ValueError(f"K_global+K_local={n_total} > d_in={d_in}")
    all_features = _orthogonalise(n_vectors=n_total, d_in=d_in, rng=rng)
    f_g = all_features[:K_global].contiguous()
    f_l = all_features[K_global:].contiguous()

    # Locals' parents in terms of group GLOBALS (random parent per local)
    C = torch.zeros(K_local, K_global)
    for j in range(K_local):
        parents = torch.randperm(K_global, generator=rng)[:n_global_parents]
        C[j, parents] = 1.0

    # Independent Markov chain per GROUP (n_groups chains)
    pi_t = torch.full((n_groups,), float(pi_g))
    rho_t = torch.full((n_groups,), float(rho_g))
    h_group = _markov_chain_batch(
        n_seqs=n_seqs, k=n_groups, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )
    # Replicate to per-global by repeating each group's chain
    # (n_seqs, n_groups, T) -> (n_seqs, K_global, T)
    h_g = h_group.repeat_interleave(members_per_group, dim=1)

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
