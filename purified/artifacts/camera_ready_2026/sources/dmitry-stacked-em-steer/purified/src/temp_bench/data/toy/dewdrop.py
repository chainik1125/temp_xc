"""Setup dewdrop generator — globals fire at predictable intervals.

Each global k fires DETERMINISTICALLY every ``period`` tokens, with phase
``k * stride mod period`` (rolling phases). NOT Markov; no randomness in
firing schedule. Locals are modulated by parents as in Setup E.

Per-token observation: at each t, exactly ``period//stride`` globals are
on (those with the right phase). Per-token SAE sees this as a fixed
co-firing pattern across tokens; TXC's window pool over T tokens
captures the ROTATION of which globals are on, recovering the periodic
structure.

Hypothesis: TXC's window-pooled dictionary aligns with global directions
once T spans a full period; per-token SAE's dictionary only captures
the joint co-firing pattern at any single token.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _orthogonalise,
    _sample_magnitudes,
)


def dewdrop_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    period: int = 16,
    stride: int = 1,
    p_l_high: float = 0.8,
    p_l_low: float = 0.1,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Periodic-firing globals + modulated locals.

    Global k fires at t where (t - k*stride) % period == 0. With default
    period=16, stride=1: globals 0..9 fire at offsets 0, 1, 2, ..., 9
    relative to each period's start; only K_global of the 16 phase slots
    are used.
    """
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

    # Deterministic global firing schedule.
    # h_g[n, k, t] = 1 if (t - k*stride) % period == 0 else 0
    t_idx = torch.arange(seq_len)
    h_g = torch.zeros(n_seqs, K_global, seq_len)
    for k in range(K_global):
        offset = (k * stride) % period
        fires = ((t_idx - offset) % period == 0).float()
        h_g[:, k, :] = fires.unsqueeze(0).expand(n_seqs, -1)

    # Locals modulated by parents (Setup E style)
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
