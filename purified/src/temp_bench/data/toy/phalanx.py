"""Setup PHALANX generator — period-locked global pulses.

agent_pro follow-up to Setup K + L (mission 2026-05-07). The purest
TEST OF PHASE-DETECTION-via-window-pooling: globals fire on a
DETERMINISTIC period τ (no Markov stochastics). A single-token
observation cannot identify phase; only window pooling over T ≥ τ
can.

Setup:
  - K_g globals; global k is "on" at token t iff (t + φ_k) mod τ < δ.
    φ_k is a per-global random phase offset uniform in {0, ..., τ-1}.
    δ controls the duty cycle (default 1 → exactly 1/τ tokens on per
    global).
  - K_l locals fire iid p_l = 0.5 (independent of globals).
  - Local MAGNITUDE is modulated by parent global (same as Setup L):
        mag(j, t) = (base_mag + alpha · 1[parent of j is on])
                   · folded_normal(μ=1, σ=0.15)
  - Observation: x(t) = Σ_j s_l[j,t] · mag(j,t) · f_l[j].
    NO global-direction term — pure Effect 2.

Why this should favor TXC at the right T:
  - Per-token observation is statistically indistinguishable from
    Setup L's iid case (parent_on follows the same marginal at any
    given t), so per-token SAE gAUC ≈ chance.
  - But a window of length T ≥ τ contains each global "on"-state
    deterministically — TXC's window pool can detect the period.
  - Predicts a SHARP THRESHOLD at T = τ in the gAUC vs T curve, in
    contrast to Setup L's smooth monotone rise.

Returned ``CoupledData``:
  - ``hidden_features``  (K_global, d) — global directions f_g
    (NOT in observation; returned for gAUC eval).
  - ``emission_features``(K_local,  d) — local directions f_l.
  - ``coupling_matrix``  (K_local, K_global) — local→global parent map.
  - ``hidden_states``    (n_seqs, K_global, T) — periodic patterns.
  - ``emission_support`` (n_seqs, K_local, T) — fast iid firings.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _orthogonalise,
    _sample_magnitudes,
)


def phalanx_features(
    *,
    K_global: int = 10,
    K_local: int = 30,
    n_global_parents: int = 1,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    period: int = 8,
    duty_cycle: int = 1,
    p_l: float = 0.5,
    alpha: float = 1.0,
    base_mag: float = 1.0,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate the Setup PHALANX dataset.

    Defaults match
    ``configs/datasources.yaml::toy_phalanx_Kg10_Kl30_d256_period8``.
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

    # Per-global random phase offset; same across sequences (deterministic
    # mapping from t to global state given offsets).
    phase = torch.randint(0, int(period), (K_global,), generator=rng)
    t = torch.arange(seq_len)[None, :]                 # (1, T)
    h_g_per_global = (
        ((t + phase[:, None]) % int(period)) < int(duty_cycle)
    ).float()                                          # (K_global, T)
    # Broadcast to (n_seqs, K_global, T) — identical pattern across seqs.
    h_g = h_g_per_global.unsqueeze(0).expand(n_seqs, -1, -1).contiguous()

    # Local firings — iid Bernoulli(p_l), independent of globals.
    u = torch.rand(n_seqs, K_local, seq_len, generator=rng)
    s_l = (u < float(p_l)).float()

    # Local magnitude: base + alpha · (parent_on count).
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
