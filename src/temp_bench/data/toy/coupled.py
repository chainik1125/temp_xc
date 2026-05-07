"""Coupled-feature data generator for C2 (synthetic coupled HMM, OR-gate).

Ports the Phase 3 coupled-feature pipeline from
``origin/wasteland-canonical @ 94119bc0:src/data/toy/{coupled_dataset,
coupling}.py`` into a single ``temp_bench.data.toy`` module.

Architecture:

- ``K`` independent two-state Markov chains drive ``M > K`` emission
  features through a binary coupling matrix ``C ∈ {0,1}^{M×K}`` (each
  emission has exactly ``n_parents`` parents).
- Emission rule is OR-gate: emission ``j`` fires iff ANY of its parents
  is on. This dissociates *local* (emission-level) from *global*
  (hidden-state-level) feature recovery — per-token SAEs tend to
  converge to emission directions, while temporal-window archs can
  recover hidden directions.
- Magnitudes are sampled i.i.d. (folded normal, μ=1, σ=0.15 by default).
- Activations = magnitudes × emission_support; emissions get projected
  via orthogonal emission feature directions ``F ∈ R^{M×d}`` to produce
  observation vectors ``x_t = sum_j a_{j,t} F_j``.
- Hidden feature directions ``H ∈ R^{K×d}`` are computed as the
  normalised mean of children: ``H_i = normalize(sum_{j: C_ji=1} F_j)``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch


class CoupledData(NamedTuple):
    """Output of :func:`coupled_hmm`.

    Attributes:
        x:                 ``(n_seqs, T, d_in)`` observation vectors.
        emission_features: ``(M, d_in)`` orthogonal local directions.
        hidden_features:   ``(K, d_in)`` global directions (unit norm).
        coupling_matrix:   ``(M, K)`` binary coupling.
        hidden_states:     ``(n_seqs, K, T)`` binary hidden chains.
        emission_support:  ``(n_seqs, M, T)`` post-coupling emissions.
    """
    x: torch.Tensor
    emission_features: torch.Tensor
    hidden_features: torch.Tensor
    coupling_matrix: torch.Tensor
    hidden_states: torch.Tensor
    emission_support: torch.Tensor


def coupled_hmm(
    *,
    K_hidden: int = 10,
    M_emissions: int = 20,
    n_parents: int = 2,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi: float = 0.05,
    rho: float = 0.7,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate a coupled HMM dataset (C2 default).

    Output ``x`` has shape ``(n_seqs, seq_len, d_in)`` ready to feed a
    batch_iter that samples ``(B, seq_len, d_in)`` per call.

    Args follow ``configs/datasources.yaml::toy_coupled_K10_M20_d256``.
    ``seed`` controls the entire pipeline deterministically (coupling
    matrix, emission directions, hidden states, magnitudes).
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    # 1. Orthogonal emission feature directions F ∈ R^{M × d}.
    emission_features = _orthogonalise(
        n_vectors=M_emissions, d_in=d_in, rng=rng,
    )

    # 2. Binary coupling C ∈ {0,1}^{M × K}: each emission has n_parents.
    coupling_matrix = _generate_coupling(
        K=K_hidden, M=M_emissions, n_parents=n_parents, rng=rng,
    )

    # 3. Hidden feature directions H ∈ R^{K × d}: normalised parent means.
    hidden_features = _compute_hidden_features(
        emission_features, coupling_matrix,
    )

    # 4. K independent Markov chains × n_seqs sequences.
    pi_t = torch.full((K_hidden,), float(pi))
    rho_t = torch.full((K_hidden,), float(rho))
    hidden_states = _markov_chain_batch(
        n_seqs=n_seqs, k=K_hidden, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )                                                # (n_seqs, K, T)

    # 5. OR-gate coupling: s_j = 1[ sum_i C_ji * h_i >= 1 ].
    parent_sum = torch.einsum(
        "mk,nkt->nmt", coupling_matrix, hidden_states,
    )                                                # (n_seqs, M, T)
    emission_support = (parent_sum >= 1).float()

    # 6. Magnitudes (folded normal by default).
    magnitudes = _sample_magnitudes(
        shape=(n_seqs, M_emissions, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )

    # 7. Activations = support * magnitudes.
    activations = emission_support * magnitudes      # (n_seqs, M, T)

    # 8. Project: x_t = sum_j a_{j,t} F_j.
    # activations: (n, M, T), emission_features: (M, d) → x: (n, T, d).
    x = torch.einsum("nmt,md->ntd", activations, emission_features)

    return CoupledData(
        x=x.to(device_t),
        emission_features=emission_features.to(device_t),
        hidden_features=hidden_features.to(device_t),
        coupling_matrix=coupling_matrix.to(device_t),
        hidden_states=hidden_states.to(device_t),
        emission_support=emission_support.to(device_t),
    )


def make_batch_iter(
    data: CoupledData,
    *,
    seed: int = 0,
) -> Callable[[int], torch.Tensor]:
    """Sampling-with-replacement iterator over ``data.x``.

    Returns a callable ``(batch_size) → (B, seq_len, d_in)`` torch.float32
    tensor on the same device as ``data.x``.

    Determinism: ``np.random``-style RNG seeded by ``seed``; same seed
    + same ``data`` → same sampled batches. ``train_keys`` derived
    downstream depend on this contract.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n = data.x.shape[0]
    device = data.x.device

    def _iter(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, n, (batch_size,), generator=rng)
        return data.x[idx.to(device)].to(torch.float32)

    return _iter


# ── Internals ──────────────────────────────────────────────────────────────


def _orthogonalise(
    *, n_vectors: int, d_in: int, rng: torch.Generator,
) -> torch.Tensor:
    """Orthogonalise ``n_vectors`` random Gaussians into a (n, d) matrix.

    Uses Gram-Schmidt via QR. Each row has unit norm; rows are mutually
    orthogonal when n ≤ d_in. The wasteland's
    ``src/utils/orthogonalize.py`` accepts a ``target_cos_sim`` knob;
    we hardcode 0.0 (true orthogonal) for paper-faithful C2 setup.
    """
    g = torch.randn(d_in, n_vectors, generator=rng)
    q, _ = torch.linalg.qr(g, mode="reduced")        # (d_in, n_vectors)
    return q.T.contiguous()                          # (n_vectors, d_in)


def _generate_coupling(
    *, K: int, M: int, n_parents: int, rng: torch.Generator,
) -> torch.Tensor:
    if n_parents > K:
        raise ValueError(f"n_parents={n_parents} exceeds K={K}")
    C = torch.zeros(M, K)
    for j in range(M):
        parents = torch.randperm(K, generator=rng)[:n_parents]
        C[j, parents] = 1.0
    return C


def _compute_hidden_features(
    emission_features: torch.Tensor,
    coupling_matrix: torch.Tensor,
) -> torch.Tensor:
    """Hidden direction = normalised mean of children's emission directions.

    Args:
        emission_features: ``(M, d)``.
        coupling_matrix:   ``(M, K)``.
    Returns:
        ``(K, d)`` unit-norm rows.
    """
    hidden = coupling_matrix.T @ emission_features   # (K, d)
    norms = hidden.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return hidden / norms


def _markov_chain_batch(
    *, n_seqs: int, k: int, T: int,
    pi: torch.Tensor, rho: torch.Tensor, rng: torch.Generator,
) -> torch.Tensor:
    """Batch-generate ``n_seqs`` independent K-chain Markov supports.

    Per-chain transition: stationary marginal ``pi``, lag-1
    autocorrelation ``rho``. Initial state sampled from stationary
    distribution Bernoulli(``pi``).

    Returns ``(n_seqs, k, T)`` binary tensor on CPU.
    """
    # Transition probabilities derived from (pi, rho):
    #   P(s_{t+1}=1 | s_t=0) = pi (1 - rho)        (off → on)
    #   P(s_{t+1}=0 | s_t=1) = (1 - pi) (1 - rho)  (on → off)
    p01 = pi * (1.0 - rho)                            # (k,)
    p10 = (1.0 - pi) * (1.0 - rho)                    # (k,)
    p_stay_on  = 1.0 - p10                            # P(s=1|prev=1)
    p_stay_off = 1.0 - p01                            # P(s=0|prev=0)

    # Pre-sample all per-step uniforms. Shape (n_seqs, k, T).
    u = torch.rand(n_seqs, k, T, generator=rng)

    s = torch.empty(n_seqs, k, T)
    # Initial state ~ Bernoulli(pi): u[:,:,0] < pi -> 1.
    s[:, :, 0] = (u[:, :, 0] < pi.unsqueeze(0)).float()

    for t in range(1, T):
        prev = s[:, :, t - 1]                         # (n, k)
        # if prev=1: stay on with prob p_stay_on
        # if prev=0: turn on with prob p01
        prob_on = prev * p_stay_on.unsqueeze(0) + (1 - prev) * p01.unsqueeze(0)
        s[:, :, t] = (u[:, :, t] < prob_on).float()

    return s


def _sample_magnitudes(
    *, shape: tuple[int, ...], dist: str, mean: float, std: float,
    rng: torch.Generator,
) -> torch.Tensor:
    if dist == "folded_normal":
        raw = torch.randn(*shape, generator=rng) * std + mean
        return raw.abs()
    if dist == "half_normal":
        if mean != 0.0:
            raise ValueError(f"half_normal requires mean=0; got {mean}")
        return torch.randn(*shape, generator=rng).abs() * std
    raise ValueError(f"Unknown magnitude distribution: {dist}")
