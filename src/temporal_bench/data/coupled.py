"""Coupled-features generative model (Aniket Level 3 / Dmitry exp1c3).

K hidden Markov chains drive M > K emission features through a binary
coupling matrix C in {0,1}^{M x K}, with each emission having exactly
n_parents parent hidden chains. Each emission fires (s_j(t)=1) iff at
least one of its parents is on (deterministic OR gate). The observation
x_t is then the standard sparse linear combination of M emission
features.

This dissociates two notions of "feature direction" in observation
space:
  - emission_features : (M, d) — the directions actually present in x
  - hidden_features   : (K, d) — direction of each hidden chain,
                                  defined as the normalized sum of
                                  emission directions it controls

A token-local SAE can in principle recover the M emission directions
without ever recovering the K hidden directions. Recovering the hidden
directions requires inverting the coupling, which is ill-posed at any
single token (n_parents > 1 conflates information across chains) and
becomes solvable only with temporal context — which is the regime
TXCDR is supposed to exploit.

Reference: docs/aniket/coupled_features_plan.md on the dmitry-synthetic
branch; src/data_generation/coupling.py at that revision.
"""

from __future__ import annotations

import torch


def generate_coupling_matrix(
    K: int, M: int, n_parents: int, generator: torch.Generator
) -> torch.Tensor:
    """Random binary coupling matrix.

    Each emission row has exactly n_parents ones at distinct columns;
    parent assignments are independent across emissions.

    Returns:
        (M, K) float tensor with values in {0, 1}.
    """
    if n_parents > K:
        raise ValueError(f"n_parents={n_parents} exceeds K={K}")
    if n_parents <= 0:
        raise ValueError(f"n_parents must be positive, got {n_parents}")
    C = torch.zeros(M, K)
    for j in range(M):
        parents = torch.randperm(K, generator=generator)[:n_parents]
        C[j, parents] = 1.0
    return C


def apply_coupling_or(
    hidden_states: torch.Tensor, coupling_matrix: torch.Tensor
) -> torch.Tensor:
    """OR gate: emission j fires iff any of its parent hidden chains is on.

    Args:
        hidden_states: (..., K, T) binary tensor.
        coupling_matrix: (M, K) binary tensor.

    Returns:
        (..., M, T) float tensor in {0, 1} — emission support s.
    """
    parent_sum = torch.einsum("mk,...kt->...mt", coupling_matrix, hidden_states)
    return (parent_sum >= 1).float()


def compute_hidden_features(
    emission_features: torch.Tensor, coupling_matrix: torch.Tensor
) -> torch.Tensor:
    """Aggregate hidden-chain feature directions from emission features.

    For each hidden chain k, the hidden direction is the unit-norm sum of
    emission directions it parents:
        hidden_features[k] = normalize(sum_{j: C[j,k] = 1} emission_features[j])

    Args:
        emission_features: (M, d) emission feature directions.
        coupling_matrix: (M, K) binary coupling matrix.

    Returns:
        (K, d) unit-norm hidden feature directions.
    """
    hidden_dirs = coupling_matrix.T @ emission_features  # (K, d)
    norms = hidden_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return hidden_dirs / norms
