"""Coupled features: many-to-many mapping from hidden states to emissions.

K hidden states produce M > K emission features through a binary coupling
matrix C in {0,1}^{M x K}. This dissociates local (emission-level) from
global (hidden-state-level) feature recovery.

See docs/aniket/coupled_features_plan.md for full math.
"""

import torch


def generate_coupling_matrix(
    K: int,
    M: int,
    n_parents: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate a random binary coupling matrix.

    Each emission j has exactly n_parents parent hidden states.
    Parents are assigned uniformly at random (with replacement across emissions,
    but without replacement within a single emission).

    Args:
        K: Number of hidden states.
        M: Number of emission features.
        n_parents: Number of parent hidden states per emission.
        rng: Torch random number generator.

    Returns:
        Binary tensor of shape (M, K).
    """
    if n_parents > K:
        raise ValueError(f"n_parents={n_parents} exceeds K={K}")

    C = torch.zeros(M, K)
    for j in range(M):
        # Sample n_parents distinct parents for emission j
        parents = torch.randperm(K, generator=rng)[:n_parents]
        C[j, parents] = 1.0
    return C


def apply_coupling_or(
    hidden_states: torch.Tensor,
    coupling_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply OR-gate coupling: emission fires if ANY parent is on.

    s_j(t) = 1[ sum_i C_{ji} * h_i(t) >= 1 ]

    Args:
        hidden_states: (n_seq, K, T) or (K, T) binary hidden states.
        coupling_matrix: (M, K) binary coupling matrix.

    Returns:
        Binary tensor of shape (..., M, T) emission support.
    """
    # hidden: (..., K, T), C: (M, K) -> sum: (..., M, T)
    parent_sum = torch.einsum("mk,...kt->...mt", coupling_matrix, hidden_states)
    return (parent_sum >= 1).float()


def apply_coupling_sigmoid(
    hidden_states: torch.Tensor,
    coupling_matrix: torch.Tensor,
    alpha: float,
    beta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Apply sigmoid coupling: emission probability is a soft function of parents.

    s_j(t) ~ Bernoulli(sigmoid(alpha * sum_i C_{ji} * h_i(t) + beta))

    Args:
        hidden_states: (..., K, T) binary hidden states.
        coupling_matrix: (M, K) binary coupling matrix.
        alpha: Sharpness parameter (higher = more deterministic).
        beta: Bias (negative = sparser emissions).
        rng: Torch random number generator.

    Returns:
        Binary tensor of shape (..., M, T) emission support.
    """
    parent_sum = torch.einsum("mk,...kt->...mt", coupling_matrix, hidden_states)
    logits = alpha * parent_sum + beta
    probs = torch.sigmoid(logits)
    noise = torch.rand_like(probs, generator=rng if probs.device.type == "cpu" else None)
    return (noise < probs).float()


def apply_coupling(
    hidden_states: torch.Tensor,
    coupling_matrix: torch.Tensor,
    mode: str = "or",
    alpha: float = 5.0,
    beta: float = -2.0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply coupling to map hidden states to emission support.

    Args:
        hidden_states: (..., K, T) binary hidden states.
        coupling_matrix: (M, K) binary coupling matrix.
        mode: "or" for deterministic OR gate, "sigmoid" for soft coupling.
        alpha: Sigmoid sharpness (ignored for mode="or").
        beta: Sigmoid bias (ignored for mode="or").
        rng: Torch RNG (required for mode="sigmoid").

    Returns:
        Binary tensor of shape (..., M, T).
    """
    if mode == "or":
        return apply_coupling_or(hidden_states, coupling_matrix)
    elif mode == "sigmoid":
        if rng is None:
            raise ValueError("rng required for sigmoid coupling")
        return apply_coupling_sigmoid(
            hidden_states, coupling_matrix, alpha, beta, rng
        )
    else:
        raise ValueError(f"Unknown coupling mode: {mode}")


def compute_hidden_features(
    emission_features: torch.Tensor,
    coupling_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute global feature directions from emission features and coupling.

    For each hidden state i, the "global" direction is the normalized mean
    of the emission directions it controls:
        hidden_feature_i = normalize(sum_{j: C_{ji}=1} f_j)

    Args:
        emission_features: (M, d) emission feature directions.
        coupling_matrix: (M, K) binary coupling matrix.

    Returns:
        (K, d) hidden feature directions (unit norm).
    """
    # C.T: (K, M), emission_features: (M, d) -> (K, d)
    hidden_dirs = coupling_matrix.T @ emission_features
    norms = hidden_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return hidden_dirs / norms
