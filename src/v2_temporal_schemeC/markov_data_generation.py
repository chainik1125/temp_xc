"""Scheme C data generation: temporally correlated features via two-state Markov chains.

Each feature i has an independent two-state Markov chain controlling its support
(on/off) at each sequence position. The transition probabilities are parametrized
by the marginal activation probability pi_i and the lag-1 autocorrelation rho_i:

    p_01 = pi_i * (1 - rho_i)     (off -> on)
    p_10 = (1 - pi_i) * (1 - rho_i)  (on -> off)

This gives orthogonal control over sparsity (pi_i) and temporal persistence (rho_i).
Magnitudes are sampled i.i.d. to isolate the support effect.

Reference: docs/han/research_plan/toy_model_idea.md, Scheme C.
"""

import torch

from src.utils.device import DEFAULT_DEVICE


def markov_transition_probs(
    pi: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Markov chain transition probabilities from (pi, rho).

    Args:
        pi: Marginal activation probability per feature, shape (n_features,).
        rho: Lag-1 autocorrelation per feature, shape (n_features,).

    Returns:
        p01: Off-to-on transition probability, shape (n_features,).
        p10: On-to-off transition probability, shape (n_features,).
    """
    p01 = pi * (1 - rho)
    p10 = (1 - pi) * (1 - rho)
    return p01, p10


def generate_markov_support(
    batch_size: int,
    seq_len: int,
    pi: torch.Tensor,
    rho: torch.Tensor,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Generate binary support sequences via independent two-state Markov chains.

    For each feature, the chain is initialized from the stationary distribution
    (Bernoulli(pi_i)), then evolved using the transition probabilities derived
    from (pi_i, rho_i).

    Args:
        batch_size: Number of sequences.
        seq_len: Number of positions per sequence (T).
        pi: Marginal activation probability per feature, shape (n_features,).
        rho: Lag-1 autocorrelation per feature, shape (n_features,).
        device: Torch device.

    Returns:
        Binary tensor of shape (batch_size, seq_len, n_features).
    """
    n_features = pi.shape[0]
    pi = pi.to(device)
    rho = rho.to(device)

    p01, p10 = markov_transition_probs(pi, rho)

    # Initialize from stationary distribution
    support = torch.zeros(batch_size, seq_len, n_features, device=device)
    support[:, 0, :] = (torch.rand(batch_size, n_features, device=device) < pi).float()

    # Evolve the chain
    for t in range(1, seq_len):
        prev = support[:, t - 1, :]  # (batch, n_features)
        u = torch.rand(batch_size, n_features, device=device)

        # If prev == 0: turn on with probability p01
        # If prev == 1: stay on with probability (1 - p10)
        turn_on = (prev == 0) & (u < p01)
        stay_on = (prev == 1) & (u >= p10)
        support[:, t, :] = (turn_on | stay_on).float()

    return support


def generate_markov_activations(
    batch_size: int,
    seq_len: int,
    pi: torch.Tensor,
    rho: torch.Tensor,
    mean_magnitudes: torch.Tensor | None = None,
    std_magnitudes: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate feature activations with Markov-chain temporal support.

    Combines the Markov support process with i.i.d. magnitude sampling:
        a_{i,t} = s_{i,t} * ReLU(mean_i + std_i * epsilon)

    Args:
        batch_size: Number of sequences.
        seq_len: Number of positions per sequence (T).
        pi: Marginal activation probability per feature, shape (n_features,).
        rho: Lag-1 autocorrelation per feature, shape (n_features,).
        mean_magnitudes: Mean magnitude per feature. Defaults to ones.
        std_magnitudes: Std of magnitude noise per feature. Defaults to zeros.
        device: Torch device.

    Returns:
        Tuple of (activations, support):
            activations: shape (batch_size, seq_len, n_features)
            support: binary tensor, shape (batch_size, seq_len, n_features)
    """
    n_features = pi.shape[0]

    support = generate_markov_support(batch_size, seq_len, pi, rho, device)

    if mean_magnitudes is None:
        mean_magnitudes = torch.ones(n_features, device=device)
    if std_magnitudes is None:
        std_magnitudes = torch.zeros(n_features, device=device)

    mean_magnitudes = mean_magnitudes.to(device)
    std_magnitudes = std_magnitudes.to(device)

    # Sample magnitudes i.i.d.
    noise = torch.randn(batch_size, seq_len, n_features, device=device)
    magnitudes = (mean_magnitudes + std_magnitudes * noise).relu()

    activations = support * magnitudes

    return activations, support


def theoretical_autocorrelation(rho: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """Compute the theoretical lag-k autocorrelation for the Markov support process.

    For a two-state Markov chain with parameters (pi, rho), the lag-k
    autocorrelation of the support is rho^k.

    When magnitudes are i.i.d. with mean mu and std sigma, the activation
    autocorrelation at lag k is gamma * rho^k, where
        gamma = pi * sigma^2 / (sigma^2 + (1 - pi) * mu^2)

    For unit magnitudes (std=0), gamma = 0 and the activation autocorrelation
    equals the support autocorrelation (since Var(a) = Var(s) * mu^2 and
    Cov(a_t, a_{t+k}) = mu^2 * Cov(s_t, s_{t+k})).

    Args:
        rho: Per-feature autocorrelation, shape (n_features,).
        lag: The lag k.

    Returns:
        Theoretical autocorrelation at lag k, shape (n_features,).
    """
    return rho ** lag
