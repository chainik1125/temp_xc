"""Factorial HMM data generation: event-level temporal correlations.

Unlike the independent Markov chain setup (markov_data_generation.py) where
each feature has its own support chain, here features are grouped into events.
Each event has a single Markov chain controlling whether the event is active,
and when an event is active, all features in that event group activate together.

This creates block structure in the temporal correlation matrix: features within
the same event are perfectly correlated (hard events) or strongly correlated
(soft events), while features across events are independent.

Supports two modes:
  - Non-overlapping (original): contiguous blocks, features belong to exactly one event.
  - Overlapping (general): arbitrary membership matrix, features can belong to
    multiple events (activated via logical OR). Optional per-feature dropout
    for "soft" events.

The key question: does this event-level structure make TFA's temporal channel
more useful than per-feature autocorrelation alone?
"""

import torch

from src.utils.device import DEFAULT_DEVICE
from src.v2_temporal_schemeC.markov_data_generation import markov_transition_probs


# ── Membership matrix constructors ──────────────────────────────────


def create_block_membership(
    n_events: int,
    features_per_event: int,
) -> torch.Tensor:
    """Create non-overlapping block membership matrix.

    Feature j belongs to event j // features_per_event. This is equivalent
    to the original generate_event_support behavior.

    Returns:
        Binary tensor of shape (n_features, n_events).
    """
    n_features = n_events * features_per_event
    membership = torch.zeros(n_features, n_events)
    for g in range(n_events):
        start = g * features_per_event
        end = start + features_per_event
        membership[start:end, g] = 1.0
    return membership


def create_overlapping_membership(
    n_features: int,
    n_events: int,
    base_features_per_event: int,
    overlap_features: int,
    seed: int = 0,
) -> torch.Tensor:
    """Create membership matrix with some features shared across events.

    Each event gets ``base_features_per_event`` exclusive features (assigned
    to that event only). The remaining features (up to ``overlap_features``)
    are each assigned to exactly 2 randomly chosen events, creating cross-event
    correlations. All n_features are guaranteed to belong to at least one event.

    Requires n_features >= n_events * base_features_per_event.
    The number of shared features is min(overlap_features, n_features - n_exclusive).

    Args:
        n_features: Total number of features.
        n_events: Number of event groups.
        base_features_per_event: Exclusive features per event.
        overlap_features: Max number of features that belong to two events.
        seed: RNG seed for reproducible overlap assignment.

    Returns:
        Binary tensor of shape (n_features, n_events).
    """
    n_exclusive = n_events * base_features_per_event
    assert n_features >= n_exclusive, (
        f"Need at least {n_exclusive} features for "
        f"{n_events} events × {base_features_per_event} base features"
    )

    membership = torch.zeros(n_features, n_events)

    # Assign exclusive features: features [g*B, (g+1)*B) belong to event g only
    for g in range(n_events):
        start = g * base_features_per_event
        end = start + base_features_per_event
        membership[start:end, g] = 1.0

    # Remaining features become shared: each assigned to 2 different events
    rng = torch.Generator().manual_seed(seed)
    n_remaining = n_features - n_exclusive
    n_shared = min(overlap_features, n_remaining)

    for i in range(n_shared):
        feat_idx = n_exclusive + i
        # Pick 2 distinct events
        events = torch.randperm(n_events, generator=rng)[:2]
        membership[feat_idx, events[0]] = 1.0
        membership[feat_idx, events[1]] = 1.0

    # Any features beyond n_exclusive + n_shared that aren't shared:
    # assign each to a single random event so they're not dead
    for i in range(n_shared, n_remaining):
        feat_idx = n_exclusive + i
        event = torch.randint(n_events, (1,), generator=rng).item()
        membership[feat_idx, event] = 1.0

    return membership


# ── Marginal statistics ─────────────────────────────────────────────


def compute_marginal_pi(
    pi_events: torch.Tensor,
    membership: torch.Tensor,
    dropout_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-feature marginal activation probability.

    For hard events with OR combination:
        P(feature j active) = 1 - prod_{g in events(j)} (1 - pi_events[g])

    With dropout:
        P(feature j active) = (1 - dropout_prob[j]) * [above]

    Args:
        pi_events: (n_events,) marginal probability per event.
        membership: (n_features, n_events) binary membership matrix.
        dropout_prob: (n_features,) per-feature dropout probability, or None.

    Returns:
        (n_features,) marginal activation probability per feature.
    """
    # For each feature, compute product of (1 - pi_g) over member events
    # membership[j, g] == 1 means feature j is in event g
    # We want: prod_g (1 - pi_g)^{membership[j,g]}
    log_inactive = torch.log(1.0 - pi_events + 1e-12)  # (n_events,)
    log_all_inactive = membership @ log_inactive  # (n_features,)
    pi_features = 1.0 - torch.exp(log_all_inactive)

    if dropout_prob is not None:
        pi_features = pi_features * (1.0 - dropout_prob)

    return pi_features


# ── General event support generation ────────────────────────────────


def _generate_event_chains(
    batch_size: int,
    seq_len: int,
    pi_events: torch.Tensor,
    rho_events: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate event-level Markov chains.

    Returns:
        event_support: Binary (batch, seq_len, n_events).
    """
    n_events = pi_events.shape[0]
    pi_events = pi_events.to(device)
    rho_events = rho_events.to(device)

    p01, p10 = markov_transition_probs(pi_events, rho_events)

    event_support = torch.zeros(batch_size, seq_len, n_events, device=device)
    event_support[:, 0, :] = (
        torch.rand(batch_size, n_events, device=device) < pi_events
    ).float()

    for t in range(1, seq_len):
        prev = event_support[:, t - 1, :]
        u = torch.rand(batch_size, n_events, device=device)
        turn_on = (prev == 0) & (u < p01)
        stay_on = (prev == 1) & (u >= p10)
        event_support[:, t, :] = (turn_on | stay_on).float()

    return event_support


def generate_event_support_general(
    batch_size: int,
    seq_len: int,
    pi_events: torch.Tensor,
    rho_events: torch.Tensor,
    membership: torch.Tensor,
    dropout_prob: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate feature support via factorial HMM with arbitrary membership.

    Feature j activates when ANY of its member events is active (logical OR).
    Optional per-feature dropout provides "soft" events.

    Args:
        batch_size: Number of sequences.
        seq_len: Sequence length T.
        pi_events: (n_events,) marginal probability per event.
        rho_events: (n_events,) lag-1 autocorrelation per event.
        membership: (n_features, n_events) binary membership matrix.
        dropout_prob: (n_features,) per-feature dropout probability, or None.
        device: Torch device.

    Returns:
        feature_support: Binary (batch, seq_len, n_features).
        event_support: Binary (batch, seq_len, n_events).
    """
    membership = membership.to(device)
    event_support = _generate_event_chains(
        batch_size, seq_len, pi_events, rho_events, device
    )

    # Expand: feature j is on if any member event is on
    # event_support: (B, T, G), membership.T: (G, F)
    # matmul gives (B, T, F) with counts; threshold at > 0 for OR
    feature_support = (event_support @ membership.T > 0).float()

    # Optional per-feature dropout
    if dropout_prob is not None:
        dropout_prob = dropout_prob.to(device)
        drop_mask = (torch.rand_like(feature_support) >= dropout_prob).float()
        feature_support = feature_support * drop_mask

    return feature_support, event_support


def generate_event_activations_general(
    batch_size: int,
    seq_len: int,
    pi_events: torch.Tensor,
    rho_events: torch.Tensor,
    membership: torch.Tensor,
    dropout_prob: torch.Tensor | None = None,
    mean_magnitudes: torch.Tensor | None = None,
    std_magnitudes: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate activations with general factorial HMM support.

    Args:
        batch_size, seq_len: Sequence dimensions.
        pi_events: (n_events,) marginal probability per event.
        rho_events: (n_events,) lag-1 autocorrelation per event.
        membership: (n_features, n_events) binary membership matrix.
        dropout_prob: (n_features,) per-feature dropout, or None.
        mean_magnitudes: Per-feature magnitude mean. Defaults to ones.
        std_magnitudes: Per-feature magnitude std. Defaults to zeros.
        device: Torch device.

    Returns:
        activations: (batch, seq_len, n_features)
        feature_support: (batch, seq_len, n_features)
        event_support: (batch, seq_len, n_events)
    """
    n_features = membership.shape[0]

    feature_support, event_support = generate_event_support_general(
        batch_size, seq_len, pi_events, rho_events,
        membership, dropout_prob, device,
    )

    if mean_magnitudes is None:
        mean_magnitudes = torch.ones(n_features, device=device)
    if std_magnitudes is None:
        std_magnitudes = torch.zeros(n_features, device=device)

    mean_magnitudes = mean_magnitudes.to(device)
    std_magnitudes = std_magnitudes.to(device)

    noise = torch.randn(batch_size, seq_len, n_features, device=device)
    magnitudes = (mean_magnitudes + std_magnitudes * noise).relu()

    activations = feature_support * magnitudes

    return activations, feature_support, event_support


# ── Original non-overlapping API (backward compatible) ──────────────


def generate_event_support(
    batch_size: int,
    seq_len: int,
    n_events: int,
    features_per_event: int,
    pi_events: torch.Tensor,
    rho_events: torch.Tensor,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate feature support via factorial HMM (hard, non-overlapping events).

    Each event has an independent 2-state Markov chain. When event g is
    active at time t, ALL features in group g are active. Features in
    group g are indices [g*F, (g+1)*F) where F = features_per_event.

    Delegates to generate_event_support_general with a block membership matrix.

    Args:
        batch_size: Number of sequences.
        seq_len: Sequence length T.
        n_events: Number of event groups G.
        features_per_event: Features per event F. Total features = G*F.
        pi_events: Marginal activation probability per event, shape (G,).
        rho_events: Lag-1 autocorrelation per event, shape (G,).
        device: Torch device.

    Returns:
        feature_support: Binary (batch, seq_len, n_features).
        event_support: Binary (batch, seq_len, n_events).
    """
    membership = create_block_membership(n_events, features_per_event)
    return generate_event_support_general(
        batch_size, seq_len, pi_events, rho_events,
        membership, dropout_prob=None, device=device,
    )


def generate_event_activations(
    batch_size: int,
    seq_len: int,
    n_events: int,
    features_per_event: int,
    pi_events: torch.Tensor,
    rho_events: torch.Tensor,
    mean_magnitudes: torch.Tensor | None = None,
    std_magnitudes: torch.Tensor | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate activations with factorial HMM support and i.i.d. magnitudes.

    Non-overlapping block events. Delegates to generate_event_activations_general.

    Args:
        batch_size, seq_len, n_events, features_per_event: As above.
        pi_events: Marginal activation probability per event, shape (G,).
        rho_events: Lag-1 autocorrelation per event, shape (G,).
        mean_magnitudes: Per-feature magnitude mean. Defaults to ones.
        std_magnitudes: Per-feature magnitude std. Defaults to zeros (unit mag).
        device: Torch device.

    Returns:
        activations: (batch, seq_len, n_features)
        feature_support: (batch, seq_len, n_features)
        event_support: (batch, seq_len, n_events)
    """
    membership = create_block_membership(n_events, features_per_event)
    return generate_event_activations_general(
        batch_size, seq_len, pi_events, rho_events,
        membership, dropout_prob=None,
        mean_magnitudes=mean_magnitudes,
        std_magnitudes=std_magnitudes,
        device=device,
    )
