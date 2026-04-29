"""Synthetic data: mixture of K independent ergodic components.

Supports two component types:
- LeakyReset: simple reset dynamics parameterized by (d, lambda)
- HMM: general hidden Markov model with explicit transition matrices (MESS3-style)

The joint process samples one component per timestep according to fixed
mixture weights omega, emits from that component, and updates only that
component's hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# ── Component types ──


@dataclass
class LeakyResetComponent:
    """Leaky-reset ergodic component."""

    d: int  # number of hidden states
    lam: float  # write strength (0 = no update, 1 = hard reset)
    V: int = 4  # vocabulary size

    def __post_init__(self):
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {self.lam}")


@dataclass
class HMMComponent:
    """General HMM component defined by transition matrices.

    transition_matrices: (V, S, S) tensor where T[v, i, j] = P(state j, emit v | state i).
        Rows of the net matrix sum(T, axis=0) must be stochastic.
    """

    transition_matrices: torch.Tensor  # (V, S, S)
    label: str = ""

    @property
    def V(self) -> int:
        return self.transition_matrices.shape[0]

    @property
    def d(self) -> int:
        return self.transition_matrices.shape[1]


def mess3_transitions(x: float, a: float) -> torch.Tensor:
    """Build MESS3 transition matrices (V=3 symbols, S=3 states).

    Parameters:
        x: leak/mixing parameter (0 = deterministic, 0.5 = max mixing)
        a: asymmetry (controls how strongly each state prefers its own symbol)

    Returns: (3, 3, 3) tensor T[v, i, j].
    """
    b = (1 - a) / 2
    y = 1 - 2 * x

    # T[v, i, j] = P(emit v, go to state j | in state i)
    # Each symbol v has a 3x3 sub-stochastic matrix
    T = torch.zeros(3, 3, 3)

    # Symbol 0: state 0 is "home" for this symbol
    T[0, 0, 0] = a * (1 - x)   # state 0 stays, emits 0 with high prob
    T[0, 0, 1] = b * x          # state 0 -> 1 via leak
    T[0, 0, 2] = b * x          # state 0 -> 2 via leak
    T[0, 1, 0] = b * (1 - x)
    T[0, 1, 1] = a * x
    T[0, 1, 2] = b * x
    T[0, 2, 0] = b * (1 - x)
    T[0, 2, 1] = b * x
    T[0, 2, 2] = a * x

    # Symbol 1: state 1 is "home"
    T[1, 0, 0] = a * x
    T[1, 0, 1] = b * (1 - x)
    T[1, 0, 2] = b * x
    T[1, 1, 0] = b * x
    T[1, 1, 1] = a * (1 - x)
    T[1, 1, 2] = b * x
    T[1, 2, 0] = b * x
    T[1, 2, 1] = b * (1 - x)
    T[1, 2, 2] = a * x

    # Symbol 2: state 2 is "home"
    T[2, 0, 0] = a * x
    T[2, 0, 1] = b * x
    T[2, 0, 2] = b * (1 - x)
    T[2, 1, 0] = b * x
    T[2, 1, 1] = a * x
    T[2, 1, 2] = b * (1 - x)
    T[2, 2, 0] = b * x
    T[2, 2, 1] = b * x
    T[2, 2, 2] = a * (1 - x)

    return T


def leaky_rrxor_transitions(p1: float, p2: float, epsilon: float) -> torch.Tensor:
    """Build leaky RRXOR transition matrices (V=2 symbols, S=5 states).

    States: S=start, 0=saw-0, 1=saw-1, T=true(XOR), F=false(XOR).
    The process computes a running XOR of consecutive token pairs.

    Parameters:
        p1: probability of emitting 0 from start state (controls token bias)
        p2: probability of the XOR-consistent transition (controls XOR strength)
        epsilon: leak parameter (0 = pure RRXOR, higher = more mixing/ergodic)
    """
    S, O, ONE, T_, F_ = 0, 1, 2, 3, 4  # state indices
    T = torch.zeros(2, 5, 5)

    # Pure RRXOR
    T[0, S, O] = p1           # start -> saw-0 (emit 0)
    T[1, S, ONE] = 1 - p1     # start -> saw-1 (emit 1)
    T[0, O, F_] = p2          # saw-0, emit 0 -> false (0 XOR 0 = 0)
    T[1, O, T_] = 1 - p2      # saw-0, emit 1 -> true  (0 XOR 1 = 1)
    T[0, ONE, T_] = p2        # saw-1, emit 0 -> true  (1 XOR 0 = 1)
    T[1, ONE, F_] = 1 - p2    # saw-1, emit 1 -> false (1 XOR 1 = 0)
    T[1, T_, S] = 1.0         # true -> start (emit 1)
    T[0, F_, S] = 1.0         # false -> start (emit 0)

    # Add leak
    leak = torch.ones(2, 5, 5)
    T = (1 - epsilon) * T + (epsilon / 10) * leak

    return T


Component = LeakyResetComponent | HMMComponent


# ── Mixture config ──


@dataclass
class MixtureConfig:
    """Configuration for the composed mixture process."""

    components: list[Component]
    omega: list[float] | None = None
    decode_noise: float = 0.05  # only used by LeakyReset components
    shared_vocab: bool = False  # if True, all components share the same V symbols
    seed: int = 42

    def __post_init__(self):
        K = len(self.components)
        if self.omega is None:
            self.omega = [1.0 / K] * K
        if len(self.omega) != K:
            raise ValueError(f"omega has {len(self.omega)} entries but {K} components")
        if abs(sum(self.omega) - 1.0) > 1e-6:
            raise ValueError(f"omega must sum to 1, got {sum(self.omega)}")
        if self.shared_vocab:
            Vs = {c.V for c in self.components}
            if len(Vs) != 1:
                raise ValueError(f"shared_vocab requires all components to have same V, got {Vs}")

    @property
    def K(self) -> int:
        return len(self.components)

    @property
    def V_total(self) -> int:
        if self.shared_vocab:
            return self.components[0].V
        return sum(c.V for c in self.components)


# ── Data generation helpers ──


def _build_signatures(V: int, d: int) -> torch.Tensor:
    sigs = torch.full((V, d), 1.0 / (d * 10), dtype=torch.float32)
    for k in range(V):
        sigs[k, k % d] = 1.0
    return sigs / sigs.sum(dim=1, keepdim=True)


def _build_emission_matrix(V: int, d: int, decode_noise: float) -> torch.Tensor:
    E = torch.full((d, V), decode_noise / (V - 1) if V > 1 else 1.0)
    for j in range(d):
        E[j, j % V] = 1.0 - decode_noise
    return E


@dataclass
class MixtureDataset:
    """Generated dataset from the mixture process."""

    tokens: torch.Tensor       # (N, L) int, values in [0, V_total)
    components: torch.Tensor   # (N, L) int, which component emitted
    observations: torch.Tensor # (N, L, V_total) one-hot
    config: MixtureConfig
    sequence_omegas: torch.Tensor | None = None  # (N, K), if sampled per sequence
    posterior_omegas: torch.Tensor | None = None  # (N, L, K), posterior mean after x_t

    def windowed(self, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create windowed observations by concatenating W consecutive one-hots.

        Returns:
            flat_obs: (N * (L - W + 1), W * V_total) — concatenated window vectors
            flat_comp: (N * (L - W + 1),) — component label of the CENTER token
        """
        N, L, V = self.observations.shape
        n_windows = L - W + 1
        center = W // 2

        # Unfold: (N, n_windows, W, V)
        windows = self.observations.unfold(1, W, 1)  # (N, n_windows, V, W)
        windows = windows.permute(0, 1, 3, 2).contiguous()  # (N, n_windows, W, V)
        flat_obs = windows.reshape(N * n_windows, W * V)

        # Component label at center of each window
        comp_windows = self.components[:, center: center + n_windows]
        flat_comp = comp_windows.reshape(-1)

        return flat_obs, flat_comp


def generate_mixture_data(
    config: MixtureConfig,
    n_sequences: int,
    seq_len: int,
) -> MixtureDataset:
    """Generate sequences from the mixture of ergodic components."""
    gen = torch.Generator().manual_seed(config.seed)
    K = config.K
    V_total = config.V_total

    # Vocab offsets per component (0 for all if shared_vocab)
    vocab_offsets = []
    if config.shared_vocab:
        vocab_offsets = [0] * K
    else:
        offset = 0
        for c in config.components:
            vocab_offsets.append(offset)
            offset += c.V

    # Build per-component state update machinery
    # For LeakyReset: signatures + emission matrices
    # For HMM: transition matrices (already contain emission info)
    comp_data = []
    for c in config.components:
        if isinstance(c, LeakyResetComponent):
            comp_data.append({
                "type": "leaky_reset",
                "sigs": _build_signatures(c.V, c.d),
                "emissions": _build_emission_matrix(c.V, c.d, config.decode_noise),
                "lam": c.lam,
            })
        elif isinstance(c, HMMComponent):
            # Derive emission probs from transition matrices:
            # P(emit v | state i) = sum_j T[v, i, j]
            T = c.transition_matrices  # (V, S, S)
            emit = T.sum(dim=2)  # (V, S) -> P(v | i)
            emit = emit.T  # (S, V) -> rows are states
            # Net transition: P(j | i) = sum_v T[v, i, j] (already row-stochastic)
            net = T.sum(dim=0)  # (S, S)
            # Per-symbol transition: P(j | i, v) = T[v, i, j] / P(v | i)
            comp_data.append({
                "type": "hmm",
                "T": T,
                "emit": emit,
                "net": net,
            })
        else:
            raise TypeError(f"Unknown component type: {type(c)}")

    # Initialize hidden states: uniform
    states = [torch.ones(n_sequences, c.d) / c.d for c in config.components]
    omega = torch.tensor(config.omega, dtype=torch.float32)

    all_tokens = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    all_components = torch.zeros(n_sequences, seq_len, dtype=torch.long)

    for t in range(seq_len):
        # 1. Sample which component emits
        comp_idx = torch.multinomial(
            omega.unsqueeze(0).expand(n_sequences, -1), 1, generator=gen,
        ).squeeze(1)

        token_global = torch.zeros(n_sequences, dtype=torch.long)

        for c_idx in range(K):
            mask = comp_idx == c_idx
            if not mask.any():
                continue
            n_active = mask.sum().item()
            cd = comp_data[c_idx]

            if cd["type"] == "leaky_reset":
                # Emit: state @ E
                emit_probs = states[c_idx][mask] @ cd["emissions"]
                emit_probs = emit_probs.clamp(min=1e-8)
                emit_probs = emit_probs / emit_probs.sum(dim=1, keepdim=True)
                local_token = torch.multinomial(emit_probs, 1, generator=gen).squeeze(1)
                token_global[mask] = local_token + vocab_offsets[c_idx]
                # Update: leaky reset
                sig = cd["sigs"][local_token]
                states[c_idx][mask] = (1 - cd["lam"]) * states[c_idx][mask] + cd["lam"] * sig

            elif cd["type"] == "hmm":
                # Emit: P(v | state) = state @ emit.T ... but state is a belief vector
                # P(v) = belief @ emit_col(v) = sum_i belief_i * P(v|i)
                emit = cd["emit"]  # (S, V)
                emit_probs = states[c_idx][mask] @ emit  # (n_active, V)
                emit_probs = emit_probs.clamp(min=1e-8)
                emit_probs = emit_probs / emit_probs.sum(dim=1, keepdim=True)
                local_token = torch.multinomial(emit_probs, 1, generator=gen).squeeze(1)
                token_global[mask] = local_token + vocab_offsets[c_idx]

                # Update belief via Bayes rule (vectorized):
                # new_belief_j = sum_i belief_i * T[v, i, j] / P(v)
                T = cd["T"]  # (V, S, S)
                belief = states[c_idx][mask]  # (n_active, S)
                # Gather T[v] for each sample's emitted symbol
                Tv = T[local_token]  # (n_active, S, S)
                # new_unnorm[n, j] = sum_i belief[n, i] * Tv[n, i, j]
                new_unnorm = torch.bmm(belief.unsqueeze(1), Tv).squeeze(1)  # (n_active, S)
                new_norm = new_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-10)
                states[c_idx][mask] = new_unnorm / new_norm

        all_tokens[:, t] = token_global
        all_components[:, t] = comp_idx

    observations = F.one_hot(all_tokens, num_classes=V_total).float()
    return MixtureDataset(
        tokens=all_tokens, components=all_components,
        observations=observations, config=config,
    )


def _sample_dirichlet_rows(alpha: torch.Tensor, n_sequences: int, seed: int) -> torch.Tensor:
    """Sample n_sequences rows from Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)
    samples = rng.gamma(
        shape=np.asarray(alpha, dtype=np.float64),
        scale=1.0,
        size=(n_sequences, alpha.numel()),
    )
    samples /= samples.sum(axis=1, keepdims=True)
    return torch.tensor(samples, dtype=torch.float32)


def generate_mixture_data_with_sequence_omega(
    config: MixtureConfig,
    n_sequences: int,
    seq_len: int,
    concentration: float = 10.0,
) -> MixtureDataset:
    """Generate data with a latent omega sampled separately for each sequence.

    The prior mean over the latent omega matches config.omega. In the distinct-
    vocab setting used by experiment 1, the posterior mean over omega after x_t
    is analytic because the emitted component is directly observed.
    """
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0, got {concentration}")

    gen = torch.Generator().manual_seed(config.seed)
    K = config.K
    V_total = config.V_total

    if config.shared_vocab:
        vocab_offsets = [0] * K
    else:
        vocab_offsets = []
        offset = 0
        for c in config.components:
            vocab_offsets.append(offset)
            offset += c.V

    comp_data = []
    for c in config.components:
        if isinstance(c, LeakyResetComponent):
            comp_data.append({
                "type": "leaky_reset",
                "sigs": _build_signatures(c.V, c.d),
                "emissions": _build_emission_matrix(c.V, c.d, config.decode_noise),
                "lam": c.lam,
            })
        elif isinstance(c, HMMComponent):
            T = c.transition_matrices
            emit = T.sum(dim=2).T
            comp_data.append({
                "type": "hmm",
                "T": T,
                "emit": emit,
            })
        else:
            raise TypeError(f"Unknown component type: {type(c)}")

    states = [torch.ones(n_sequences, c.d) / c.d for c in config.components]

    prior_alpha = torch.tensor(config.omega, dtype=torch.float32) * concentration
    sequence_omegas = _sample_dirichlet_rows(
        prior_alpha,
        n_sequences=n_sequences,
        seed=config.seed,
    )

    all_tokens = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    all_components = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    posterior_omegas = torch.zeros(n_sequences, seq_len, K, dtype=torch.float32)
    component_counts = torch.zeros(n_sequences, K, dtype=torch.float32)

    for t in range(seq_len):
        comp_idx = torch.multinomial(sequence_omegas, 1, generator=gen).squeeze(1)
        token_global = torch.zeros(n_sequences, dtype=torch.long)

        for c_idx in range(K):
            mask = comp_idx == c_idx
            if not mask.any():
                continue

            cd = comp_data[c_idx]
            if cd["type"] == "leaky_reset":
                emit_probs = states[c_idx][mask] @ cd["emissions"]
                emit_probs = emit_probs.clamp(min=1e-8)
                emit_probs = emit_probs / emit_probs.sum(dim=1, keepdim=True)
                local_token = torch.multinomial(emit_probs, 1, generator=gen).squeeze(1)
                token_global[mask] = local_token + vocab_offsets[c_idx]
                sig = cd["sigs"][local_token]
                states[c_idx][mask] = (1 - cd["lam"]) * states[c_idx][mask] + cd["lam"] * sig

            elif cd["type"] == "hmm":
                emit_probs = states[c_idx][mask] @ cd["emit"]
                emit_probs = emit_probs.clamp(min=1e-8)
                emit_probs = emit_probs / emit_probs.sum(dim=1, keepdim=True)
                local_token = torch.multinomial(emit_probs, 1, generator=gen).squeeze(1)
                token_global[mask] = local_token + vocab_offsets[c_idx]

                T = cd["T"]
                belief = states[c_idx][mask]
                Tv = T[local_token]
                new_unnorm = torch.bmm(belief.unsqueeze(1), Tv).squeeze(1)
                new_norm = new_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-10)
                states[c_idx][mask] = new_unnorm / new_norm

        all_tokens[:, t] = token_global
        all_components[:, t] = comp_idx

        component_counts.scatter_add_(
            dim=1,
            index=comp_idx.unsqueeze(1),
            src=torch.ones(n_sequences, 1, dtype=torch.float32),
        )
        posterior_omegas[:, t] = (component_counts + prior_alpha.unsqueeze(0)) / (
            concentration + t + 1
        )

    observations = F.one_hot(all_tokens, num_classes=V_total).float()
    return MixtureDataset(
        tokens=all_tokens,
        components=all_components,
        observations=observations,
        config=config,
        sequence_omegas=sequence_omegas,
        posterior_omegas=posterior_omegas,
    )


# ── Default configurations ──

# Experiment 1: Three leaky resets
DEFAULT_LEAKY_COMPONENTS = [
    LeakyResetComponent(d=3, lam=0.2, V=4),
    LeakyResetComponent(d=4, lam=0.5, V=4),
    LeakyResetComponent(d=2, lam=0.8, V=4),
]

DEFAULT_MIXTURE = MixtureConfig(
    components=DEFAULT_LEAKY_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    decode_noise=0.05,
    seed=42,
)

# Experiment 2b: Three MESS3 processes at different regimes
# Each has V=3 symbols, S=3 states -> V_total = 9
MESS3_COMPONENTS = [
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.05, a=0.8),
        label="Structured(x=0.05,a=0.8)",
    ),
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.15, a=0.5),
        label="Moderate(x=0.15,a=0.5)",
    ),
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.3, a=0.3),
        label="Diffuse(x=0.3,a=0.3)",
    ),
]

MESS3_MIXTURE = MixtureConfig(
    components=MESS3_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    seed=42,
)

# Experiment 2c: Three MESS3 with SHARED vocabulary
# All components emit from the same V=3 symbols -> V_total = 3
# Component identity must be inferred from temporal emission patterns
MESS3_SHARED = MixtureConfig(
    components=MESS3_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    shared_vocab=True,
    seed=42,
)

# Experiment 2e: Three more separated MESS3 regimes
# Chosen to span sticky / moderate / near-memoryless dynamics while remaining
# within the MESS3 family. In a small raw-window search, this triplet gave the
# strongest linear access to latent omega among the candidate shared-vocab sets
# we tried.
MESS3_SEPARATED_COMPONENTS = [
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.08, a=0.9),
        label="Sticky(x=0.08,a=0.9)",
    ),
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.25, a=0.65),
        label="Moderate(x=0.25,a=0.65)",
    ),
    HMMComponent(
        transition_matrices=mess3_transitions(x=0.4, a=0.34),
        label="Diffuse(x=0.4,a=0.34)",
    ),
]

MESS3_SEPARATED = MixtureConfig(
    components=MESS3_SEPARATED_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    seed=42,
)

MESS3_SEPARATED_SHARED = MixtureConfig(
    components=MESS3_SEPARATED_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    shared_vocab=True,
    seed=42,
)

# Experiment 2d: Three leaky RRXORs with shared binary vocabulary
# V=2, S=5 per component. Maximally distinct: extreme marginals + different
# temporal structure (mixing times span 3 to 99 steps).
# P(0) = 0.90, 0.41, 0.37 — C0 is visually obvious (runs of 0s).
RRXOR_COMPONENTS = [
    HMMComponent(
        transition_matrices=leaky_rrxor_transitions(p1=0.95, p2=0.9, epsilon=0.01),
        label="Extreme-0(p1=.95,p2=.9,eps=.01)",
    ),
    HMMComponent(
        transition_matrices=leaky_rrxor_transitions(p1=0.5, p2=0.1, epsilon=0.3),
        label="Balanced(p1=.5,p2=.1,eps=.3)",
    ),
    HMMComponent(
        transition_matrices=leaky_rrxor_transitions(p1=0.05, p2=0.9, epsilon=0.01),
        label="Extreme-1(p1=.05,p2=.9,eps=.01)",
    ),
]

RRXOR_SHARED = MixtureConfig(
    components=RRXOR_COMPONENTS,
    omega=[0.4, 0.35, 0.25],
    shared_vocab=True,
    seed=42,
)
