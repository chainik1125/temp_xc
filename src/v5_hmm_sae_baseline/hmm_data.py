"""HMM data generation: stochastic emissions on top of the existing Markov chain pipeline.

Extends the deterministic emission model (s_t = z_t) to a proper Hidden Markov Model
where each hidden state emits s=1 with probability p_A or p_B:

    s_t | z_t = A ~ Bernoulli(p_A)
    s_t | z_t = B ~ Bernoulli(p_B)

The current MC is the special case p_A=0, p_B=1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.data_generation.configs import (
    FeatureConfig,
    MagnitudeConfig,
    SequenceConfig,
    TransitionConfig,
)
from src.data_generation.magnitudes import sample_magnitudes
from src.data_generation.transition import build_transition_matrix, stationary_distribution
from src.shared.orthogonalize import orthogonalize
from src.shared.temporal_support import generate_support_markov
from src.utils.logging import log


@dataclass
class HMMEmissionConfig:
    """Emission probabilities for the two hidden states.

    p_A: P(s=1 | z=A), the emission probability in state A.
    p_B: P(s=1 | z=B), the emission probability in state B.
    """

    p_A: float = 0.0
    p_B: float = 1.0


@dataclass
class HMMDataConfig:
    """Top-level configuration for HMM data generation."""

    transition: TransitionConfig = field(default_factory=TransitionConfig)
    emission: HMMEmissionConfig = field(default_factory=HMMEmissionConfig)
    magnitude: MagnitudeConfig = field(default_factory=MagnitudeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    seed: int = 42

    @classmethod
    def from_reset_process_hmm(
        cls,
        lam: float,
        q: float,
        p_A: float,
        p_B: float,
        k: int = 10,
        d: int = 64,
        T: int = 128,
        n_sequences: int = 200,
        seed: int = 42,
    ) -> HMMDataConfig:
        """Construct from the reset process parameterization with HMM emissions.

        Args:
            lam: Mixing parameter in [0, 1]. 0 = perfect memory, 1 = i.i.d.
            q: Stationary probability of being in hidden state B.
            p_A: Emission probability in state A.
            p_B: Emission probability in state B.
            k: Number of ground-truth features.
            d: Ambient dimension.
            T: Sequence length.
            n_sequences: Number of sequences.
            seed: Random seed.
        """
        matrix = build_transition_matrix(lam, q)
        return cls(
            transition=TransitionConfig(matrix=matrix, stationary_on_prob=q),
            emission=HMMEmissionConfig(p_A=p_A, p_B=p_B),
            magnitude=MagnitudeConfig(),
            features=FeatureConfig(k=k, d=d),
            sequence=SequenceConfig(T=T, n_sequences=n_sequences),
            seed=seed,
        )


def hmm_marginal_sparsity(config: HMMDataConfig) -> float:
    """Compute the marginal firing probability mu = (1-q)*p_A + q*p_B.

    Args:
        config: HMM data configuration.

    Returns:
        Marginal probability that the observed support is 1.
    """
    pi = stationary_distribution(config.transition.matrix)
    pi_A, pi_B = pi[0].item(), pi[1].item()
    p_A, p_B = config.emission.p_A, config.emission.p_B
    return pi_A * p_A + pi_B * p_B


def hmm_autocorrelation_amplitude(config: HMMDataConfig) -> float:
    """Compute the autocorrelation amplitude prefactor gamma.

    gamma = pi_A * pi_B * (p_B - p_A)^2 / [mu * (1 - mu)]

    This prefactor is 1 for the MC case (p_A=0, p_B=1) and 0 when p_A=p_B.

    Args:
        config: HMM data configuration.

    Returns:
        Amplitude prefactor in [0, 1].
    """
    pi = stationary_distribution(config.transition.matrix)
    pi_A, pi_B = pi[0].item(), pi[1].item()
    p_A, p_B = config.emission.p_A, config.emission.p_B
    mu = pi_A * p_A + pi_B * p_B

    if mu < 1e-12 or (1 - mu) < 1e-12:
        return 0.0

    return pi_A * pi_B * (p_B - p_A) ** 2 / (mu * (1 - mu))


def hmm_theoretical_autocorrelation(
    config: HMMDataConfig, max_lag: int
) -> torch.Tensor:
    """Compute theoretical autocorrelation of the observations at each lag.

    Corr(s_t, s_{t+tau}) = rho^|tau| * gamma

    where rho = alpha - beta is the second eigenvalue of P and gamma is the
    amplitude prefactor.

    Args:
        config: HMM data configuration.
        max_lag: Maximum lag to compute.

    Returns:
        Tensor of shape (max_lag + 1,) with autocorrelation at each lag.
    """
    P = config.transition.matrix
    alpha = P[1, 1].item()
    beta = P[0, 1].item()
    rho = alpha - beta
    gamma = hmm_autocorrelation_amplitude(config)

    # Lag 0 is 1.0 by definition; for tau > 0: rho^tau * gamma
    autocorr = [1.0] + [gamma * rho ** tau for tau in range(1, max_lag + 1)]
    return torch.tensor(autocorr)


def generate_hmm_dataset(
    config: HMMDataConfig,
    rng: torch.Generator | None = None,
) -> dict:
    """Generate a complete HMM dataset.

    Pipeline:
        1. Generate ground-truth feature directions (orthogonalized)
        2. Generate hidden state sequences via Markov chain
        3. Generate observed support by sampling Bernoulli(p_{z_{i,t}})
        4. Sample magnitudes
        5. Compute activations a = s * m
        6. Compute observations x = sum_i a_{i,t} * f_i

    Args:
        config: HMM data configuration.
        rng: Optional torch RNG. If None, one is created from config.seed.

    Returns:
        Dict containing:
            features: (k, d) ground-truth feature directions
            hidden_states: (n_seq, k, T) hidden Markov chain states
            support: (n_seq, k, T) observed (emitted) binary support
            magnitudes: (n_seq, k, T) magnitude values
            activations: (n_seq, k, T) a = s * m
            x: (n_seq, T, d) observation vectors
            config: the HMMDataConfig used
    """
    if rng is None:
        rng = torch.Generator().manual_seed(config.seed)

    k = config.features.k
    d = config.features.d
    T = config.sequence.T
    n_seq = config.sequence.n_sequences

    # Step 1: Generate feature directions
    if config.features.orthogonal:
        global_rng_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        features = orthogonalize(
            num_vectors=k,
            vector_len=d,
            target_cos_sim=config.features.target_cos_sim,
        )
        torch.random.set_rng_state(global_rng_state)
    else:
        features = torch.randn(k, d, generator=rng)
        features = features / features.norm(dim=1, keepdim=True)

    log("data", "generated feature directions", k=k, d=d)

    p_A = config.emission.p_A
    p_B = config.emission.p_B

    all_hidden = torch.zeros(n_seq, k, T)
    all_support = torch.zeros(n_seq, k, T)
    all_magnitudes = torch.zeros(n_seq, k, T)
    all_activations = torch.zeros(n_seq, k, T)
    all_x = torch.zeros(n_seq, T, d)

    # Pre-allocate emission probability tensors (avoid repeated allocation in loop)
    prob_B = torch.full((k, T), p_B)
    prob_A = torch.full((k, T), p_A)

    for seq_idx in range(n_seq):
        # Step 2: Hidden states from Markov chain
        hidden = generate_support_markov(
            k, T, config.transition.matrix, config.transition.stationary_on_prob, rng
        )

        # Step 3: Observed support via stochastic emission
        emission_noise = torch.rand(k, T, generator=rng)
        emission_probs = torch.where(hidden > 0.5, prob_B, prob_A)
        support = (emission_noise < emission_probs).float()

        # Step 4: Magnitudes
        magnitudes = sample_magnitudes(k, T, config.magnitude, rng)

        # Step 5: Activations
        activations = support * magnitudes

        # Step 6: Observations x_t = sum_i a_{i,t} * f_i
        x = activations.T @ features  # (T, d)

        all_hidden[seq_idx] = hidden
        all_support[seq_idx] = support
        all_magnitudes[seq_idx] = magnitudes
        all_activations[seq_idx] = activations
        all_x[seq_idx] = x

    log("data", "generated HMM dataset", n_sequences=n_seq, T=T)

    return {
        "features": features,
        "hidden_states": all_hidden,
        "support": all_support,
        "magnitudes": all_magnitudes,
        "activations": all_activations,
        "x": all_x,
        "config": config,
    }
