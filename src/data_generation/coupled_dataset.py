"""Coupled-feature data generation pipeline.

Generates data where K hidden states drive M > K emission features through
a coupling matrix, creating a genuine separation between local (emission-level)
and global (hidden-state-level) structure.

See docs/aniket/coupled_features_plan.md for the mathematical specification.
"""

import torch

from src.data_generation.activations import generate_activations
from src.data_generation.configs import CoupledDataGenerationConfig
from src.data_generation.coupling import (
    apply_coupling,
    compute_hidden_features,
    generate_coupling_matrix,
)
from src.data_generation.magnitudes import sample_magnitudes
from src.data_generation.support import generate_hidden_states
from src.shared.orthogonalize import orthogonalize
from src.utils.logging import log


def generate_coupled_dataset(
    config: CoupledDataGenerationConfig,
    rng: torch.Generator | None = None,
) -> dict:
    """Generate a coupled-feature dataset.

    K hidden Markov chains produce M emission features through a coupling
    matrix. The result includes both emission-level and hidden-state-level
    ground truth for evaluating local vs global feature recovery.

    Args:
        config: Coupled data generation configuration.
        rng: Optional torch RNG. If None, created from config.seed.

    Returns:
        Dict containing:
            emission_features: (M, d) local ground truth directions
            hidden_features: (K, d) global ground truth directions
            coupling_matrix: (M, K) binary coupling matrix
            hidden_states: (n_seq, K, T) binary hidden chains
            support: (n_seq, M, T) binary emission support
            magnitudes: (n_seq, M, T)
            activations: (n_seq, M, T)
            x: (n_seq, T, d) observation vectors
            config: the CoupledDataGenerationConfig used
    """
    if rng is None:
        rng = torch.Generator().manual_seed(config.seed)

    K = config.coupling.K_hidden
    M = config.coupling.M_emission
    d = config.hidden_dim
    T = config.sequence.T
    n_seq = config.sequence.n_sequences
    n_parents = config.coupling.n_parents

    # Generate M emission feature directions
    global_rng_state = torch.random.get_rng_state()
    torch.manual_seed(config.seed)
    emission_features = orthogonalize(
        num_vectors=M,
        vector_len=d,
        target_cos_sim=config.target_cos_sim,
    )
    torch.random.set_rng_state(global_rng_state)
    log("data", "generated emission features", M=M, d=d)

    # Generate coupling matrix
    coupling_matrix = generate_coupling_matrix(K, M, n_parents, rng)
    log("data", "generated coupling matrix", shape=f"{M}x{K}", n_parents=n_parents)

    # Compute hidden feature directions (global ground truth)
    hidden_features = compute_hidden_features(emission_features, coupling_matrix)
    log("data", "computed hidden features", K=K)

    # Generate hidden states and coupled emissions
    all_hidden = torch.zeros(n_seq, K, T)
    all_support = torch.zeros(n_seq, M, T)
    all_magnitudes = torch.zeros(n_seq, M, T)
    all_activations = torch.zeros(n_seq, M, T)
    all_x = torch.zeros(n_seq, T, d)

    for seq_idx in range(n_seq):
        # K independent hidden chains
        hidden_states = generate_hidden_states(K, T, config.transition, rng)

        # Map hidden states to M emission features via coupling
        support = apply_coupling(
            hidden_states,
            coupling_matrix,
            mode=config.coupling.emission_mode,
            alpha=config.coupling.sigmoid_alpha,
            beta=config.coupling.sigmoid_beta,
            rng=rng,
        )

        # Sample magnitudes for M emissions
        magnitudes = sample_magnitudes(M, T, config.magnitude, rng)
        activations = generate_activations(support, magnitudes)

        # x_t = sum_j a_j(t) * f_j  =>  x = activations.T @ emission_features
        x = activations.T @ emission_features  # (T, d)

        all_hidden[seq_idx] = hidden_states
        all_support[seq_idx] = support
        all_magnitudes[seq_idx] = magnitudes
        all_activations[seq_idx] = activations
        all_x[seq_idx] = x

    log("data", "generated coupled dataset", n_sequences=n_seq, T=T)

    return {
        "emission_features": emission_features,
        "hidden_features": hidden_features,
        "coupling_matrix": coupling_matrix,
        "hidden_states": all_hidden,
        "support": all_support,
        "magnitudes": all_magnitudes,
        "activations": all_activations,
        "x": all_x,
        "config": config,
    }
