"""Full data generation pipeline: configs -> dataset dict."""

import torch

from src.data_generation.activations import generate_activations
from src.data_generation.configs import DataGenerationConfig
from src.data_generation.magnitudes import sample_magnitudes
from src.data_generation.support import generate_support
from src.shared.orthogonalize import orthogonalize
from src.utils.logging import log


def generate_dataset(
    config: DataGenerationConfig,
    rng: torch.Generator | None = None,
) -> dict:
    """Generate a complete synthetic dataset for temporal crosscoder experiments.

    Args:
        config: Full pipeline configuration.
        rng: Optional torch RNG. If None, one is created from config.seed.

    Returns:
        Dict containing:
            features: (k, d) ground-truth feature directions
            support: (n_sequences, k, T) binary support
            magnitudes: (n_sequences, k, T) magnitude values
            activations: (n_sequences, k, T) activation coefficients
            x: (n_sequences, T, d) activation vectors
            config: the DataGenerationConfig used
    """
    if rng is None:
        rng = torch.Generator().manual_seed(config.seed)

    k = config.features.k
    d = config.features.d
    T = config.sequence.T
    n_seq = config.sequence.n_sequences

    # Generate feature directions (orthogonalize uses global torch RNG,
    # so we seed it locally and restore state afterwards)
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

    # Generate support, magnitudes, activations for each sequence
    all_support = torch.zeros(n_seq, k, T)
    all_magnitudes = torch.zeros(n_seq, k, T)
    all_activations = torch.zeros(n_seq, k, T)
    all_x = torch.zeros(n_seq, T, d)

    for seq_idx in range(n_seq):
        support = generate_support(k, T, config.transition, rng)
        magnitudes = sample_magnitudes(k, T, config.magnitude, rng)
        activations = generate_activations(support, magnitudes)

        # x_t = sum_i a_{i,t} * f_i  =>  x = activations.T @ features
        # activations: (k, T), features: (k, d)
        # x: (T, d) = activations.T @ features
        x = activations.T @ features  # (T, d)

        all_support[seq_idx] = support
        all_magnitudes[seq_idx] = magnitudes
        all_activations[seq_idx] = activations
        all_x[seq_idx] = x

    log("data", "generated dataset", n_sequences=n_seq, T=T)

    return {
        "features": features,
        "support": all_support,
        "magnitudes": all_magnitudes,
        "activations": all_activations,
        "x": all_x,
        "config": config,
    }
