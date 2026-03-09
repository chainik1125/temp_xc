"""Smoke test for the data generation pipeline."""

import torch

from src.data_generation.configs import (
    DataGenerationConfig,
    FeatureConfig,
    MagnitudeConfig,
    SequenceConfig,
    TransitionConfig,
)
from src.data_generation.dataset import generate_dataset
from src.data_generation.transition import stationary_distribution
from src.utils.logging import log

# ============================================================================
# Test 1: Reset process dataset
# ============================================================================

log("info", "smoke test 1: reset process dataset")

cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
    magnitude=MagnitudeConfig(distribution="half_normal", mu=0.0, sigma=1.0),
    features=FeatureConfig(k=10, d=64, orthogonal=True),
    sequence=SequenceConfig(T=128, n_sequences=5),
    seed=42,
)

result = generate_dataset(cfg)

log("info", "output shapes:")
for key in ["features", "support", "magnitudes", "activations", "x"]:
    log("data", f"  {key}: {tuple(result[key].shape)}")

# Verify x = activations.T @ features
for seq_idx in range(cfg.sequence.n_sequences):
    expected_x = result["activations"][seq_idx].T @ result["features"]
    max_err = (result["x"][seq_idx] - expected_x).abs().max().item()
    assert max_err < 1e-5, f"Sequence {seq_idx}: max reconstruction error = {max_err}"

log("result", "x vectors are correct linear combinations of features")

# Check empirical sparsity
empirical_p = result["support"].mean().item()
log("result", f"empirical_p={empirical_p:.4f}", expected_p=cfg.transition.stationary_on_prob)

# ============================================================================
# Test 2: Custom transition matrix
# ============================================================================

log("info", "smoke test 2: custom transition matrix")

P_custom = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
pi = stationary_distribution(P_custom)
log("data", f"custom matrix stationary distribution: [{pi[0]:.4f}, {pi[1]:.4f}]")

cfg2 = DataGenerationConfig(
    transition=TransitionConfig(matrix=P_custom, stationary_on_prob=pi[1].item()),
    features=FeatureConfig(k=10, d=64, orthogonal=True),
    sequence=SequenceConfig(T=128, n_sequences=5),
    seed=42,
)

result2 = generate_dataset(cfg2)

log("info", "output shapes:")
for key in ["features", "support", "magnitudes", "activations", "x"]:
    log("data", f"  {key}: {tuple(result2[key].shape)}")

empirical_p2 = result2["support"].mean().item()
log("result", f"empirical_p={empirical_p2:.4f}", expected_p=pi[1].item())

# ============================================================================
# Summary
# ============================================================================

log("done", "all smoke tests passed")
