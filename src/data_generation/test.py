"""Smoke test for the data generation pipeline."""

import torch

from src.data_generation.configs import (
    DataGenerationConfig,
    EmissionConfig,
    FeatureConfig,
    MagnitudeConfig,
    SequenceConfig,
    TransitionConfig,
)
from src.data_generation.dataset import generate_dataset
from src.data_generation.transition import (
    hmm_autocorrelation_amplitude,
    hmm_marginal_sparsity,
    stationary_distribution,
)
from src.utils.logging import log

# ============================================================================
# Test 1: Reset process dataset (deterministic emissions, backwards compat)
# ============================================================================

log("info", "smoke test 1: reset process dataset (MC, p_A=0 p_B=1)")

cfg = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
    magnitude=MagnitudeConfig(distribution="half_normal", mu=0.0, sigma=1.0),
    features=FeatureConfig(k=10, d=64, orthogonal=True),
    sequence=SequenceConfig(T=128, n_sequences=5),
    seed=42,
)

result = generate_dataset(cfg)

log("info", "output shapes:")
for key in ["features", "hidden_states", "support", "magnitudes", "activations", "x"]:
    log("data", f"  {key}: {tuple(result[key].shape)}")

# With default emission (p_A=0, p_B=1), support should equal hidden_states
assert torch.equal(result["support"], result["hidden_states"]), (
    "With p_A=0, p_B=1, support must equal hidden_states"
)
log("result", "support == hidden_states (deterministic emission)")

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
for key in ["features", "hidden_states", "support", "magnitudes", "activations", "x"]:
    log("data", f"  {key}: {tuple(result2[key].shape)}")

empirical_p2 = result2["support"].mean().item()
log("result", f"empirical_p={empirical_p2:.4f}", expected_p=pi[1].item())

# ============================================================================
# Test 3: HMM with stochastic emissions
# ============================================================================

log("info", "smoke test 3: HMM stochastic emissions (p_A=0.0, p_B=0.5, q=0.1)")

cfg3 = DataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    emission=EmissionConfig(p_A=0.0, p_B=0.5),
    features=FeatureConfig(k=10, d=64, orthogonal=True),
    sequence=SequenceConfig(T=128, n_sequences=500),
    seed=42,
)

result3 = generate_dataset(cfg3)

# Verify support != hidden_states (stochastic emission)
assert not torch.equal(result3["support"], result3["hidden_states"]), (
    "With p_B=0.5, support should differ from hidden_states"
)
log("result", "support != hidden_states (stochastic emission)")

# Verify empirical marginal sparsity matches theory
P3 = cfg3.transition.matrix
mu_theory = hmm_marginal_sparsity(P3, 0.0, 0.5)
mu_empirical = result3["support"].mean().item()
mu_err = abs(mu_empirical - mu_theory)
log(
    "result",
    f"mu: theory={mu_theory:.4f} empirical={mu_empirical:.4f} err={mu_err:.4f}",
)
assert mu_err < 0.01, f"Marginal sparsity error too large: {mu_err}"

# Verify amplitude prefactor
gamma = hmm_autocorrelation_amplitude(P3, 0.0, 0.5)
log("result", f"gamma={gamma:.4f} (expected ~0.556 for q=0.1, p_A=0, p_B=0.5)")
assert 0 < gamma < 1, f"Gamma should be in (0, 1), got {gamma}"

# Verify x = activations.T @ features
for seq_idx in range(min(5, cfg3.sequence.n_sequences)):
    expected_x = result3["activations"][seq_idx].T @ result3["features"]
    max_err = (result3["x"][seq_idx] - expected_x).abs().max().item()
    assert max_err < 1e-5, f"Sequence {seq_idx}: max reconstruction error = {max_err}"

log("result", "x vectors correct for HMM case")

# ============================================================================
# Summary
# ============================================================================

log("done", "all smoke tests passed")
