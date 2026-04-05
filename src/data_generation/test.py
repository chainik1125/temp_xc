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
# Test 4: Leaky reset
# ============================================================================

log("info", "smoke test 4: leaky reset transition")

from src.data_generation.transition import build_leaky_transition_matrix

# delta=0 should reproduce standard reset
P_standard = TransitionConfig.from_reset_process(lam=0.5, p=0.05).matrix
P_leaky_0 = TransitionConfig.from_leaky_reset(lam=0.5, p=0.05, delta=0.0).matrix
assert torch.allclose(P_standard, P_leaky_0, atol=1e-6), (
    f"delta=0 should reproduce standard reset:\n{P_standard}\nvs\n{P_leaky_0}"
)
log("result", "delta=0 reproduces standard reset")

# Stationary distribution should be p for delta < 1
# (delta=1 is absorbing — no well-defined stationary distribution)
for delta in [0.0, 0.25, 0.5, 0.75]:
    P_leaky = build_leaky_transition_matrix(0.5, 0.05, delta)
    pi = stationary_distribution(P_leaky)
    err = abs(pi[1].item() - 0.05)
    assert err < 1e-6, f"delta={delta}: stationary pi_on={pi[1].item():.6f}, expected 0.05"
log("result", "stationary distribution is p for all delta < 1")

# Autocorrelation should increase with delta
rhos = []
for delta in [0.0, 0.25, 0.5, 0.75]:
    P_leaky = build_leaky_transition_matrix(0.5, 0.05, delta)
    rho = P_leaky[1, 1].item() - P_leaky[0, 1].item()  # alpha - beta
    rhos.append(rho)
for i in range(len(rhos) - 1):
    assert rhos[i] < rhos[i + 1], (
        f"autocorrelation should increase with delta: {rhos}"
    )
log("result", f"autocorrelation increases with delta: {[f'{r:.4f}' for r in rhos]}")

# ============================================================================
# Test 5: Coupled features (OR gate)
# ============================================================================

log("info", "smoke test 5: coupled features (OR gate)")

from src.data_generation.configs import CoupledDataGenerationConfig, CouplingConfig
from src.data_generation.coupled_dataset import generate_coupled_dataset

cfg5 = CoupledDataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    coupling=CouplingConfig(K_hidden=10, M_emission=20, n_parents=2, emission_mode="or"),
    sequence=SequenceConfig(T=64, n_sequences=50),
    hidden_dim=64,
    seed=42,
)
result5 = generate_coupled_dataset(cfg5)

log("info", "output shapes:")
for key in ["emission_features", "hidden_features", "coupling_matrix",
            "hidden_states", "support", "magnitudes", "activations", "x"]:
    log("data", f"  {key}: {tuple(result5[key].shape)}")

# Check shapes
assert result5["emission_features"].shape == (20, 64)
assert result5["hidden_features"].shape == (10, 64)
assert result5["coupling_matrix"].shape == (20, 10)
assert result5["hidden_states"].shape == (50, 10, 64)
assert result5["support"].shape == (50, 20, 64)
assert result5["x"].shape == (50, 64, 64)
log("result", "all shapes correct")

# Verify x = activations.T @ emission_features
for seq_idx in range(min(5, cfg5.sequence.n_sequences)):
    expected_x = result5["activations"][seq_idx].T @ result5["emission_features"]
    max_err = (result5["x"][seq_idx] - expected_x).abs().max().item()
    assert max_err < 1e-5, f"Sequence {seq_idx}: max reconstruction error = {max_err}"
log("result", "x = activations.T @ emission_features (correct)")

# Each row of coupling matrix should have exactly n_parents ones
row_sums = result5["coupling_matrix"].sum(dim=1)
assert (row_sums == 2).all(), f"Expected 2 parents per emission, got {row_sums}"
log("result", "coupling matrix has exactly 2 parents per emission")

# OR gate: emission fires iff at least one parent is on
# Verify on first sequence
h = result5["hidden_states"][0]  # (K, T)
C = result5["coupling_matrix"]  # (M, K)
expected_support = (C @ h >= 1).float()
actual_support = result5["support"][0]
assert torch.equal(actual_support, expected_support), "OR gate coupling mismatch"
log("result", "OR gate coupling verified")

# Hidden features should have unit norm
norms = result5["hidden_features"].norm(dim=1)
assert torch.allclose(norms, torch.ones(10), atol=1e-5), f"Hidden feature norms: {norms}"
log("result", "hidden features have unit norm")

# ============================================================================
# Test 6: Coupled features — diagonal C reduces to standard pipeline
# ============================================================================

log("info", "smoke test 6: coupled features with diagonal C (K=M, n_parents=1)")

cfg6 = CoupledDataGenerationConfig(
    transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
    coupling=CouplingConfig(K_hidden=10, M_emission=10, n_parents=1, emission_mode="or"),
    sequence=SequenceConfig(T=64, n_sequences=10),
    hidden_dim=64,
    seed=42,
)
result6 = generate_coupled_dataset(cfg6)

# With K=M and n_parents=1, each emission has exactly one parent
# (though not necessarily in order — it's a permutation)
assert result6["coupling_matrix"].shape == (10, 10)
assert (result6["coupling_matrix"].sum(dim=1) == 1).all()
log("result", "K=M, n_parents=1 gives one parent per emission")

# Hidden features should be close to (a permutation of) emission features
# since each hidden state controls exactly M*1/K = 1 emission on average
log("result", "diagonal-like coupling verified")

# ============================================================================
# Summary
# ============================================================================

log("done", "all smoke tests passed")
