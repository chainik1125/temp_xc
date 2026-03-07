"""Phase 0: Data sanity check for Scheme C Markov chain data.

Verifies that empirical statistics match theoretical predictions:
- Marginal activation rates match pi_i
- Lag-1 autocorrelation matches rho_i
- L0 distribution has correct mean
- Input norms and scaling factor
"""

import math
import torch
import numpy as np

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.markov_data_generation import (
    generate_markov_activations,
    generate_markov_support,
    theoretical_autocorrelation,
)
from src.v2_temporal_schemeC.toy_model import ToyModel

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 10
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.2] * 10
RHO = [0.0, 0.0, 0.0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.95]
SEED = 42
N_SEQUENCES = 10_000


def main():
    device = DEFAULT_DEVICE
    set_seed(SEED)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    print("=" * 60)
    print("Phase 0: Data Sanity Check")
    print("=" * 60)

    # ── 1. Support statistics ──
    print("\n--- 1. Support statistics ---")
    support = generate_markov_support(
        N_SEQUENCES, SEQ_LEN, pi_t, rho_t, device=device
    )  # (N, T, n_features)

    # Marginal activation rates
    empirical_pi = support.mean(dim=(0, 1))  # (n_features,)
    print(f"\n  Marginal activation rates (pi):")
    print(f"  {'Feature':>8} {'Target':>8} {'Empirical':>10} {'Error':>8}")
    for i in range(NUM_FEATURES):
        err = abs(empirical_pi[i].item() - PI[i])
        print(f"  {i:>8d} {PI[i]:>8.3f} {empirical_pi[i].item():>10.4f} {err:>8.4f}")

    # L0 distribution
    l0_per_token = support.sum(dim=-1)  # (N, T)
    mean_l0 = l0_per_token.mean().item()
    std_l0 = l0_per_token.std().item()
    expected_l0 = sum(PI)
    print(f"\n  L0 per token: mean={mean_l0:.4f} (expected {expected_l0:.1f}), "
          f"std={std_l0:.4f}")

    # ── 2. Lag-1 autocorrelation ──
    print("\n--- 2. Lag-1 autocorrelation ---")
    # Compute empirical autocorrelation for each feature
    s = support.float()  # (N, T, n_features)
    s_t = s[:, :-1, :]   # (N, T-1, n)
    s_tp1 = s[:, 1:, :]  # (N, T-1, n)

    # Pearson correlation per feature
    mean_t = s_t.mean(dim=(0, 1))
    mean_tp1 = s_tp1.mean(dim=(0, 1))
    cov = ((s_t - mean_t) * (s_tp1 - mean_tp1)).mean(dim=(0, 1))
    std_t = s_t.std(dim=(0, 1))
    std_tp1 = s_tp1.std(dim=(0, 1))
    empirical_rho = cov / (std_t * std_tp1 + 1e-12)

    theoretical_rho = theoretical_autocorrelation(rho_t, lag=1)

    print(f"\n  {'Feature':>8} {'Target rho':>11} {'Empirical':>10} {'Error':>8}")
    for i in range(NUM_FEATURES):
        err = abs(empirical_rho[i].item() - theoretical_rho[i].item())
        print(f"  {i:>8d} {RHO[i]:>11.3f} {empirical_rho[i].item():>10.4f} {err:>8.4f}")

    # ── 3. Lag-k autocorrelation decay ──
    print("\n--- 3. Autocorrelation decay (feature 9, rho=0.95) ---")
    feature_idx = 9
    lags = [1, 2, 4, 8, 16]
    print(f"  {'Lag':>5} {'Theory':>8} {'Empirical':>10}")
    for lag in lags:
        s_t_lag = s[:, :-lag, feature_idx]
        s_tp_lag = s[:, lag:, feature_idx]
        mean_a = s_t_lag.mean()
        mean_b = s_tp_lag.mean()
        cov_lag = ((s_t_lag - mean_a) * (s_tp_lag - mean_b)).mean()
        std_a = s_t_lag.std()
        std_b = s_tp_lag.std()
        empirical_corr = (cov_lag / (std_a * std_b + 1e-12)).item()
        theory = RHO[feature_idx] ** lag
        print(f"  {lag:>5d} {theory:>8.4f} {empirical_corr:>10.4f}")

    # ── 4. Hidden activations and scaling ──
    print("\n--- 4. Hidden activation norms and scaling ---")
    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()

    with torch.no_grad():
        acts, _ = generate_markov_activations(
            N_SEQUENCES, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)  # (N, T, hidden_dim)

    norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    target_norm = math.sqrt(HIDDEN_DIM)
    scaling_factor = target_norm / mean_norm

    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Mean ||x||: {mean_norm:.4f}")
    print(f"  Std  ||x||: {std_norm:.4f}")
    print(f"  Target (sqrt(d_in)): {target_norm:.4f}")
    print(f"  Scaling factor: {scaling_factor:.4f}")

    # After scaling
    scaled_mean = mean_norm * scaling_factor
    print(f"  After scaling, mean ||x||: {scaled_mean:.4f}")

    # ── 5. Feature direction orthogonality ──
    print("\n--- 5. Feature direction orthogonality ---")
    F_mat = model.feature_directions  # (n_features, hidden_dim)
    F_normed = F_mat / F_mat.norm(dim=1, keepdim=True)
    cos_sim = (F_normed @ F_normed.T)
    # Off-diagonal max
    mask = ~torch.eye(NUM_FEATURES, device=device).bool()
    off_diag = cos_sim[mask].abs()
    print(f"  Max off-diagonal |cos_sim|: {off_diag.max().item():.6f}")
    print(f"  Mean off-diagonal |cos_sim|: {off_diag.mean().item():.6f}")
    print(f"  (Should be ~0 for orthogonal features)")

    print("\n" + "=" * 60)
    print("Sanity check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
