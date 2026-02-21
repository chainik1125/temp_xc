"""Distribution check: confirm that causal mixing changes what the SAE sees.

Key insight: mixing doesn't change the expected projection (both global and
local have E[projection] = p regardless of gamma). The difference is in what
happens when a feature IS active:

- Global feature active: present at all positions, so the running mean
  reinforces it. Projection stays at ~1.0 regardless of position.
- Local feature active at position t only: the running mean dilutes it.
  Projection at position t drops to ~(1 - gamma * t/(t+1)).

This script measures projections CONDITIONAL on feature activation to show
this differential effect.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.seed import set_seed
from src.v1_temporal.temporal_data_generation import (
    generate_temporal_batch,
    generate_temporal_features,
)
from src.v1_temporal.temporal_toy_model import TemporalToyModel

matplotlib.use("Agg")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "distribution_check")


def run():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_global = 5
    n_local = 5
    num_features = n_global + n_local
    hidden_dim = 20
    seq_len = 4
    p = 0.4
    batch_size = 20_000

    global_probs = torch.tensor([p] * n_global)
    local_probs = torch.tensor([p] * n_local)

    gammas = [0.0, 0.1, 0.3, 0.5, 0.7]

    # Results: gamma -> {position -> {global_proj, local_proj}}
    results = {}

    for gamma in gammas:
        model = TemporalToyModel(
            num_features, hidden_dim, gamma=gamma, ortho_num_steps=1000,
        )
        model = model.to(device)

        with torch.no_grad():
            # Generate binary features (to know which are active)
            binary_feats = generate_temporal_features(
                batch_size, seq_len, global_probs, local_probs, device=device,
            )  # (batch, T, num_features)

            # Use unit magnitudes so projection = 1.0 when feature is active and no mixing
            feats_with_mag = binary_feats.float()

            hidden = model(feats_with_mag)  # (batch, T, d)

            # Project back onto feature directions
            F = model.feature_directions.to(device)  # (num_features, d)
            F_normed = F / F.norm(dim=-1, keepdim=True)
            projections = torch.einsum("btd,fd->btf", hidden, F_normed)
            # projections shape: (batch, T, num_features)

            results[gamma] = {}
            for t in range(seq_len):
                # For global features: conditional on being active (same at all positions)
                global_active_mask = binary_feats[:, 0, :n_global] > 0  # (batch, n_global)
                global_projs_at_t = projections[:, t, :n_global]  # (batch, n_global)
                # Mean projection for active global features
                if global_active_mask.sum() > 0:
                    global_cond = global_projs_at_t[global_active_mask].mean().item()
                else:
                    global_cond = float("nan")

                # For local features: conditional on being active at THIS position
                local_active_mask = binary_feats[:, t, n_global:] > 0  # (batch, n_local)
                local_projs_at_t = projections[:, t, n_global:]  # (batch, n_local)
                if local_active_mask.sum() > 0:
                    local_cond = local_projs_at_t[local_active_mask].mean().item()
                else:
                    local_cond = float("nan")

                results[gamma][t] = {
                    "global_cond_proj": global_cond,
                    "local_cond_proj": local_cond,
                }

    # --- Print summary table ---
    print("\nConditional projection: mean projection onto feature direction")
    print("given that the feature IS active\n")
    print(f"{'gamma':>6} | {'pos':>3} | {'global (active)':>16} | {'local (active)':>16} | {'expected local':>16}")
    print("-" * 65)
    for gamma in gammas:
        for t in range(seq_len):
            r = results[gamma][t]
            # Expected: for local feature active only at t, projection =
            # (1-gamma) * 1 + gamma * (1/(t+1)) * 1 = 1 - gamma * t/(t+1)
            expected_local = 1 - gamma * t / (t + 1)
            print(f"{gamma:>6.1f} | {t+1:>3} | {r['global_cond_proj']:>16.4f} | {r['local_cond_proj']:>16.4f} | {expected_local:>16.4f}")
        print("-" * 65)

    # --- Plot: conditional projection vs position for each gamma ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    positions = list(range(1, seq_len + 1))

    ax = axes[0]
    for gamma in gammas:
        vals = [results[gamma][t]["global_cond_proj"] for t in range(seq_len)]
        ax.plot(positions, vals, "o-", label=f"gamma={gamma}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean projection (active features)")
    ax.set_title("Global features: projection given active")
    ax.legend()
    ax.set_ylim(0, 1.2)

    ax = axes[1]
    for gamma in gammas:
        vals = [results[gamma][t]["local_cond_proj"] for t in range(seq_len)]
        ax.plot(positions, vals, "o-", label=f"gamma={gamma}")
    # Also plot theoretical for each gamma
    for gamma in [0.3, 0.7]:
        theoretical = [1 - gamma * t / (t + 1) for t in range(seq_len)]
        ax.plot(positions, theoretical, "--", alpha=0.5, color="gray")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean projection (active features)")
    ax.set_title("Local features: projection given active")
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "conditional_projection.png"), dpi=150)
    plt.close()

    # --- Plot: global vs local gap as function of gamma at position T ---
    fig, ax = plt.subplots(figsize=(8, 5))
    last_t = seq_len - 1
    g_vals = [results[g][last_t]["global_cond_proj"] for g in gammas]
    l_vals = [results[g][last_t]["local_cond_proj"] for g in gammas]
    ax.plot(gammas, g_vals, "o-", label="global (active)", color="steelblue")
    ax.plot(gammas, l_vals, "s-", label="local (active)", color="coral")
    ax.set_xlabel("gamma")
    ax.set_ylabel(f"Mean projection at position {seq_len}")
    ax.set_title(f"Global vs local feature signal at last position (T={seq_len})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gap_vs_gamma.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
