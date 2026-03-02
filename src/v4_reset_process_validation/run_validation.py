"""Reset process validation: verify that the Markov chain data generation
matches theoretical predictions for autocorrelation, stationarity, and
temporal structure.

Four sub-experiments:
  1. Support heatmaps (visual block-length intuition)
  2. Empirical vs theoretical autocorrelation
  3. Activation heatmaps (support x magnitude)
  4. Stationary probability check
"""

# %% Imports

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from src.shared.plotting import save_figure
from src.shared.temporal_support import generate_support_reset
from src.utils.logging import log
from src.utils.seed import set_seed

matplotlib.use("Agg")

# %% Configuration

SEED = 42
K = 30  # number of features
T = 256  # sequence length
P = 0.05  # sparsity (stationary probability)
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
N_SEQUENCES = 1000  # for empirical statistics
MAX_LAG = 30  # for autocorrelation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = str(_PROJECT_ROOT / "results" / "reset_process_validation")

# ============================================================================
# Exp 1: Support heatmaps
# ============================================================================

# %% Generate and plot support heatmaps

log("exp1", "support heatmaps")
set_seed(SEED)

fig, axes = plt.subplots(
    len(LAMBDAS), 1,
    figsize=(12, 1.5 * len(LAMBDAS)),
    sharex=True,
)

for i, lam in enumerate(LAMBDAS):
    rng = torch.Generator().manual_seed(SEED)
    support = generate_support_reset(K, T, P, lam, rng)
    # Plot feature 0 as a single-row binary heatmap
    axes[i].imshow(
        support[0:1].numpy(),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[i].set_ylabel(f"λ={lam}", rotation=0, labelpad=40, va="center")
    axes[i].set_yticks([])

axes[-1].set_xlabel("Position t")
fig.suptitle("Support Heatmap (feature 0) across λ values", fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, f"{RESULTS_DIR}/support_heatmaps")
log("plot", "saved support_heatmaps")

# ============================================================================
# Exp 2: Empirical vs theoretical autocorrelation
# ============================================================================

# %% Compute and plot autocorrelation

log("exp2", "empirical vs theoretical autocorrelation", n_sequences=N_SEQUENCES)

n_cols = 4
n_rows = (len(LAMBDAS) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes_flat = axes.flatten()

lags = torch.arange(0, MAX_LAG + 1)

for i, lam in enumerate(LAMBDAS):
    rng = torch.Generator().manual_seed(SEED)
    # Generate N_SEQUENCES sequences, extract feature 0
    all_support = torch.stack([
        generate_support_reset(K, T, P, lam, rng)[0]  # feature 0, shape (T,)
        for _ in range(N_SEQUENCES)
    ])  # shape (N_SEQUENCES, T)

    # Empirical autocorrelation: Corr(s_t, s_{t+tau})
    mean_s = all_support.mean()
    var_s = all_support.var()
    empirical_acf = torch.zeros(MAX_LAG + 1)
    for tau in range(MAX_LAG + 1):
        if var_s > 0:
            cov = ((all_support[:, :T - tau] - mean_s) * (all_support[:, tau:] - mean_s)).mean()
            empirical_acf[tau] = cov / var_s
        else:
            empirical_acf[tau] = 1.0 if tau == 0 else 0.0

    # Theoretical: (1 - lam)^tau
    theoretical_acf = (1 - lam) ** lags.float()

    ax = axes_flat[i]
    ax.plot(lags.numpy(), theoretical_acf.numpy(), "r-", linewidth=2, label="theory")
    ax.plot(lags.numpy(), empirical_acf.numpy(), "b.", markersize=4, label="empirical")
    ax.set_title(f"λ={lam}")
    ax.set_xlabel("lag τ")
    ax.set_ylabel("autocorrelation")
    ax.set_ylim(-0.15, 1.1)
    ax.legend(fontsize=8)

# Hide unused subplots
for j in range(len(LAMBDAS), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Empirical vs Theoretical Autocorrelation (feature 0)", fontsize=14)
plt.tight_layout()
save_figure(fig, f"{RESULTS_DIR}/autocorrelation")
log("plot", "saved autocorrelation")

# ============================================================================
# Exp 3: Activation heatmaps
# ============================================================================

# %% Generate and plot activation heatmaps

log("exp3", "activation heatmaps")
set_seed(SEED)

fig, axes = plt.subplots(
    len(LAMBDAS), 1,
    figsize=(14, 2 * len(LAMBDAS)),
    sharex=True,
    gridspec_kw={"right": 0.88},
)

vmax_global = 0.0
activation_data = []
for lam in LAMBDAS:
    rng = torch.Generator().manual_seed(SEED)
    support = generate_support_reset(K, T, P, lam, rng)  # (K, T)
    # Half-normal magnitudes: |N(0, 1)|
    magnitudes = torch.randn(K, T, generator=rng).abs()
    activations = support * magnitudes  # (K, T)
    activation_data.append(activations)
    vmax_global = max(vmax_global, activations.max().item())

for i, (lam, activations) in enumerate(zip(LAMBDAS, activation_data)):
    im = axes[i].imshow(
        activations.numpy(),
        aspect="auto",
        cmap="hot",
        vmin=0,
        vmax=vmax_global,
        interpolation="nearest",
    )
    axes[i].set_ylabel(f"λ={lam}", rotation=0, labelpad=40, va="center")
    axes[i].set_yticks(range(K))
    axes[i].set_yticklabels([f"f{j}" for j in range(K)], fontsize=8)

axes[-1].set_xlabel("Position t")
fig.suptitle("Activation Heatmap (all features) across λ values", fontsize=14)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="activation magnitude")
save_figure(fig, f"{RESULTS_DIR}/activation_heatmaps")
log("plot", "saved activation_heatmaps")

# ============================================================================
# Exp 4: Stationary probability check
# ============================================================================

# %% Compute and plot stationary probability

log("exp4", "stationary probability check", n_sequences=N_SEQUENCES)

fig, ax = plt.subplots(figsize=(10, 5))

for lam in LAMBDAS:
    rng = torch.Generator().manual_seed(SEED)
    # Generate N_SEQUENCES sequences with all K features
    all_support = torch.stack([
        generate_support_reset(K, T, P, lam, rng)
        for _ in range(N_SEQUENCES)
    ])  # shape (N_SEQUENCES, K, T)

    # Empirical firing probability at each position t, averaged over features and sequences
    prob_by_t = all_support.mean(dim=(0, 1))  # shape (T,)
    ax.plot(range(T), prob_by_t.numpy(), label=f"λ={lam}", alpha=0.7)

ax.axhline(y=P, color="k", linestyle="--", linewidth=2, label=f"p={P}")
ax.set_xlabel("Position t")
ax.set_ylabel("Empirical firing probability")
ax.set_title("Stationary Probability Check across λ values")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, max(0.15, 2 * P))
plt.tight_layout()
save_figure(fig, f"{RESULTS_DIR}/stationary_probability")
log("plot", "saved stationary_probability")

# %% Summary

log("summary", "reset process validation complete", results_dir=RESULTS_DIR)
