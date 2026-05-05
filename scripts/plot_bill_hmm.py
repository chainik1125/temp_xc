"""Generate visualizations for docs/bill/results/HMM-Spec.md.

Produces three figures that make the spec concrete by showing actual
samples from the generators it describes:

  - hmm_spec_chains_three_arch.png — hidden chains at ρ ∈ {0.0, 0.6, 0.9}
    (three-arch sweep), 10 features × 64 timesteps, spike train.
  - hmm_spec_chains_hmm_denoising.png — heterogeneous-ρ chains at the
    four groups of the HMM-denoising bench (10 features per group).
  - hmm_spec_emission_noise.png — h(t) vs observed s(t) overlay for one
    chain at p_B=0.625, showing the missed-detection mechanism.

Outputs into the same dir as the docs/bill/results/ writeup so the doc
can reference them with relative paths.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "src")

from temporal_bench.data.markov import (
    generate_markov_support,
    generate_markov_support_hetero,
    emit,
)


OUT_DIR = "docs/bill/results"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
T = 64


def _spike_panel(ax, h: torch.Tensor, title: str, ylabel: str | None = None):
    """h: (n_features, T) binary. Render as a black/white spike train."""
    ax.imshow(h.numpy(), aspect="auto", interpolation="nearest",
              cmap="binary", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("time step t")


def plot_three_arch_chains() -> None:
    """ρ-sweep used in the three-arch bench: shared ρ, π=0.05, 10 features × T=64."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    rng = torch.Generator().manual_seed(SEED)
    for i, rho in enumerate([0.0, 0.6, 0.9]):
        # Need a single fresh generator per row so rows are visually different.
        sub_rng = torch.Generator().manual_seed(SEED + i)
        h = generate_markov_support(
            n_features=10, T=T, pi=0.05, rho=rho,
            n_sequences=1, generator=sub_rng,
        )[0]  # (10, T)
        avg = h.float().mean().item()
        _spike_panel(
            axes[i], h,
            title=f"ρ = {rho}   (mean firing rate = {avg:.3f}, expected = 0.05)",
            ylabel=f"feature\n(ρ = {rho})",
        )
    fig.suptitle(
        "Three-arch bench: 10 independent 2-state Markov chains, "
        "shared ρ, π = 0.05, T = 64\n"
        "(black = on, white = off)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hmm_spec_chains_three_arch.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_hmm_denoising_chains() -> None:
    """Heterogeneous-ρ groups used in the HMM-denoising bench: 4 groups × 10 features each."""
    GROUPS = [0.1, 0.4, 0.7, 0.95]
    GROUP_SIZE = 10
    rhos = torch.tensor([r for r in GROUPS for _ in range(GROUP_SIZE)])

    rng = torch.Generator().manual_seed(SEED)
    h = generate_markov_support_hetero(
        rhos=rhos, T=T, pi=0.15, n_sequences=1, generator=rng,
    )[0]  # (40, T)

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for i, rho in enumerate(GROUPS):
        rows = h[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
        avg = rows.float().mean().item()
        _spike_panel(
            axes[i], rows,
            title=f"ρ = {rho}   (mean firing rate = {avg:.3f}, expected = 0.15)",
            ylabel=f"feature\n(group {i+1})",
        )
    fig.suptitle(
        "HMM denoising bench: 40 chains in 4 ρ groups (10 each), "
        "shared π = 0.15, T = 64",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hmm_spec_chains_hmm_denoising.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_emission_noise() -> None:
    """Show h(t) vs s(t) overlay for one persistent chain with p_B = 0.625."""
    rng = torch.Generator().manual_seed(SEED)
    # Use ρ = 0.95 for a chain with long on-runs so missed detections are
    # easy to spot; π=0.15 to match HMM-denoising bench.
    T_long = 200
    h = generate_markov_support(
        n_features=1, T=T_long, pi=0.15, rho=0.95,
        n_sequences=1, generator=rng,
    )[0, 0]  # (T_long,)
    s = emit(h, p_A=0.0, p_B=0.625, generator=rng)

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
    ax.fill_between(range(T_long), 0, h.numpy(),
                    step="post", alpha=0.4, color="#1f77b4",
                    label="hidden state h(t)")
    ax.fill_between(range(T_long), 0, s.numpy(),
                    step="post", alpha=0.7, color="#d62728",
                    label="observed emission s(t),  p_B = 0.625")
    on_h = int(h.sum().item())
    on_s = int(s.sum().item())
    missed = int(((h == 1) & (s == 0)).sum().item())
    ax.set_title(
        f"Stochastic emission (HMM-denoising bench): "
        f"{missed} missed detections out of {on_h} hidden-on tokens "
        f"({100 * missed / max(on_h, 1):.0f}% miss rate, observed {on_s})",
        fontsize=10,
    )
    ax.set_xlabel("time step t")
    ax.set_ylabel("state")
    ax.set_yticks([0, 1])
    ax.set_xlim(0, T_long - 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hmm_spec_emission_noise.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_three_arch_chains()
    plot_hmm_denoising_chains()
    plot_emission_noise()


if __name__ == "__main__":
    main()
