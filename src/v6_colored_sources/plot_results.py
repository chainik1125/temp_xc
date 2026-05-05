"""Plotting for colored-source experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


_TXC_COLOR = "#1f77b4"
_SAE_COLOR = "#d62728"
_ORACLE_COLOR = "#2ca02c"
_RANDOM_COLOR = "#888888"


def plot_phase_transition(stage1_path: Path, out_path: Path) -> None:
    """Stage 1 figure: S_adj vs W at D=1.

    Layout: TXC curve over W (solid), SAE flat baseline (dashed horizontal),
    oracle ceiling (dotted), random floor (dotted gray).
    """
    with open(stage1_path) as f:
        data = json.load(f)

    txc_cells = data["txc_cells"]
    W_grid = sorted(c["W"] for c in txc_cells)
    by_W = {c["W"]: c for c in txc_cells}
    txc_y = [by_W[W]["txc"]["s_adj"] for W in W_grid]

    sae_s = data["sae"]["s_adj"]
    oracle_s = data["oracle"]["s_adj"]
    random_s = data["random"]["s_adj"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(W_grid, txc_y, "o-", color=_TXC_COLOR, label="TXC")
    ax.axhline(sae_s, linestyle="--", color=_SAE_COLOR, label=f"Regular SAE (S_adj={sae_s:.2f})")
    ax.axhline(oracle_s, linestyle=":", color=_ORACLE_COLOR, label=f"Spectral oracle (S_adj={oracle_s:.2f})")
    ax.axhline(random_s, linestyle=":", color=_RANDOM_COLOR, label=f"Random (S_adj={random_s:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(W_grid)
    ax.set_xticklabels([str(W) for W in W_grid])
    ax.set_xlabel("Window length W")
    ax.set_ylabel("Chance-adjusted recovery S_adj")
    ax.set_title("Colored sources, D=1: TXC vs regular SAE")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_phase_transition_by_delay(stage2_path: Path, out_path: Path) -> None:
    """Stage 2 headline: TXC S_adj vs W, one curve per D; SAE at horizontal
    baseline per D (dashed); vertical lines at W=D+1; oracle ceiling per D."""
    with open(stage2_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    D_entries = sorted(data["by_D"], key=lambda e: e["D"])
    n_D = len(D_entries)
    all_W: set[int] = set()

    for i, entry in enumerate(D_entries):
        D = entry["D"]
        color = cmap(i / max(n_D - 1, 1))
        txc_W = sorted(entry["txc_cells"], key=lambda c: c["W"])
        W_grid = [c["W"] for c in txc_W]
        all_W.update(W_grid)
        txc_y = [c["txc"]["s_adj"] for c in txc_W]
        sae_s = entry["sae"]["s_adj"]
        oracle_s = entry["oracle"]["s_adj"]

        ax.plot(W_grid, txc_y, "o-", color=color, label=f"TXC D={D}")
        ax.axhline(sae_s, color=color, linestyle="--", alpha=0.6, label=f"SAE D={D} (S_adj={sae_s:.2f})")
        ax.axvline(D + 1, color=color, linestyle=":", alpha=0.4)
        ax.axhline(oracle_s, color=color, linestyle=":", alpha=0.3)

    all_W_sorted = sorted(all_W)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_W_sorted)
    ax.set_xticklabels([str(W) for W in all_W_sorted])
    ax.set_xlabel("Window length W")
    ax.set_ylabel("Chance-adjusted recovery S_adj")
    ax.set_title("Colored sources: phase transition at W = D + 1")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot colored-source results.")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--results_dir", type=str, default="results/v6_colored_sources"
    )
    parser.add_argument(
        "--out_dir", type=str, default="plots/v6_colored_sources"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    if args.stage == 1:
        plot_phase_transition(
            results_dir / "stage1.json", out_dir / "phase_transition_stage1.png"
        )
    else:
        plot_phase_transition_by_delay(
            results_dir / "stage2.json", out_dir / "phase_transition_stage2.png"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
