"""Plotting for colored-source experiment results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


_TXC_COLOR = "#1f77b4"
_SAE_COLOR = "#d62728"
_ORACLE_COLOR = "#2ca02c"
_RANDOM_COLOR = "#888888"


def _by_W(cells: list[dict]) -> dict[int, dict]:
    """Index cells by window length."""
    return {c["W"]: c for c in cells}


def plot_phase_transition(stage1_path: Path, out_path: Path) -> None:
    """Stage 1 figure: S_adj vs W at D=1."""
    with open(stage1_path) as f:
        data = json.load(f)
    cells = data["cells"]
    indexed = _by_W(cells)
    W_grid = sorted(indexed.keys())

    sae_y = [indexed[W].get("stacked_sae", {}).get("s_adj", float("nan")) for W in W_grid]
    txc_y = [indexed[W].get("txc", {}).get("s_adj", float("nan")) for W in W_grid]
    oracle_s = indexed[W_grid[0]]["oracle"]["s_adj"]
    random_s = indexed[W_grid[0]]["random"]["s_adj"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(W_grid, sae_y, "o--", color=_SAE_COLOR, label="Stacked SAE")
    ax.plot(W_grid, txc_y, "o-", color=_TXC_COLOR, label="TXC")
    ax.axhline(oracle_s, linestyle=":", color=_ORACLE_COLOR, label=f"Oracle (S_adj={oracle_s:.2f})")
    ax.axhline(random_s, linestyle=":", color=_RANDOM_COLOR, label=f"Random (S_adj={random_s:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(W_grid)
    ax.set_xticklabels([str(W) for W in W_grid])
    ax.set_xlabel("Window length W")
    ax.set_ylabel("Chance-adjusted recovery S_adj")
    ax.set_title("Colored sources: phase transition at D=1")
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
    """Stage 2 headline: S_adj vs W, one curve per D, vertical lines at W=D+1."""
    with open(stage2_path) as f:
        data = json.load(f)
    cells = data["cells"]

    by_D: dict[int, dict[int, dict]] = defaultdict(dict)
    for cell in cells:
        by_D[cell["D"]][cell["W"]] = cell

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    D_grid = sorted(by_D.keys())
    n_D = len(D_grid)
    for i, D in enumerate(D_grid):
        color = cmap(i / max(n_D - 1, 1))
        cells_D = by_D[D]
        W_grid = sorted(cells_D.keys())
        txc_y = [cells_D[W].get("txc", {}).get("s_adj", float("nan")) for W in W_grid]
        sae_y = [cells_D[W].get("stacked_sae", {}).get("s_adj", float("nan")) for W in W_grid]
        ax.plot(W_grid, txc_y, "o-", color=color, label=f"TXC D={D}")
        ax.plot(W_grid, sae_y, "x--", color=color, alpha=0.5, label=f"SAE D={D}")
        ax.axvline(D + 1, color=color, linestyle=":", alpha=0.4)
        oracle_s = cells_D[W_grid[0]]["oracle"]["s_adj"]
        ax.axhline(oracle_s, color=color, linestyle="--", alpha=0.3)

    ax.set_xscale("log", base=2)
    all_W = sorted({W for d in by_D.values() for W in d.keys()})
    ax.set_xticks(all_W)
    ax.set_xticklabels([str(W) for W in all_W])
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
