"""C2 ρ-sweep analysis — Effect 1 vs Effect 2 test (Han 2026-05-06).

Reads c2 leaderboard rows with `eval_cfg.rho` set, aggregates by
(arch, t_label, k_pos, ρ), plots gAUC vs ρ — one line per arch.

Decision rule (Dmitry's Effect framing):
- gAUC roughly flat across ρ → Effect 1 (sample aggregation; weak
  temporal claim).
- gAUC grows with ρ → Effect 2 (temporal pattern detection; strong
  claim defensible).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c2"
RHO_VALUES = [0.0, 0.3, 0.6, 0.7, 0.9]

# Headline trio (per agent_paper directive 7bd38bfd):
HEADLINE_ARCHS: list[tuple[str, str, str]] = [
    # (arch, t_label, display_label)
    ("topk_sae", "default", "TopK-SAE"),
    ("txc_base", "default", "TXC-base T=5"),
    ("txc_pro",  "T=2",     "TXC-pro T=2"),
]

PLOT_STYLE = {
    "TopK-SAE":     {"color": "#000000", "marker": "o", "ls": "-"},
    "TXC-base T=5": {"color": "#882255", "marker": "P", "ls": "-"},
    "TXC-pro T=2":  {"color": "#1f77b4", "marker": "X", "ls": "-"},
}


def _collect():
    """Return {(arch, t_label): {ρ: {k: [(seed, gauc)]}}}."""
    by_cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _read_jsonl(leaderboard_path()):
        if r.get("component") != COMPONENT:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        cfg = r["eval_cfg"]
        # Legacy ρ=0.7 cells (no rho field) are imputed.
        rho = cfg.get("rho", 0.7)
        if rho not in RHO_VALUES:
            continue
        k_pos = cfg.get("k_pos")
        if k_pos not in (1, 5):
            continue
        gauc = r.get("metrics", {}).get("gauc", r.get("metrics", {}).get("auc"))
        if gauc is None:
            continue
        arch = r["arch"]
        t = cfg.get("t_label", "default")
        by_cell[(arch, t)][rho][k_pos].append((r["seed"], float(gauc)))
    return by_cell


def _aggregate(by_cell):
    """Return {(arch, t_label): {ρ: {k_pos: (mean, std, n)}}}."""
    agg = {}
    for key, by_rho in by_cell.items():
        agg[key] = {}
        for rho, by_k in by_rho.items():
            agg[key][rho] = {}
            for k, cells in by_k.items():
                vals = np.array([g for _, g in cells])
                agg[key][rho][k] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    len(vals),
                )
    return agg


def plot_rho_sweep(agg, out_path: Path, *, k_pos: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    rhos_x = np.array(RHO_VALUES)

    for arch, t, label in HEADLINE_ARCHS:
        cell = agg.get((arch, t), {})
        ys = []
        es = []
        xs_valid = []
        for rho in RHO_VALUES:
            stat = cell.get(rho, {}).get(k_pos)
            if stat is None:
                continue
            xs_valid.append(rho)
            ys.append(stat[0])
            es.append(stat[1])
        if not xs_valid:
            continue
        s = PLOT_STYLE.get(label, {"color": "gray", "marker": "o", "ls": "-"})
        ax.errorbar(
            xs_valid, ys, yerr=es,
            label=label, color=s["color"], linestyle=s["ls"],
            marker=s["marker"], markersize=8, capsize=4,
            linewidth=2, alpha=0.9,
        )

    ax.set_xlabel(r"Hidden-chain temporal coherence  $\rho$", fontsize=12)
    ax.set_ylabel(r"gAUC  (global feature recovery)", fontsize=12)
    ax.set_title(rf"Setup A: gAUC vs $\rho$  (k_pos = {k_pos})", fontsize=13)
    ax.set_xticks(RHO_VALUES)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.4, 1.02)
    ax.axvline(0.0, color="gray", ls=":", alpha=0.4, lw=1)
    ax.text(0.01, 0.42, r"i.i.d. tokens (null)", fontsize=8, color="gray", alpha=0.7)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".thumb.png"), bbox_inches="tight", dpi=72)
    plt.close(fig)


def main():
    by_cell = _collect()
    agg = _aggregate(by_cell)

    # Print summary
    print(f"{'arch':14s} {'T':6s} {'ρ':>5s} {'k':>3s}  {'gAUC':>10s} {'n':>3s}")
    print("-" * 60)
    for key in sorted(agg.keys()):
        arch, t = key
        for rho in sorted(agg[key].keys()):
            for k in sorted(agg[key][rho].keys()):
                m, s, n = agg[key][rho][k]
                print(f"{arch:14s} {t:6s} {rho:>5} {k:>3}   {m:.3f}±{s:.3f}  {n:>3}")

    # Plots
    plots_dir = Path("experiments/c2_synthetic_coupled/plots")
    plot_rho_sweep(agg, plots_dir / "c2_rho_sweep_k5.png", k_pos=5)
    plot_rho_sweep(agg, plots_dir / "c2_rho_sweep_k1.png", k_pos=1)
    print(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    main()
