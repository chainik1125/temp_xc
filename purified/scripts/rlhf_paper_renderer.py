"""Render the HH-RLHF case-study figures in c7-paper style.

Reads per-architecture ``top_features.json`` files (sourced from
``origin/han-phase7-unification`` and committed under
``purified/results/case_studies/hh_rlhf/<arch>/top_features.json``)
and emits two PNGs:

  rlhf_summary.png  — 1x2 panel:
      left:  vertical stacked bars, top-20 features per arch split into
             semantic / mixed / spurious tiers (counts annotated in
             segments).
      right: per-arch share of total top-20 |diff| coming from the
             semantic tier.
      Single legend above the row (horizontal). No embedded titles.

  rlhf_scatter.png  — 1x4 panel, one per arch:
      x: |diff|, y: |length-Pearson r|. Points coloured by tier.
      Horizontal dotted lines at |r| = 0.2 (semantic boundary) and
      |r| = 0.5 (spurious boundary). Per-panel x-axis label is the
      arch name. Single shared legend above the row.

Usage:
    .venv/bin/python -m scripts.rlhf_paper_renderer \\
        --data-dir results/case_studies/hh_rlhf \\
        --output-dir /workspace/aniket/temp_xc_paper/purified/docs/aniket/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEMANTIC_THRESHOLD = 0.2
SPURIOUS_THRESHOLD = 0.5

# Tier colours match the c7_contingency stacked-bar palette (see
# experiments/c7_backtracking/analyze_optimal.py:write_plots).
TIER_COLOR = {
    "semantic": "#1f9e58",
    "mixed":    "#e89146",
    "spurious": "#c33",
}
TIER_LABEL = {
    "semantic": r"semantic ($|r|<0.2$)",
    "mixed":    r"mixed ($0.2 \leq |r| < 0.5$)",
    "spurious": r"spurious ($|r| \geq 0.5$)",
}

# The four locked architectures for this case study, in display order.
ARCH_ORDER = [
    "topk_sae",
    "tsae_paper_k500",
    "tsae_paper_k20",
    "agentic_txc_02",
]
ARCH_LABEL = {
    "topk_sae":         "TopK SAE\n(per-token, $k{=}500$)",
    "tsae_paper_k500":  "T-SAE\n(per-token, $k{=}500$)",
    "tsae_paper_k20":   "T-SAE\n(paper, $k{=}20$)",
    "agentic_txc_02":   "TXC\n(matryoshka, $T{=}5$)",
}


def categorise(r: float) -> str:
    if abs(r) < SEMANTIC_THRESHOLD:
        return "semantic"
    if abs(r) >= SPURIOUS_THRESHOLD:
        return "spurious"
    return "mixed"


def load_arch(data_dir: Path, arch_id: str) -> list[dict]:
    p = data_dir / arch_id / "top_features.json"
    if not p.exists():
        return []
    blob = json.loads(p.read_text())
    return blob.get("features", [])


def render_summary(arch_data: dict[str, list[dict]], out_path: Path) -> None:
    """Two panels: top-20 stacked counts + semantic |diff| share."""
    archs = [a for a in ARCH_ORDER if arch_data.get(a)]
    n = len(archs)
    if n == 0:
        return

    # --- compute per-arch counts and semantic-mass shares -----------
    counts: dict[str, dict[str, int]] = {}
    sem_share: dict[str, float] = {}
    for a in archs:
        feats = arch_data[a][:20]
        c = {"semantic": 0, "mixed": 0, "spurious": 0}
        sem_mass = 0.0
        total_mass = 0.0
        for f in feats:
            tier = categorise(float(f["length_pearson_r"]))
            c[tier] += 1
            mag = abs(float(f["diff"]))
            total_mass += mag
            if tier == "semantic":
                sem_mass += mag
        counts[a] = c
        sem_share[a] = (sem_mass / total_mass) if total_mass > 0 else 0.0

    # --- two-panel canvas ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8),
                             gridspec_kw={"width_ratios": [1.0, 0.85]})
    ax_stack, ax_share = axes
    x_pos = np.arange(n)
    bar_w = 0.65

    # stacked bars: semantic (bottom) → mixed → spurious (top)
    handles: list = []
    bottom = np.zeros(n, dtype=float)
    for tier in ("semantic", "mixed", "spurious"):
        heights = np.array([counts[a][tier] for a in archs], dtype=float)
        bars = ax_stack.bar(x_pos, heights, width=bar_w,
                            color=TIER_COLOR[tier], edgecolor="#222",
                            linewidth=0.4, bottom=bottom,
                            label=TIER_LABEL[tier])
        handles.append(bars)
        # in-segment value labels (only when segment >= 2 to avoid clutter)
        for xi, hi, bi in zip(x_pos, heights, bottom):
            if hi >= 2:
                ax_stack.text(xi, bi + hi / 2, f"{int(hi)}",
                              ha="center", va="center",
                              fontsize=9, color="white" if tier in ("semantic", "spurious") else "#222",
                              fontweight="bold")
        bottom += heights

    ax_stack.set_xticks(x_pos)
    ax_stack.set_xticklabels([ARCH_LABEL[a] for a in archs],
                             rotation=20, ha="right", fontsize=8.5)
    ax_stack.set_ylabel("count of top-$20$ features", fontsize=9)
    ax_stack.set_ylim(0, 22)
    ax_stack.tick_params(axis="y", labelsize=8)
    ax_stack.spines["top"].set_visible(False)
    ax_stack.spines["right"].set_visible(False)

    # right panel: semantic |diff| share
    arch_colors = [TIER_COLOR["semantic"]] * n
    bars_share = ax_share.bar(x_pos, [sem_share[a] for a in archs],
                              width=bar_w, color="#3b73d6", alpha=0.92,
                              edgecolor="#222", linewidth=0.4)
    for xi, a, b in zip(x_pos, archs, bars_share):
        ax_share.text(xi, sem_share[a] + 0.02,
                      f"{sem_share[a]*100:.0f}%",
                      ha="center", va="bottom", fontsize=8.5, color="#222",
                      fontweight="bold")
    ax_share.set_xticks(x_pos)
    ax_share.set_xticklabels([ARCH_LABEL[a] for a in archs],
                             rotation=20, ha="right", fontsize=8.5)
    ax_share.set_ylabel(r"share of top-$20$ $|\Delta|$ from semantic", fontsize=9)
    ax_share.set_ylim(0, 1.0)
    ax_share.tick_params(axis="y", labelsize=8)
    ax_share.spines["top"].set_visible(False)
    ax_share.spines["right"].set_visible(False)

    # shared horizontal legend above the row
    fig.legend(
        handles=[h[0] for h in handles],
        labels=[TIER_LABEL[t] for t in ("semantic", "mixed", "spurious")],
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=3, frameon=False, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def render_scatter(arch_data: dict[str, list[dict]], out_path: Path) -> None:
    """1x4 row of |diff| vs |r| scatters, one per arch."""
    archs = [a for a in ARCH_ORDER if arch_data.get(a)]
    n = len(archs)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 0.6, 3.4),
                             sharey=True)
    if n == 1:
        axes = [axes]

    handles_for_legend = {}

    for ax, arch in zip(axes, archs):
        feats = arch_data[arch]
        diffs = np.array([abs(float(f["diff"])) for f in feats])
        rs = np.array([abs(float(f["length_pearson_r"])) for f in feats])
        tiers = [categorise(float(f["length_pearson_r"])) for f in feats]

        for tier in ("semantic", "mixed", "spurious"):
            mask = np.array([t == tier for t in tiers])
            if not mask.any():
                continue
            sc = ax.scatter(diffs[mask], rs[mask],
                            color=TIER_COLOR[tier], alpha=0.85,
                            edgecolors="#222", linewidths=0.4,
                            s=36, label=TIER_LABEL[tier])
            handles_for_legend.setdefault(tier, sc)

        # tier-boundary lines
        ax.axhline(SEMANTIC_THRESHOLD, color="#888", linestyle=":",
                   linewidth=0.8)
        ax.axhline(SPURIOUS_THRESHOLD, color="#888", linestyle=":",
                   linewidth=0.8)

        ax.set_xlabel(ARCH_LABEL[arch], fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$|\,\mathrm{Pearson}\;r\,|$ vs response length",
                       fontsize=9)
    fig.text(0.5, -0.02, r"$|\mu_{\mathrm{rejected}} - \mu_{\mathrm{chosen}}|$",
             ha="center", va="top", fontsize=9.5)

    # shared legend above the row
    fig.legend(
        handles=[handles_for_legend[t] for t in ("semantic", "mixed", "spurious") if t in handles_for_legend],
        labels=[TIER_LABEL[t] for t in ("semantic", "mixed", "spurious") if t in handles_for_legend],
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=3, frameon=False, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(*, data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arch_data = {a: load_arch(data_dir, a) for a in ARCH_ORDER}
    render_summary(arch_data, output_dir / "rlhf_summary.png")
    render_scatter(arch_data, output_dir / "rlhf_scatter.png")
    print(f"[rlhf_paper] wrote rlhf_summary.png + rlhf_scatter.png → {output_dir}")


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli():
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "RLHF (HH-RLHF case study) paper figure renderer. "
        "Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--data-dir", type=Path,
        default=root / "results" / "case_studies" / "hh_rlhf",
        help="Per-arch top_features.json root (default: "
             "purified/results/case_studies/hh_rlhf/).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs" / "rlhf",
        help="Output directory (default: purified/figs/rlhf/).",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    cli()
