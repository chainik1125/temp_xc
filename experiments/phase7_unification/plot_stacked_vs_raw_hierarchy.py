"""Plot the raw < SAE-concat < SAE-meanpool < TXC hierarchy.

Companion plot for `2026-04-29-stacked-sae-control.md`. Visualises
the headline finding that Han's "more candidate features" hypothesis
is rejected, and the full pipeline of representation choices monotonically
adds value.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.plot_stacked_vs_raw_hierarchy
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.phase7_unification._paths import OUT_DIR, PLOTS_DIR
from experiments.phase7_unification.task_sets import HEADLINE as PAPER


def save_figure(fig, path: str, dpi: int = 150, thumb_max_width: int = 288, thumb_dpi: int = 48):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    w_in, _ = fig.get_size_inches()
    thumb_dpi_eff = min(thumb_dpi, int(thumb_max_width / max(w_in, 0.1)))
    fig.savefig(p.with_suffix(".thumb.png"), dpi=thumb_dpi_eff, bbox_inches="tight")


def load_seed42_means(path, arch_filter=None):
    """Return dict[(arch, ...) -> mean across 36 tasks at seed=42]."""
    by_key = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("seed") not in (None, 42): continue
            yield r


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    task_set = PAPER

    # Aggregate per-seed-42 task lists, filtered to PAPER
    stacked = defaultdict(list)        # (arch, K, k_feat) -> auc list
    raw     = defaultdict(list)        # (K, k_feat) -> auc list
    leader  = defaultdict(list)        # (arch, k_feat) -> auc list (mean-pool S=32)

    # SAE-concat (stacked_probing_results.jsonl)
    for r in load_seed42_means(OUT_DIR / "stacked_probing_results.jsonl"):
        if r.get("seed") != 42: continue
        if r.get("task_name") not in task_set: continue
        stacked[(r["arch_id"], r["K_positions"], r["k_feat"])].append(
            r.get("test_auc_flip", r["test_auc"])
        )
    # Raw-concat (raw_concat_probing_results.jsonl) — no seed (architecture-free)
    for r in load_seed42_means(OUT_DIR / "raw_concat_probing_results.jsonl"):
        if r.get("task_name") not in task_set: continue
        raw[(r["K_positions"], r["k_feat"])].append(
            r.get("test_auc_flip", r["test_auc"])
        )
    # Leaderboard mean-pool (probing_results.jsonl, seed=42, S=32)
    for r in load_seed42_means(OUT_DIR / "probing_results.jsonl"):
        if r.get("seed") != 42 or r.get("S") != 32: continue
        if r.get("task_name") not in task_set: continue
        if "skipped" in r: continue
        leader[(r["arch_id"], r["k_feat"])].append(
            r.get("test_auc_flip", r["test_auc"])
        )

    def m(d, key):
        v = d.get(key, [])
        return float(np.mean(v)) if v else float("nan")

    # ─── Hierarchy bar plot, 1 panel per k_feat ───
    # Wider figure + rotated labels to keep text from overlapping
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    for ax, kf in zip(axes, (5, 20)):
        bars = []
        # Group A — raw-activation concat
        bars.append(("raw_concat K=2",         m(raw, (2, kf)),                              "#9aa3b2", "raw"))
        bars.append(("raw_concat K=5",         m(raw, (5, kf)),                              "#9aa3b2", "raw"))
        # Group B — SAE concat
        bars.append(("topk_sae stack K=2",     m(stacked, ("topk_sae", 2, kf)),              "#7aa3d8", "sae_concat"))
        bars.append(("topk_sae stack K=5",     m(stacked, ("topk_sae", 5, kf)),              "#7aa3d8", "sae_concat"))
        bars.append(("tsae_k500 stack K=5",    m(stacked, ("tsae_paper_k500", 5, kf)),       "#5f7fb5", "sae_concat"))
        # Group C — SAE mean-pool S=32
        bars.append(("topk_sae meanpool S=32", m(leader, ("topk_sae", kf)),                  "#3a73c4", "sae_meanpool"))
        bars.append(("tsae_k500 meanpool S=32",m(leader, ("tsae_paper_k500", kf)),           "#3a73c4", "sae_meanpool"))
        # Group D — TXC champion
        if kf == 5:
            txc_arch  = "phase57_partB_h8_bare_multidistance_t8"
            txc_label = "TXC champ (h8_md_t8)"
        else:
            txc_arch  = "txc_bare_antidead_t5"
            txc_label = "TXC champ (bare_t5)"
        bars.append((txc_label, m(leader, (txc_arch, kf)), "#c00000", "txc"))

        labels = [b[0] for b in bars]
        values = [b[1] for b in bars]
        colors = [b[2] for b in bars]

        x = np.arange(len(bars))
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
        ax.set_ylabel("mean test_auc_flip across 16 tasks (PAPER, seed=42)")
        ax.set_title(f"k_feat = {kf}", weight="bold")
        ax.set_ylim(0.66, 0.97)
        ax.grid(axis="y", alpha=0.3)

        # Bar-top value annotations
        for xi, v in zip(x, values):
            if not np.isnan(v):
                ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

        # Group-divider lines + group titles ABOVE bars (not inside ylim)
        for x_div in [1.5, 4.5, 6.5]:
            ax.axvline(x_div, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        # Place group labels at the top of each panel rather than overlapping bottom labels
        for x_pos, txt, color in [
            (0.5, "raw concat",     "#5a6573"),
            (3.0, "SAE concat",     "#5f7fb5"),
            (5.5, "SAE meanpool",   "#3a73c4"),
            (7.0, "TXC",            "#c00000"),
        ]:
            ax.text(x_pos, 0.965, txt, ha="center", fontsize=9, color=color,
                    style="italic", weight="bold")

    fig.suptitle("Probing AUC hierarchy — raw < SAE-concat < SAE-meanpool < TXC\n"
                 "Han's \"more candidate features\" hypothesis rejected; "
                 "more positions ≠ better; mean-pool > concat\n"
                 "(PAPER task set; seed=42)",
                 fontsize=11, weight="bold")
    out_path = PLOTS_DIR / "phase7_stacked_vs_raw_hierarchy.png"
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
