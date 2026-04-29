"""Build the 2-seed leaderboard with σ across seed=1 and seed=42.

Reads `results/probing_results.jsonl`, filters to S=32 + leaderboard
archs, computes:
  - per-task: mean across seeds {1, 42}
  - per-arch: mean of per-task seed-means + standard deviation across
    the per-task per-seed values

Output:
  - prints to stdout (table)
  - writes `results/plots/phase7_leaderboard_2seed.png` + thumbnail

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.build_leaderboard_2seed
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


def save_figure(fig, path: str, dpi: int = 150, thumb_max_width: int = 288, thumb_dpi: int = 48):
    """Save high-res .png + a thumbnail .thumb.png; AI agents read the thumb."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    # Thumbnail
    w_in, _ = fig.get_size_inches()
    thumb_dpi_eff = min(thumb_dpi, int(thumb_max_width / max(w_in, 0.1)))
    fig.savefig(p.with_suffix(".thumb.png"), dpi=thumb_dpi_eff, bbox_inches="tight")


PROBING_PATH = OUT_DIR / "probing_results.jsonl"
LEADERBOARD_ARCHS = [
    "topk_sae",
    "tsae_paper_k20",
    "tsae_paper_k500",
    "tfa_big",
    "txcdr_t5",
    "txcdr_t16",
    "phase5b_subseq_h8",
    "txc_bare_antidead_t5",
    "phase57_partB_h8_bare_multidistance_t8",
    "hill_subseq_h8_T12_s5",
]
SEEDS = (1, 2, 42)
S_FILTER = 32


def load_seed_task_aucs() -> dict:
    """Returns dict[(arch_id, k_feat, seed)] -> dict[task_name -> auc_flip]."""
    out = defaultdict(dict)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != S_FILTER: continue
            if r.get("seed") not in SEEDS: continue
            if r.get("arch_id") not in LEADERBOARD_ARCHS: continue
            if r.get("k_feat") not in (5, 20): continue
            if "skipped" in r: continue
            key = (r["arch_id"], r["k_feat"], r["seed"])
            auc = r.get("test_auc_flip", r["test_auc"])
            out[key][r["task_name"]] = auc
    return out


def summarise(seed_task_aucs):
    """For each (arch, k_feat) compute:
      - per-seed mean across tasks
      - cross-seed mean
      - std across (per-task per-seed) — captures both seed noise + task variance
    """
    rows = []
    for arch in LEADERBOARD_ARCHS:
        for k_feat in (5, 20):
            per_seed_means = []
            all_aucs_pooled = []
            tasks_per_seed = {}
            for seed in SEEDS:
                d = seed_task_aucs.get((arch, k_feat, seed), {})
                tasks_per_seed[seed] = len(d)
                if d:
                    per_seed_means.append(np.mean(list(d.values())))
                    all_aucs_pooled.extend(d.values())
            if not per_seed_means:
                continue
            cross_seed_mean = float(np.mean(per_seed_means))
            # σ across tasks (pooling seeds) — common reporting convention
            pooled_std = float(np.std(all_aucs_pooled, ddof=1)) if len(all_aucs_pooled) > 1 else 0.0
            # σ across seeds at the per-arch summary level
            seed_std = float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0
            rows.append({
                "arch_id": arch,
                "k_feat": k_feat,
                "n_seeds": len(per_seed_means),
                "tasks_per_seed": tasks_per_seed,
                "per_seed_means": per_seed_means,
                "cross_seed_mean": cross_seed_mean,
                "pooled_task_std": pooled_std,
                "seed_only_std": seed_std,
            })
    return rows


def print_table(rows):
    print("=" * 110)
    print("PHASE 7 LEADERBOARD — multi-seed (seed ∈ {1, 2, 42}) at S=32, FLIP-corrected, per-task means")
    print("=" * 110)
    for k_feat in (5, 20):
        print()
        print(f"k_feat = {k_feat}")
        print(f"{'arch':45s} {'mean_AUC':>10s} {'σ_tasks':>9s} {'σ_seeds':>9s} {'n_seeds':>9s}")
        rows_k = sorted([r for r in rows if r["k_feat"] == k_feat],
                        key=lambda r: -r["cross_seed_mean"])
        for r in rows_k:
            print(f"  {r['arch_id']:43s} {r['cross_seed_mean']:>10.4f} "
                  f"{r['pooled_task_std']:>9.4f} {r['seed_only_std']:>9.4f} "
                  f"{r['n_seeds']:>9d}")


def make_plot(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, k_feat in zip(axes, (5, 20)):
        rows_k = sorted([r for r in rows if r["k_feat"] == k_feat],
                        key=lambda r: r["cross_seed_mean"])
        archs = [r["arch_id"] for r in rows_k]
        means = [r["cross_seed_mean"] for r in rows_k]
        # Use seed-only std as the error bar (smaller, more honest at the mean-of-means level)
        # Use pooled task std / sqrt(N_tasks) as the SEM-like band — this captures the
        # uncertainty of the cross-seed mean estimate
        sigmas = [r["seed_only_std"] for r in rows_k]
        ypos = np.arange(len(archs))
        # Color the TXC family vs the SAE family
        colors = []
        for a in archs:
            if a in ("topk_sae", "tsae_paper_k20", "tsae_paper_k500"):
                colors.append("#4472c4")  # blue (per-token SAE)
            elif a == "tfa_big":
                colors.append("#888888")  # grey (TFA — hybrid)
            else:
                colors.append("#c00000")  # red (TXC family / window)
        ax.barh(ypos, means, xerr=sigmas, color=colors,
                error_kw={"ecolor": "k", "alpha": 0.6, "capsize": 3})
        ax.set_yticks(ypos); ax.set_yticklabels(archs, fontsize=9)
        ax.set_xlabel(f"mean test_auc_flip (k_feat={k_feat})")
        ax.set_title(f"k_feat = {k_feat}")
        ax.set_xlim(0.65, 0.95 if k_feat == 5 else 0.97)
        ax.grid(axis="x", alpha=0.3)
        # Add value labels
        for i, m in enumerate(means):
            ax.text(m + 0.005, i, f"{m:.4f}", va="center", fontsize=8)

    fig.suptitle("Phase 7 leaderboard — 3-seed mean ± σ_seeds across {1, 2, 42}",
                 fontsize=12, weight="bold")
    out_path = out_dir / "phase7_leaderboard_multiseed.png"
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main():
    seed_task_aucs = load_seed_task_aucs()
    rows = summarise(seed_task_aucs)
    print_table(rows)
    make_plot(rows, PLOTS_DIR)


if __name__ == "__main__":
    main()
