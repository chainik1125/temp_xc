"""Cross-arch summary + plots for HH-RLHF case study C.i.

Reads the per-arch `top_features.json` produced by `label_top_features.py`
and emits:

  results/case_studies/plots/phase7_hh_rlhf_scatter.png
    2×2 grid of (|diff|, |length_pearson_r|) scatter plots — one panel
    per arch — with the top-3 highest-|diff| features annotated by their
    Haiku autointerp labels. Horizontal line at |r|=0.4 marks the paper's
    implicit length-spurious cutoff.

  results/case_studies/plots/phase7_hh_rlhf_summary.png
    Stacked bar chart per arch: among top-20 features ranked by |rejected
    - chosen|, how many fall in each Pearson-|r|-with-length tier
    (semantic <0.2, mixed [0.2, 0.5), spurious ≥0.5).

  Markdown writeup printed to stdout summarising the cross-arch findings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, PLOTS_DIR, banner
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS,
)
from src.plotting.save_figure import save_figure


SEMANTIC_THRESHOLD = 0.2          # |r| below this → "semantic" (low length confound)
SPURIOUS_THRESHOLD = 0.5          # |r| at or above this → "length-spurious"

# Display labels (more readable than the arch_id slugs in plot titles).
DISPLAY = {
    "topk_sae": "TopKSAE\n(per-token, k=500)",
    "tsae_paper_k500": "T-SAE\n(per-token, k=500)",
    "tsae_paper_k20": "T-SAE\n(paper-faithful, k=20)",
    "agentic_txc_02": "TXC\n(matryoshka multi-scale, T=5)",
    "mlc_contrastive_alpha100_batchtopk": "MLC\n(contrastive, k=500)",
    "phase5b_subseq_h8": "SubseqH8\n(T_max=10, k=500)",
    "phase57_partB_h8_bare_multidistance_t5": "H8\n(multi-distance, T=5)",
}


def _load_arch_data(arch_id: str) -> dict | None:
    p = CASE_STUDIES_DIR / "hh_rlhf" / arch_id / "top_features.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _categorise(r: float) -> str:
    if abs(r) < SEMANTIC_THRESHOLD:
        return "semantic"
    if abs(r) >= SPURIOUS_THRESHOLD:
        return "spurious"
    return "mixed"


CATEGORY_COLOR = {"semantic": "#2ca02c", "mixed": "#ff9900", "spurious": "#d62728"}


def make_scatter(arch_data_list: list[tuple[str, dict]], out_path: Path) -> None:
    n = len(arch_data_list)
    cols = 2
    rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(13, 5.5 * rows + 0.6))
    axs = np.atleast_2d(axs)
    for i, (arch_id, data) in enumerate(arch_data_list):
        ax = axs[i // cols, i % cols]
        feats = data["features"]
        diffs = np.array([abs(f["diff"]) for f in feats])
        rs = np.array([abs(f["length_pearson_r"]) for f in feats])
        cats = np.array([_categorise(f["length_pearson_r"]) for f in feats])
        for cat in ("semantic", "mixed", "spurious"):
            mask = cats == cat
            ax.scatter(
                diffs[mask], rs[mask],
                c=CATEGORY_COLOR[cat], s=70, alpha=0.85,
                edgecolors="black", linewidths=0.5, label=cat,
            )
        # Annotate top-3 by |diff|.
        top3 = np.argsort(-diffs)[:3]
        for ti in top3:
            f = feats[int(ti)]
            label = f["label"][:38]
            ax.annotate(
                label, (diffs[ti], rs[ti]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=7.5, color="#222", alpha=0.85,
            )
        ax.axhline(0.4, color="grey", linestyle="--", linewidth=0.6)
        ax.set_xlabel("|mean(rejected) − mean(chosen)|")
        ax.set_ylabel("|Pearson r| (feature_diff vs length_diff)")
        ax.set_title(DISPLAY.get(arch_id, arch_id))
        ax.set_ylim(-0.02, max(0.62, rs.max() * 1.05 + 0.02))
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
    # Hide unused subplots if odd count.
    for j in range(n, rows * cols):
        axs[j // cols, j % cols].axis("off")
    fig.suptitle(
        "HH-RLHF top-20 features: |diff| vs |length-Pearson r| per arch",
        y=0.995, fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path))
    plt.close(fig)


def make_summary_bar(arch_data_list: list[tuple[str, dict]], out_path: Path) -> None:
    archs = [a for a, _ in arch_data_list]
    sem_counts, mix_counts, spur_counts = [], [], []
    sem_diff_share = []
    for _, data in arch_data_list:
        feats = data["features"]
        cats = [_categorise(f["length_pearson_r"]) for f in feats]
        sem_counts.append(cats.count("semantic"))
        mix_counts.append(cats.count("mixed"))
        spur_counts.append(cats.count("spurious"))
        # share of total |diff| from semantic features
        diffs = np.array([abs(f["diff"]) for f in feats])
        sem_mask = np.array([c == "semantic" for c in cats])
        sem_diff_share.append(float(diffs[sem_mask].sum() / diffs.sum()) if diffs.sum() > 0 else 0.0)

    x = np.arange(len(archs))
    width = 0.6
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 2]})

    ax_top.bar(x, sem_counts, width, label="semantic (|r|<0.2)",
               color=CATEGORY_COLOR["semantic"])
    ax_top.bar(x, mix_counts, width, bottom=sem_counts,
               label=f"mixed (0.2 ≤ |r| < {SPURIOUS_THRESHOLD})",
               color=CATEGORY_COLOR["mixed"])
    bottom_spur = np.array(sem_counts) + np.array(mix_counts)
    ax_top.bar(x, spur_counts, width, bottom=bottom_spur,
               label=f"spurious (|r| ≥ {SPURIOUS_THRESHOLD})",
               color=CATEGORY_COLOR["spurious"])
    ax_top.set_ylabel("count of top-20 features")
    ax_top.set_title("HH-RLHF top-20 features by length-spurious tier")
    ax_top.legend(fontsize=9, loc="lower right")
    ax_top.grid(axis="y", alpha=0.25)
    for i in range(len(archs)):
        ax_top.text(x[i], sem_counts[i] / 2, str(sem_counts[i]),
                    ha="center", va="center", color="white", fontsize=10, weight="bold")
        if mix_counts[i] > 0:
            ax_top.text(x[i], sem_counts[i] + mix_counts[i] / 2, str(mix_counts[i]),
                        ha="center", va="center", color="black", fontsize=10, weight="bold")
        if spur_counts[i] > 0:
            ax_top.text(x[i], bottom_spur[i] + spur_counts[i] / 2, str(spur_counts[i]),
                        ha="center", va="center", color="white", fontsize=10, weight="bold")

    ax_bot.bar(x, sem_diff_share, width, color="#1f77b4", alpha=0.85)
    ax_bot.set_ylabel("semantic |diff| share")
    ax_bot.set_ylim(0, max(1.0, max(sem_diff_share) * 1.1))
    ax_bot.grid(axis="y", alpha=0.25)
    ax_bot.set_title("Share of total top-20 |diff| coming from low-|r| (semantic) features")
    for i, v in enumerate(sem_diff_share):
        ax_bot.text(x[i], v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([DISPLAY.get(a, a) for a in archs], fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path))
    plt.close(fig)
    return list(zip(archs, sem_counts, mix_counts, spur_counts, sem_diff_share))


def emit_markdown_writeup(summary_rows: list[tuple], arch_data_list: list[tuple[str, dict]]) -> str:
    lines = []
    lines.append("## C.i HH-RLHF dataset understanding — top-20 features per arch (Stage 1)")
    lines.append("")
    lines.append("| arch | semantic (|r|<0.2) | mixed | spurious (|r|≥0.5) | semantic |diff| share |")
    lines.append("|---|---|---|---|---|")
    for arch, sem, mix, spur, share in summary_rows:
        lines.append(
            f"| `{arch}` | {sem} | {mix} | {spur} | {share:.0%} |"
        )
    lines.append("")
    lines.append("### Top-5 features per arch (rank by |rejected − chosen|)")
    lines.append("")
    for arch_id, data in arch_data_list:
        lines.append(f"#### `{arch_id}`")
        lines.append("")
        lines.append("| rank | feat | diff | r_len | label |")
        lines.append("|---|---|---|---|---|")
        for r in data["features"][:5]:
            lines.append(
                f"| {r['rank']} | {r['feature_idx']} | {r['diff']:+.3f} | "
                f"{r['length_pearson_r']:+.3f} | {r['label']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--scatter-path", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_hh_rlhf_scatter.png")
    ap.add_argument("--summary-path", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_hh_rlhf_summary.png")
    args = ap.parse_args()
    banner(__file__)

    arch_data_list = []
    for arch in args.archs:
        d = _load_arch_data(arch)
        if d is None:
            print(f"  [skip] {arch}: top_features.json missing")
            continue
        arch_data_list.append((arch, d))
    if not arch_data_list:
        print("nothing to summarise")
        return

    print(f"writing scatter to {args.scatter_path}")
    make_scatter(arch_data_list, args.scatter_path)
    print(f"writing summary bar to {args.summary_path}")
    summary_rows = make_summary_bar(arch_data_list, args.summary_path)

    print()
    print(emit_markdown_writeup(summary_rows, arch_data_list))


if __name__ == "__main__":
    main()
