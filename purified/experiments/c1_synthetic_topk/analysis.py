"""C1 analysis — feature recovery AUC vs k_pos sweep.

Reads c1 leaderboard rows (filtering smoke + non-canonical), computes
mean ± std across seeds per (arch, T-label, k_pos), writes
AUTO-RESULTS to ``docs/components/c1.md``.

Headline: AUC vs k_pos line per arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c1"

CANONICAL_ARCH_TS: list[tuple[str, str]] = [
    ("topk_sae",    "default"),
    ("tsae_paper",  "default"),
    ("tfa",         "default"),
    ("tfa_pos",     "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "default"),  # T=5
    ("txc_base",    "default"),  # T=5
    ("txc_pro",     "default"),  # T_max=10
]

# Display labels + colors for paper-ready plot. Palette is paper-friendly
# (color-blind safe via Okabe-Ito; agent_filler 2026-05-06).
PLOT_STYLE: dict[tuple[str, str], dict[str, Any]] = {
    ("topk_sae",    "default"): {"label": "TopK-SAE",        "color": "#000000", "ls": "-",  "marker": "o"},
    ("tsae_paper",  "default"): {"label": "T-SAE",           "color": "#0072B2", "ls": "-",  "marker": "s"},
    ("tfa",         "default"): {"label": "TFA",             "color": "#009E73", "ls": "-",  "marker": "^"},
    ("tfa_pos",     "default"): {"label": "TFA-pos",         "color": "#56B4E9", "ls": "-",  "marker": "v"},
    ("stacked_sae", "T=2"):     {"label": "Stacked T=2",     "color": "#E69F00", "ls": "--", "marker": "D"},
    ("stacked_sae", "default"): {"label": "Stacked T=5",     "color": "#D55E00", "ls": "-",  "marker": "D"},
    ("txc_base",    "default"): {"label": "TXC-base T=5",    "color": "#CC79A7", "ls": "-",  "marker": "P"},
    ("txc_pro",     "default"): {"label": "TXC-pro T_max=10","color": "#882255", "ls": "-",  "marker": "X"},
}


@dataclass
class AnalysisResult:
    markdown: str
    results: dict[str, Any]


def _save_plot(agg: dict, ks: list[int], out_path: Path) -> None:
    """Generate paper-ready AUC vs k_pos line plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for arch, t_label in CANONICAL_ARCH_TS:
        ys = []
        es = []
        xs_valid = []
        for k in ks:
            stat = agg.get((arch, t_label, k))
            if stat is None:
                continue
            xs_valid.append(k)
            ys.append(stat["auc_mean"])
            es.append(stat["auc_std"])
        if not xs_valid:
            continue
        style = PLOT_STYLE.get((arch, t_label), {"label": f"{arch} {t_label}"})
        ax.errorbar(
            xs_valid, ys, yerr=es,
            label=style.get("label", f"{arch} {t_label}"),
            color=style.get("color", None),
            linestyle=style.get("ls", "-"),
            marker=style.get("marker", "o"),
            markersize=6, capsize=3, linewidth=1.6, alpha=0.9,
        )
    ax.set_xlabel(r"$k_{\rm pos}$ (active latents per token)")
    ax.set_ylabel("Feature recovery AUC")
    ax.set_title(r"C1: Markov-chain TopK sweep — AUC vs $k_{\rm pos}$ "
                 r"(toy $n=20$, $d=40$, $d_{\rm sae}=40$)")
    ax.set_ylim(0.35, 1.02)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    # thumbnail
    thumb = out_path.with_suffix(".thumb.png")
    fig.savefig(thumb, bbox_inches="tight", dpi=72)
    plt.close(fig)


def run_analysis() -> AnalysisResult:
    rows = [
        r for r in _read_jsonl(leaderboard_path())
        if r.get("component") == COMPONENT
        and not r.get("eval_cfg", {}).get("smoke", False)
    ]
    if not rows:
        return AnalysisResult(
            markdown="_No canonical c1 cells in leaderboard yet._",
            results={"n_rows": 0},
        )

    grouped: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        arch = r["arch"]
        cfg = r.get("eval_cfg", {})
        t_label = cfg.get("t_label", "default")
        k_pos = int(cfg.get("k_pos", -1))
        if k_pos < 0:
            continue
        key = (arch, t_label, k_pos)
        grouped.setdefault(key, []).append(
            float(r["metrics"].get("auc", float("nan")))
        )

    agg: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, vals in grouped.items():
        agg[key] = {
            "auc_mean": float(np.nanmean(vals)),
            "auc_std":  float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    ks = sorted({k for (_, _, k) in agg.keys()})

    out_lines: list[str] = []
    out_lines.append("**Headline: feature recovery AUC vs k_pos**\n")
    header = "| arch | T | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep    = "|---|---|" + "|".join(["---:" for _ in ks]) + "|"
    out_lines.extend([header, sep])
    for arch, t_label in CANONICAL_ARCH_TS:
        cells = []
        for k in ks:
            stat = agg.get((arch, t_label, k))
            if stat is None:
                cells.append("—")
                continue
            mean = stat["auc_mean"]
            std = stat["auc_std"]
            cells.append(f"{mean:.3f}±{std:.3f}")
        out_lines.append(
            f"| `{arch}` | {t_label} | " + " | ".join(cells) + " |"
        )

    out_lines.append("")
    out_lines.append(
        f"_Cells aggregated over seeds; n shown per cell. Filter: "
        f"`component='{COMPONENT}'`, `smoke=False`. Skipped cells "
        f"(k_train > arch budget at toy d_sae=40) appear as `—`._"
    )

    # Paper-ready plot.
    plot_path = Path(__file__).resolve().parent / "plots" / "c1_auc_vs_kpos.png"
    try:
        _save_plot(agg, ks, plot_path)
        # Embed full-res image in markdown; thumb is also generated for
        # any external viewer that prefers smaller previews.
        rel = "../../experiments/c1_synthetic_topk/plots/c1_auc_vs_kpos.png"
        out_lines.append("")
        out_lines.append(f"![Feature recovery AUC vs k_pos]({rel})")
    except Exception as e:
        out_lines.append(f"\n_(plot generation failed: {e})_")

    return AnalysisResult(
        markdown="\n".join(out_lines),
        results={
            "n_rows": len(rows),
            "n_cells": len(grouped),
            "ks": ks,
            "plot": str(plot_path) if plot_path.exists() else None,
        },
    )


def main():
    result = run_analysis()
    print(result.markdown)


if __name__ == "__main__":
    main()
