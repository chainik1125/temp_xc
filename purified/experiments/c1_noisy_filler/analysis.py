"""c1_noisy analysis — wasteland 1c-noisy reproduction (under C2).

Reads c1_noisy leaderboard rows and aggregates by (arch, T-label, k_pos).
Han 2026-05-06 mapping: c1_noisy is paper-component-wise UNDER C2, not
C1 — it tests temporal denoising on coupled-emission setup with
Bernoulli noise (p_A=0, p_B=0.625).

Headline: AUC vs k_pos line per arch under noisy emissions.
Wasteland claim: TXCDRv2 T=2 → AUC ≥ 0.98 across k=3..12; T=5 → AUC
≈ 0.99 + corr ratio ≈ 1.0 at k=3..8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c1_noisy"

# Arch+T combos. Original c1_noisy ARCH_TS (run.py): 6 entries.
# T-sweep extension (run_t.py, 2026-05-06): adds T={4,6,8,10,12} for
# txc_base to match wasteland scatter plot (T=2..12).
CANONICAL_ARCH_TS: list[tuple[str, str]] = [
    ("tfa_pos",     "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "default"),  # T=5
    ("txc_base",    "T=2"),
    ("txc_base",    "T=4"),
    ("txc_base",    "default"),  # T=5
    ("txc_base",    "T=6"),
    ("txc_base",    "T=8"),
    ("txc_base",    "T=10"),
    ("txc_base",    "T=12"),
    ("txc_pro",     "default"),  # T_max=10
]

# Okabe-Ito palette + RdPu cmap for txc_base T-sweep, color-blind safe.
import matplotlib.cm as _cm
_txc_cmap = _cm.get_cmap("RdPu", 8)
PLOT_STYLE: dict[tuple[str, str], dict[str, Any]] = {
    ("tfa_pos",     "default"): {"label": "TFA-pos",       "color": "#56B4E9", "ls": "-",  "marker": "v"},
    ("stacked_sae", "T=2"):     {"label": "Stacked T=2",   "color": "#E69F00", "ls": "--", "marker": "D"},
    ("stacked_sae", "default"): {"label": "Stacked T=5",   "color": "#D55E00", "ls": "-",  "marker": "D"},
    ("txc_base",    "T=2"):     {"label": "TXC-base T=2",  "color": _txc_cmap(1), "ls": "-",  "marker": "P"},
    ("txc_base",    "T=4"):     {"label": "TXC-base T=4",  "color": _txc_cmap(2), "ls": "-",  "marker": "s"},
    ("txc_base",    "default"): {"label": "TXC-base T=5",  "color": _txc_cmap(3), "ls": "-",  "marker": "P"},
    ("txc_base",    "T=6"):     {"label": "TXC-base T=6",  "color": _txc_cmap(4), "ls": "-",  "marker": "^"},
    ("txc_base",    "T=8"):     {"label": "TXC-base T=8",  "color": _txc_cmap(5), "ls": "-",  "marker": "v"},
    ("txc_base",    "T=10"):    {"label": "TXC-base T=10", "color": _txc_cmap(6), "ls": "-",  "marker": "<"},
    ("txc_base",    "T=12"):    {"label": "TXC-base T=12", "color": _txc_cmap(7), "ls": "-",  "marker": "p"},
    ("txc_pro",     "default"): {"label": "TXC-pro T_max=10", "color": "#1f77b4", "ls": "-",  "marker": "X"},
}


@dataclass
class AnalysisResult:
    markdown: str
    results: dict[str, Any]


def _save_plot(agg: dict, ks: list[int], out_path: Path) -> None:
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
    ax.set_xlabel(r"Per-token sparsity $k_{\rm pos}$", fontsize=11)
    ax.set_ylabel(r"Decoder AUC (vs feature directions)", fontsize=11)
    ax.set_title(r"Setup B: Decoder AUC under noisy emissions ($\gamma{=}0.25$)",
                 fontsize=12)
    ax.set_ylim(0.35, 1.02)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
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
            markdown="_No canonical c1_noisy cells in leaderboard yet._",
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
    out_lines.append("**Decoder AUC vs k_pos** (mean ± std over seeds)\n")
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
        f"_Cells aggregated over seeds. Filter: "
        f"`component='{COMPONENT}'`, `smoke=False`. Skipped cells "
        f"(k_train > arch budget at toy d_sae=40) appear as `—`._"
    )

    plot_path = Path(__file__).resolve().parent / "plots" / "c2_noisy_auc_vs_kpos.png"
    try:
        _save_plot(agg, ks, plot_path)
        rel = "../../experiments/c1_noisy_filler/plots/c2_noisy_auc_vs_kpos.png"
        out_lines.append("")
        out_lines.append(f"![Setup B: decoder AUC vs k_pos]({rel})")
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
