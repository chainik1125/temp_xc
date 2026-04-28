"""Phase 7 headline plots from probing_results.jsonl.

Produces:
1. headline_bar_k5.{png,thumb.png}  — bar chart of all 38 archs at k_feat=5,
   sorted by mean AUC, coloured by family.
2. headline_bar_k20.{png,thumb.png} — same at k_feat=20.
3. t_sweep_k5.{png,thumb.png}       — line plot: 3 T-sweep families
   (TXCDR T=3..32, H8 multidistance T=3..9 trimmed, TXC bare T=5/10/20).
4. t_sweep_k20.{png,thumb.png}      — same at k_feat=20.

Filters:
- seed=42, S=32, skipped=False
- 11 DROPPED_FROM_HEADLINE archs excluded (per
  2026-04-27-canonical-trim-emergency.md).
- FLIP convention for winogrande_correct_completion + wsc_coreference.
- Dedupe by (run_id, task_name, k_feat) keeping last occurrence.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.plots.make_headline_plot
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.save_figure import save_figure
from experiments.phase7_unification._paths import OUT_DIR


JSONL = OUT_DIR / "probing_results.jsonl"
PLOTS_DIR = OUT_DIR / "plots"

DROPPED = {
    f"phase57_partB_h8_bare_multidistance_t{t}"
    for t in (10, 12, 14, 16, 18, 20, 24, 28, 32)
} | {"phase5b_subseq_h8_T32_s5", "phase5b_subseq_h8_T64_s5"}

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

FAMILY_COLOURS = {
    "txc":          "#1f77b4",   # blue   — TXCDR + matryoshka + agentic_txc
    "txc_bare":     "#17becf",   # cyan   — TXC bare antidead (Track 2)
    "h8":           "#bcbd22",   # olive  — H8 multidistance contrastive
    "subseq":       "#7f7f7f",   # grey   — Subseq B2/B4 sampling
    "mlc":          "#2ca02c",   # green  — MLC family
    "topk_sae":     "#ff7f0e",   # orange — single-token TopK SAE
    "tsae_paper":   "#e377c2",   # pink   — T-SAE paper BatchTopK
    "tfa":          "#d62728",   # red    — TFA (TemporalSAE)
}


def family_of(arch: str) -> str:
    if arch == "topk_sae":
        return "topk_sae"
    if arch.startswith("tsae_paper"):
        return "tsae_paper"
    if arch == "tfa_big":
        return "tfa"
    if arch == "mlc" or arch.startswith("mlc_") or arch.startswith("agentic_mlc_"):
        return "mlc"
    if arch.startswith("phase57_partB_h8") or arch.endswith("_kpos100"):
        # H8 anchor cell goes here too
        return "h8"
    if arch.startswith("phase5b_subseq"):
        return "subseq"
    if arch.startswith("txc_bare"):
        return "txc_bare"
    # Everything else TXC-shaped (txcdr, agentic_txc, matryoshka)
    return "txc"


def short_label(arch: str) -> str:
    """Compact xtick label."""
    return (arch
            .replace("phase57_partB_h8_bare_multidistance", "h8_md")
            .replace("phase5b_subseq", "subseq")
            .replace("txc_bare_antidead", "txc_bare")
            .replace("mlc_contrastive_alpha100_batchtopk", "mlc_contrast_btk")
            .replace("agentic_", "ag_"))


# ────────────────────── DATA LOADING ──────────────────────


def load_means() -> dict[tuple[str, int], tuple[float, float, int]]:
    """Returns {(arch_id, k_feat): (mean_auc, std_auc, n_tasks)} for seed=42, S=32."""
    rows = [json.loads(l) for l in JSONL.open()]
    keep = {}
    for r in rows:
        if r.get("skipped"): continue
        if r.get("seed") != 42: continue
        if r.get("S") != 32: continue
        if r.get("k_feat") not in (5, 20): continue
        if r["arch_id"] in DROPPED: continue
        keep[(r["run_id"], r["task_name"], r["k_feat"])] = r
    by_arch_k = defaultdict(list)
    for r in keep.values():
        auc = r["test_auc_flip"] if r["task_name"] in FLIP_TASKS else r["test_auc"]
        by_arch_k[(r["arch_id"], r["k_feat"])].append(auc)
    means = {}
    for (arch, k), aucs in by_arch_k.items():
        means[(arch, k)] = (float(np.mean(aucs)), float(np.std(aucs)), len(aucs))
    return means


# ────────────────────── HEADLINE BAR ──────────────────────


def headline_bar(means: dict, k: int, out_path: Path) -> None:
    archs = sorted([a for (a, kk) in means.keys() if kk == k],
                   key=lambda a: -means[(a, k)][0])
    fig, ax = plt.subplots(figsize=(max(16, 0.42 * len(archs)), 8))
    x = np.arange(len(archs))
    vals = [means[(a, k)][0] for a in archs]
    stds = [means[(a, k)][1] for a in archs]
    families = [family_of(a) for a in archs]
    colours = [FAMILY_COLOURS[f] for f in families]
    bars = ax.bar(x, vals, yerr=stds, capsize=3, color=colours,
                  edgecolor="#222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(a) for a in archs],
                       rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"mean test AUC (S=32, k_feat={k})")
    ax.set_ylim(0.50, 1.0)
    ax.axhline(0.5, color="black", lw=0.4, ls=":")
    ax.set_title(
        f"Phase 7: sparse-probing AUC by arch (seed=42, S=32, k_feat={k}) — "
        f"sorted by mean, coloured by family\n"
        f"{len(archs)} archs · 36 SAEBench tasks · "
        f"FLIP for winogrande / wsc · error bars = std across tasks"
    )

    # Family legend
    used = []
    for f in families:
        if f not in used: used.append(f)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAMILY_COLOURS[f], label=f) for f in used]
    ax.legend(handles=handles, title="family", loc="upper right",
              fontsize=8, title_fontsize=8, framealpha=0.9)

    # Value labels above each bar
    for bar, m in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", fontsize=6, rotation=90,
                va="bottom")

    fig.tight_layout()
    save_figure(fig, str(out_path), dpi=200)
    plt.close(fig)


# ────────────────────── T-SWEEP ──────────────────────


def t_sweep_plot(means: dict, k: int, out_path: Path) -> None:
    """Three families plotted as line vs T:
       - TXCDR T-sweep T ∈ {3..10, 12, 14, 16, 18, 20, 24, 28, 32}
       - H8 multidistance T ∈ {3..9} (trimmed; original spec went to T=32)
       - TXC bare antidead T ∈ {5, 10, 20}
    """
    txcdr_T = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    h8_T    = [3, 4, 5, 6, 7, 8, 9]
    bare_T  = [5, 10, 20]

    def get_curve(prefix: str, T_list: list[int]):
        ys, errs, ts = [], [], []
        for t in T_list:
            v = means.get((f"{prefix}{t}", k))
            if v is None: continue
            ts.append(t); ys.append(v[0]); errs.append(v[1])
        return ts, ys, errs

    fig, ax = plt.subplots(figsize=(11, 6))
    for prefix, T_list, label, colour in [
        ("txcdr_t", txcdr_T,
         "TXCDR T-sweep (k_win=k_pos × T = 100 × T)",
         FAMILY_COLOURS["txc"]),
        ("phase57_partB_h8_bare_multidistance_t", h8_T,
         "H8 multidistance contrastive (trimmed at T=9)",
         FAMILY_COLOURS["h8"]),
        ("txc_bare_antidead_t", bare_T,
         "TXC bare antidead",
         FAMILY_COLOURS["txc_bare"]),
    ]:
        ts, ys, errs = get_curve(prefix, T_list)
        ax.errorbar(ts, ys, yerr=errs, marker="o", capsize=3,
                    color=colour, label=label, linewidth=1.5)

    ax.set_xlabel("T (window length)")
    ax.set_ylabel(f"mean test AUC (S=32, k_feat={k})")
    ax.set_title(
        f"Phase 7: T-sweep AUC at k_feat={k} (seed=42)\n"
        f"36 SAEBench tasks · S=32 · error bars = std across tasks"
    )
    ax.set_xticks(sorted(set(txcdr_T + h8_T + bare_T)))
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, str(out_path), dpi=200)
    plt.close(fig)


# ────────────────────── MAIN ──────────────────────


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    means = load_means()
    print(f"Loaded {len(means)} (arch, k_feat) cells from {JSONL}")

    for k in (5, 20):
        bar_path = PLOTS_DIR / f"headline_bar_k{k}.png"
        ts_path  = PLOTS_DIR / f"t_sweep_k{k}.png"
        headline_bar(means, k, bar_path)
        print(f"  wrote {bar_path.name} (+ thumb)")
        t_sweep_plot(means, k, ts_path)
        print(f"  wrote {ts_path.name} (+ thumb)")
    print(f"All plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
