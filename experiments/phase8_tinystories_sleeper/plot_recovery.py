"""Plot the recovery metric for the cells we ran.

Two figures, mirroring the earlier ASR plots:

1. recovery_per_cell.png — 1 row × {hookpoints with data} cols. Each
   subplot shows per-arch mean ± std bars with individual sample dots
   overlaid. Y-axis: recovery. Dashed lines at 0 (no improvement) and
   1 (perfect recovery).

2. recovery_vs_asr_scatter.png — combined 2D Pareto. Each cell is one
   point: x = test ASR (mean across seeds, lower=better suppression),
   y = recovery (mean across seeds, higher=better coherence). Lower-
   right is ideal (low ASR, high recovery). Color = arch, marker =
   hookpoint, error bars on both axes.

Reads outputs/seeded_logs/recovery.json (recovery values) and
outputs/seeded_logs/aggregated.json (test ASR values).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RECOVERY_JSON = ROOT / "outputs/seeded_logs/recovery.json"
ASR_JSON = ROOT / "outputs/seeded_logs/aggregated.json"
OUT_BARS = ROOT / "outputs/seeded_logs/recovery_per_cell.png"
OUT_SCATTER = ROOT / "outputs/seeded_logs/recovery_vs_asr_scatter.png"


HOOK_LABEL = {
    "l0_ln1":  "ln1.0",
    "l0_pre":  "resid_pre.0",
    "l0_mid":  "resid_mid.0",
    "l0_post": "resid_post.0",
    "l1_ln1":  "ln1.1",
}
HOOK_ORDER = ["l0_ln1", "l0_pre", "l0_mid", "l0_post", "l1_ln1"]
HOOK_MARKERS = {"l0_ln1": "o", "l0_pre": "s", "l0_mid": "D",
                "l0_post": "^", "l1_ln1": "v"}
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
ARCH_COLORS = {"sae": "#9467bd", "tsae": "#2ca02c", "txc": "#1f77b4"}


def parse_tag(tag: str) -> tuple[str, str, int]:
    import re
    m = re.match(r"^(?P<arch>sae|tsae|txc)_(?P<hook>.+)_s(?P<seed>\d+)$", tag)
    return m.group("arch"), m.group("hook"), int(m.group("seed"))


def main() -> None:
    rec_data = json.loads(RECOVERY_JSON.read_text())
    cells_rec = rec_data["cells"]

    # Group recoveries by (arch, hook) → list across seeds.
    cell_groups: dict[tuple[str, str], dict[int, dict]] = {}
    for tag, d in cells_rec.items():
        arch, hook, seed = parse_tag(tag)
        cell_groups.setdefault((arch, hook), {})[seed] = d

    # Hookpoints that actually have recovery data.
    hooks_present = sorted(
        {h for (a, h) in cell_groups},
        key=HOOK_ORDER.index,
    )

    # ---------- Figure 1: per-cell bar chart ----------
    fig, axes = plt.subplots(
        1, len(hooks_present),
        figsize=(3.4 * len(hooks_present), 4.2),
        sharey=True,
    )
    if len(hooks_present) == 1:
        axes = [axes]

    for ax, hookkey in zip(axes, hooks_present):
        x = list(range(len(ARCH_ORDER)))
        means, stds, colors, labels = [], [], [], []
        for arch in ARCH_ORDER:
            seed_map = cell_groups.get((arch, hookkey), {})
            if not seed_map:
                means.append(None); stds.append(None)
                colors.append("lightgrey"); labels.append(ARCH_LABELS[arch])
                continue
            seed_means = [d["recovery_mean"] for d in seed_map.values()]
            mean_of_means = statistics.mean(seed_means)
            std = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
            means.append(mean_of_means); stds.append(std)
            colors.append(ARCH_COLORS[arch])
            labels.append(f"{ARCH_LABELS[arch]} (n={len(seed_map)})")

        ax.bar(x,
               [m if m is not None else 0 for m in means],
               yerr=[s if s is not None else 0 for s in stds],
               color=colors, capsize=5, edgecolor="black", linewidth=0.6)

        # per-seed dots overlaid
        for i, arch in enumerate(ARCH_ORDER):
            for seed, d in cell_groups.get((arch, hookkey), {}).items():
                ax.scatter([i + (seed - 1) * 0.12], [d["recovery_mean"]],
                           color="black", s=24, alpha=0.7, zorder=5)

        ax.axhline(0, linestyle="--", color="grey", alpha=0.5)
        ax.axhline(1, linestyle=":", color="grey", alpha=0.4)
        ax.text(len(ARCH_ORDER) - 0.5, 1.02, "perfect recovery",
                fontsize=8, ha="right", color="grey")
        ax.text(len(ARCH_ORDER) - 0.5, 0.02, "no improvement",
                fontsize=8, ha="right", color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(HOOK_LABEL[hookkey], fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel(
        r"recovery $= 1 - \mathrm{CE}_{\mathrm{steered}} / \mathrm{CE}_{\mathrm{poisoned}}$"
        + "\n(higher = output more like clean continuation)",
        fontsize=10,
    )
    fig.suptitle(
        "Phase 8 — recovery metric (sampled, matched RNG, n=64 samples per seed)\n"
        "(black dots = per-seed mean; bars = mean ± std across seeds; "
        "1 = perfect recovery, 0 = no improvement, < 0 = coherence collapse)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_BARS, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_BARS.with_suffix(".thumb.png"), dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {OUT_BARS}")

    # ---------- Figure 2: ASR × recovery scatter ----------
    asr_data = json.loads(ASR_JSON.read_text())
    asr_cells = {(r["arch"], r["hook"]): r for r in asr_data["cells"]}

    fig2, ax2 = plt.subplots(figsize=(8.5, 6.5))

    # Plot reference points where we have BOTH ASR and recovery.
    legend_arch = []
    legend_hook = []
    for arch in ARCH_ORDER:
        for hookkey in HOOK_ORDER:
            seed_map = cell_groups.get((arch, hookkey), {})
            asr_row = asr_cells.get((arch, hookkey))
            if not seed_map or asr_row is None:
                continue
            rec_seeds = [d["recovery_mean"] for d in seed_map.values()]
            rec_mean = statistics.mean(rec_seeds)
            rec_std = statistics.stdev(rec_seeds) if len(rec_seeds) > 1 else 0.0
            asr_mean = asr_row["test_asr_mean"]
            asr_std = asr_row["test_asr_std"]
            color = ARCH_COLORS[arch]
            marker = HOOK_MARKERS[hookkey]
            ax2.errorbar(
                asr_mean, rec_mean, xerr=asr_std, yerr=rec_std,
                fmt=marker, color=color, ecolor=color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.7, capsize=3,
                zorder=5, alpha=0.95,
            )

    for arch in ARCH_ORDER:
        legend_arch.append(plt.Line2D(
            [0], [0], marker="o", color=ARCH_COLORS[arch], linestyle="",
            markeredgecolor="black", markeredgewidth=0.6, markersize=10,
            label=ARCH_LABELS[arch]))
    for hookkey in HOOK_ORDER:
        if any((a, hookkey) in cell_groups for a in ARCH_ORDER):
            legend_hook.append(plt.Line2D(
                [0], [0], marker=HOOK_MARKERS[hookkey], color="dimgrey",
                linestyle="", markersize=10, label=HOOK_LABEL[hookkey]))

    ax2.axhline(0, linestyle="--", color="grey", alpha=0.5)
    ax2.axhline(1, linestyle=":", color="grey", alpha=0.4)
    ax2.axvline(0.99, linestyle=":", color="grey", alpha=0.4)
    ax2.text(0.985, 0.98, "baseline ASR", color="grey", fontsize=8,
             rotation=90, va="top")
    # Shade the ideal region (low ASR, high recovery).
    ax2.axhspan(0.0, 1.05, xmin=0, xmax=0.2 / 1.05, alpha=0.06, color="green",
                label="_nolegend_")

    ax2.set_xlabel(
        "test ASR$_{16}$ (greedy, lower = more suppression)", fontsize=11,
    )
    ax2.set_ylabel(
        r"recovery $= 1 - \mathrm{CE}_{\mathrm{steered}} / \mathrm{CE}_{\mathrm{pois}}$"
        + "\n(sampled, matched RNG; higher = better coherence)",
        fontsize=11,
    )
    ax2.set_title(
        "Phase 8 — suppression × coherence (greedy ASR vs sampled recovery)\n"
        "ideal = lower-right (low ASR, high recovery, no coherence collapse)",
        fontsize=11,
    )
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-5.5, 1.2)

    leg1 = ax2.legend(handles=legend_arch, loc="lower right", title="arch",
                      fontsize=9)
    ax2.add_artist(leg1)
    ax2.legend(handles=legend_hook, loc="upper right", title="hookpoint",
               fontsize=9)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
    fig2.savefig(OUT_SCATTER.with_suffix(".thumb.png"), dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {OUT_SCATTER}")


if __name__ == "__main__":
    main()
