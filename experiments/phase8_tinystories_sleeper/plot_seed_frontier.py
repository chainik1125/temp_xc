"""Coherence / suppression frontier from the 3-seed aggregate.

For each (arch, hookpoint), plot the *chosen* (best-feature) point with
mean ± std across 3 seeds:
- x: test ΔCE  (lower = less damage / more coherence)
- y: test ASR_16 (lower = more suppression)

Lower-left = ideal. Color = arch. Marker = hookpoint. Per-seed dots
overlaid lightly so seed-spread is visible.

Reads outputs/seeded_logs/aggregated.json. Writes
outputs/seeded_logs/seed_frontier.png + .thumb.png.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "outputs/seeded_logs/aggregated.json"
OUT_PNG = ROOT / "outputs/seeded_logs/seed_frontier.png"


HOOK_ORDER = [
    ("l0_ln1",  "ln1.0"),
    ("l0_pre",  "resid_pre.0"),
    ("l0_mid",  "resid_mid.0"),
    ("l0_post", "resid_post.0"),
    ("l1_ln1",  "ln1.1"),
]
HOOK_MARKERS = {"l0_ln1": "o", "l0_pre": "s", "l0_mid": "D", "l0_post": "^", "l1_ln1": "v"}
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
ARCH_COLORS = {"sae": "#9467bd", "tsae": "#2ca02c", "txc": "#1f77b4"}


def main() -> None:
    data = json.loads(JSON_PATH.read_text())
    cells = {(r["arch"], r["hook"]): r for r in data["cells"]}

    fig, ax = plt.subplots(figsize=(8, 6))

    legend_arch_handles = []
    legend_hook_handles = []

    for arch in ARCH_ORDER:
        arch_color = ARCH_COLORS[arch]
        # Track per-arch points for the Pareto envelope.
        arch_xs, arch_ys = [], []
        for hookkey, hooklabel in HOOK_ORDER:
            row = cells.get((arch, hookkey))
            if row is None:
                continue
            seeds = sorted(row["test_asr_per_seed"].keys())
            asrs = [row["test_asr_per_seed"][s] for s in seeds]
            dces = [row["test_dce_per_seed"][s] for s in seeds]
            mean_asr = statistics.mean(asrs)
            mean_dce = statistics.mean(dces)
            std_asr = statistics.stdev(asrs) if len(asrs) > 1 else 0.0
            std_dce = statistics.stdev(dces) if len(dces) > 1 else 0.0

            marker = HOOK_MARKERS[hookkey]
            # Mean point with error bars in both axes.
            ax.errorbar(
                mean_dce, mean_asr,
                xerr=std_dce, yerr=std_asr,
                fmt=marker, color=arch_color, ecolor=arch_color,
                markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                capsize=3, alpha=0.95, zorder=5,
            )
            # Per-seed dots, smaller and translucent.
            ax.scatter(dces, asrs, color=arch_color, marker=marker,
                       s=30, alpha=0.4, edgecolor="none", zorder=3)
            arch_xs.append(mean_dce); arch_ys.append(mean_asr)

        legend_arch_handles.append(
            plt.Line2D([0], [0], marker="o", color=arch_color, linestyle="",
                       markeredgecolor="black", markeredgewidth=0.6, markersize=10,
                       label=ARCH_LABELS[arch])
        )

    for hookkey, hooklabel in HOOK_ORDER:
        legend_hook_handles.append(
            plt.Line2D([0], [0], marker=HOOK_MARKERS[hookkey], color="dimgrey",
                       linestyle="", markersize=10, label=hooklabel)
        )

    # ΔCE = 0.05 utility budget reference line.
    ax.axvline(0.05, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
    ax.text(0.05, 1.02, "δ=0.05", color="grey", fontsize=9, ha="center")

    # Baseline ASR reference.
    ax.axhline(0.99, linestyle=":", color="grey", alpha=0.5, linewidth=1.0)
    ax.text(0.0, 0.985, "baseline ASR = 0.99", color="grey", fontsize=9, va="top")

    ax.set_xlabel("test Δ clean-continuation CE (nats)  —  lower = less damage")
    ax.set_ylabel("test ASR$_{16}$  —  lower = more suppression")
    ax.set_title(
        "Phase 8 — TinyStories Sleeper: best-feature coherence/suppression frontier\n"
        "(mean ± std across 3 seeds; lower-left is ideal)"
    )
    ax.set_ylim(-0.05, 1.05)

    # Two-column legend: archs (color) + hookpoints (marker).
    leg1 = ax.legend(handles=legend_arch_handles, loc="upper right", title="arch", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=legend_hook_handles, loc="lower right", title="hookpoint", fontsize=9)

    ax.grid(alpha=0.25)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")
    thumb_path = OUT_PNG.with_suffix(".thumb.png")
    fig.savefig(thumb_path, dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {thumb_path}")


if __name__ == "__main__":
    main()
