"""Plot per-cell mean ± std test ASR_16 across 3 seeds.

One subplot per hookpoint, three bars per subplot (SAE, T-SAE, TXC).
Reads outputs/seeded_logs/aggregated.json (produced by aggregate_seeded.py)
and writes outputs/seeded_logs/seed_average.png + .thumb.png.

Usage: uv run python plot_seed_average.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "outputs/seeded_logs/aggregated.json"
OUT_PNG = ROOT / "outputs/seeded_logs/seed_average.png"


HOOK_ORDER = [
    ("l0_ln1",  "blocks.0.ln1.hook_normalized"),
    ("l0_pre",  "blocks.0.hook_resid_pre"),
    ("l0_mid",  "blocks.0.hook_resid_mid"),
    ("l0_post", "blocks.0.hook_resid_post"),
    ("l1_ln1",  "blocks.1.ln1.hook_normalized"),
]
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
ARCH_COLORS = {"sae": "#9467bd", "tsae": "#2ca02c", "txc": "#1f77b4"}
BASELINE_ASR = 0.99


def main() -> None:
    data = json.loads(JSON_PATH.read_text())
    cells = {(r["arch"], r["hook"]): r for r in data["cells"]}

    n_hooks = len(HOOK_ORDER)
    fig, axes = plt.subplots(1, n_hooks, figsize=(3.0 * n_hooks, 4.4), sharey=True)

    for ax, (hookkey, hooklabel) in zip(axes, HOOK_ORDER):
        means, stds, colors, labels = [], [], [], []
        for arch in ARCH_ORDER:
            row = cells.get((arch, hookkey))
            if row is None:
                means.append(None); stds.append(None); colors.append("lightgrey"); labels.append(arch)
                continue
            means.append(row["test_asr_mean"])
            stds.append(row["test_asr_std"])
            colors.append(ARCH_COLORS[arch])
            labels.append(ARCH_LABELS[arch])

        x = list(range(len(ARCH_ORDER)))
        bars = ax.bar(
            x,
            [m if m is not None else 0 for m in means],
            yerr=[s if s is not None else 0 for s in stds],
            color=colors,
            capsize=5,
            edgecolor="black",
            linewidth=0.6,
        )

        # Per-seed dots overlaid on each bar.
        for i, arch in enumerate(ARCH_ORDER):
            row = cells.get((arch, hookkey))
            if row is None:
                continue
            for seed, val in row["test_asr_per_seed"].items():
                ax.scatter(
                    [i + (int(seed) - 1) * 0.12],
                    [val],
                    color="black",
                    s=22,
                    zorder=5,
                    alpha=0.7,
                )

        ax.axhline(BASELINE_ASR, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(hooklabel.replace("blocks.", "").replace(".hook_", "."), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("test ASR$_{16}$  (lower = more suppression)", fontsize=11)
    fig.suptitle(
        "Phase 8 — TinyStories Sleeper: 3-seed mean ± std test ASR per arch × hookpoint\n"
        "(black dots = individual seeds, dashed line = baseline ASR 0.99)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")

    # Thumbnail per CLAUDE.md (low-res, ≤288px wide).
    thumb_path = OUT_PNG.with_suffix(".thumb.png")
    fig.savefig(thumb_path, dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {thumb_path}")


if __name__ == "__main__":
    main()
