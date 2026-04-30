"""3 (arch) × 5 (hookpoint) grid of per-cell val sweeps.

Each subplot shows all stage-2 (feature, α) candidates from the three
seeds at that (arch, hookpoint) cell. x-axis = val ΔCE, y-axis = val
sampled ASR_16. Color = seed. Chosen point per seed marked with a
star (the point selected as `argmin val_asr s.t. ΔCE ≤ 0.05`).

Reads outputs/seeded_logs/val_sweeps/val_sweep_<basetag>_s<seed>.json
for all 45 (basetag × seed) combinations. Writes
outputs/seeded_logs/seed_grid.png + .thumb.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SWEEPS_DIR = ROOT / "outputs/seeded_logs/val_sweeps"
OUT_PNG = ROOT / "outputs/seeded_logs/seed_grid.png"


HOOK_ORDER = [
    ("l0_ln1",  "ln1.0"),
    ("l0_pre",  "resid_pre.0"),
    ("l0_mid",  "resid_mid.0"),
    ("l0_post", "resid_post.0"),
    ("l1_ln1",  "ln1.1"),
]
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
SEED_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
DELTA_BUDGET = 0.05


def load_sweep(basetag: str, seed: int) -> dict | None:
    path = SWEEPS_DIR / f"val_sweep_{basetag}_s{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    n_arch = len(ARCH_ORDER)
    n_hook = len(HOOK_ORDER)
    fig, axes = plt.subplots(
        n_arch, n_hook,
        figsize=(3.0 * n_hook, 2.6 * n_arch),
        sharex=True, sharey=True,
    )

    for r, arch in enumerate(ARCH_ORDER):
        for c, (hookkey, hooklabel) in enumerate(HOOK_ORDER):
            ax = axes[r, c]
            basetag = f"{arch}_{hookkey}"
            for seed in (0, 1, 2):
                d = load_sweep(basetag, seed)
                if d is None:
                    continue
                color = SEED_COLORS[seed]
                # All stage-2 candidates: a cloud of (ΔCE, val_asr).
                xs = [row["delta_clean_ce"] for row in d.get("stage2", [])]
                ys = [row["val_asr_16"] for row in d.get("stage2", [])]
                ax.scatter(xs, ys, color=color, s=14, alpha=0.45,
                           edgecolor="none", label=f"s{seed}")
                # Chosen point per seed.
                ch = d["chosen"]
                ax.scatter(
                    ch["delta_clean_ce"], ch["val_asr_16"],
                    marker="*", s=180, color=color,
                    edgecolor="black", linewidths=0.7, zorder=10,
                )

            ax.axvline(DELTA_BUDGET, linestyle="--", color="grey",
                       alpha=0.5, linewidth=0.8)

            if r == 0:
                ax.set_title(hooklabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{ARCH_LABELS[arch]}\nval ASR$_{{16}}$",
                              fontsize=10)
            if r == n_arch - 1:
                ax.set_xlabel("val ΔCE (nats)", fontsize=9)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    # Single legend for seed colors at the figure top-right.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SEED_COLORS[s],
                   markersize=8, label=f"seed {s}")
        for s in (0, 1, 2)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", linestyle="", color="grey",
                   markeredgecolor="black", markersize=14, label="chosen (f, α)")
    )
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=4,
               bbox_to_anchor=(0.98, 1.02))

    fig.suptitle(
        "Phase 8 — TinyStories Sleeper: per-cell val sweep clouds\n"
        "(rows = arch, cols = hookpoint; each dot = one (f, α) candidate; "
        "star = chosen; dashed = ΔCE budget δ=0.05)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")
    fig.savefig(OUT_PNG.with_suffix(".thumb.png"), dpi=24,
                bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
