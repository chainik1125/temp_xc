"""Per-cell α-sweep curve for the *chosen* (dominant) feature only.

3 (arch) × 5 (hookpoint) grid. Each subplot shows the α-sweep curve
for whichever feature was selected as `argmin val_asr s.t. ΔCE ≤ 0.05`
at that (arch, hookpoint, seed) cell — one curve per seed (so up to
3 distinct features per cell, each evaluated at α ∈
{0.25, 0.5, 1.0, 1.5, 2.0}).

Axes are equalised with `plot_seed_grid.py` so the two figures sit at
the same scale and can be read side-by-side.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SWEEPS_DIR = ROOT / "outputs/seeded_logs/val_sweeps"
OUT_PNG = ROOT / "outputs/seeded_logs/seed_grid_dominant.png"


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
# MUST match plot_seed_grid.py.
X_LIM = (-0.04, 0.16)
Y_LIM = (-0.05, 1.05)


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
                chosen_f = d["chosen"]["feature_idx"]
                chosen_alpha = d["chosen"]["alpha"]

                # All stage-2 rows for the chosen feature, sorted by α.
                rows = [
                    r for r in d.get("stage2", [])
                    if r["feature_idx"] == chosen_f
                ]
                rows.sort(key=lambda r: r["alpha"])
                if not rows:
                    continue
                xs = [r["delta_clean_ce"] for r in rows]
                ys = [r["val_asr_16"] for r in rows]
                color = SEED_COLORS[seed]
                ax.plot(xs, ys, marker="o", color=color, linewidth=1.6,
                        markersize=6, alpha=0.85,
                        label=f"s{seed} f={chosen_f}")
                # Star at the chosen α.
                ch_x = d["chosen"]["delta_clean_ce"]
                ch_y = d["chosen"]["val_asr_16"]
                ax.scatter(ch_x, ch_y, marker="*", s=200, color=color,
                           edgecolor="black", linewidths=0.7, zorder=10)

            ax.axvline(DELTA_BUDGET, linestyle="--", color="grey",
                       alpha=0.5, linewidth=0.8)

            if r == 0:
                ax.set_title(hooklabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{ARCH_LABELS[arch]}\nval ASR$_{{16}}$",
                              fontsize=10)
            if r == n_arch - 1:
                ax.set_xlabel("val ΔCE (nats)", fontsize=9)

            ax.set_xlim(*X_LIM)
            ax.set_ylim(*Y_LIM)
            ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=SEED_COLORS[s],
                   markersize=8, label=f"seed {s}")
        for s in (0, 1, 2)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", linestyle="", color="grey",
                   markeredgecolor="black", markersize=14, label="chosen α")
    )
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=4,
               bbox_to_anchor=(0.98, 1.02))

    fig.suptitle(
        "Phase 8 — TinyStories Sleeper: chosen-feature α-sweeps\n"
        "(rows = arch, cols = hookpoint; one line per seed = α∈{0.25,0.5,1,1.5,2} "
        "for that seed's chosen feature; dashed = ΔCE budget δ=0.05)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")
    fig.savefig(OUT_PNG.with_suffix(".thumb.png"), dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
