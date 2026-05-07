"""Cross-setup TXC-vs-SAE gap summary plot.

Reads leaderboard.jsonl, computes the TXC-base T=5 vs TopK-SAE gAUC and
eAUC gaps at k_pos=1 across all canonical c2 synthetic setups (A, D-np5,
D-np10, E, F-σ1, G-σ1, J, M), and renders a horizontal bar chart that
makes the cross-setup pattern visible at a glance.

Paper-grade headline: "TXC dictionaries recover global features that
per-token SAEs miss — across 8 distinct synthetic regimes."

Run via:
    .venv/bin/python -m experiments.c2_synthetic_coupled.cross_setup_summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

LEADERBOARD = Path("results/leaderboard.jsonl")
OUT_PATH = Path("experiments/c2_synthetic_coupled/plots/c2_cross_setup_summary.png")


# (setup_label, datasource, filter_kwargs) — canonical regime per setup.
SETUPS = [
    ("A (coupled, ρ=0.7)",
     "toy_coupled_K10_M20_d256",
     {}),
    ("D-np5 (Dmitry replicate)",
     "toy_coupled_noisy_K10_M20_d256_pB05_np5",
     {}),
    ("D-np10 (max overlap)",
     "toy_coupled_noisy_K10_M20_d256_pB05_np10",
     {}),
    ("E (hierarchical Kl=30)",
     "toy_hierarchical_Kg10_Kl30_d256",
     {}),
    ("F (coupled+noise σ=1.0)",
     "toy_coupled_obs_noise_K10_M20_d256_sigma1p0",
     {"obs_noise_sigma": 1.0}),
    ("G (hier+noise σ=1.0)",
     "toy_hierarchical_Kg10_Kl30_d256_sigma1p0",
     {"obs_noise_sigma": 1.0}),
    ("J (hier Kl=50)",
     "toy_hierarchical_Kg10_Kl50_d256",
     {}),
    ("M (slow+fast globals)",
     "toy_hetero_rho_Kg10_Kl30_d256_5slow_5fast",
     {}),
]


def _select_arch_at_k1(rows, *, arch_name: str, t_label: str | None = None):
    """Pull the cell at k_pos=1 for one arch (averaged over seeds)."""
    seeds_g, seeds_e = [], []
    for d in rows:
        ec = d.get("eval_cfg") or {}
        if ec.get("k_pos") != 1:
            continue
        if d["arch"] != arch_name:
            continue
        if t_label is not None and ec.get("t_label", "default") != t_label:
            continue
        seeds_g.append(float(d["metrics"]["gauc"]))
        seeds_e.append(float(d["metrics"]["eauc"]))
    if not seeds_g:
        return None, None
    return mean(seeds_g), mean(seeds_e)


def main():
    # Pass 1: dedupe by eval_key
    latest: dict[str, dict] = {}
    with LEADERBOARD.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("component") != "c2":
                continue
            ec = d.get("eval_cfg") or {}
            if ec.get("smoke"):
                continue
            latest[d["eval_key"]] = d

    # Bucket by datasource
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for d in latest.values():
        by_ds[d["datasource"]].append(d)

    # Compute (gauc_txc, gauc_sae, eauc_txc, eauc_sae) per setup
    rows = []
    for setup_label, ds, _filter in SETUPS:
        cells = by_ds.get(ds, [])
        # TXC-base T=5 (canonical)
        gtxc, etxc = _select_arch_at_k1(cells, arch_name="txc_base", t_label="T=5")
        if gtxc is None:
            # Fall back to default t_label (legacy cells)
            gtxc, etxc = _select_arch_at_k1(cells, arch_name="txc_base", t_label="default")
        # TopK-SAE
        gsae, esae = _select_arch_at_k1(cells, arch_name="topk_sae")
        if gtxc is None or gsae is None:
            print(f"[skip] {setup_label}: missing (gtxc={gtxc}, gsae={gsae})")
            continue
        rows.append({
            "label": setup_label,
            "gauc_txc": gtxc, "gauc_sae": gsae,
            "eauc_txc": etxc, "eauc_sae": esae,
            "gauc_gap": gtxc - gsae,
            "eauc_gap": etxc - esae,
        })
        print(f"{setup_label:30}  TXC g={gtxc:.3f} SAE g={gsae:.3f}  "
              f"gap={gtxc-gsae:+.3f}")

    if not rows:
        print("[plot] no setups with data; skipping")
        return

    # Render: 2-panel horizontal bar chart
    # Top panel: gAUC gap (TXC - SAE), bigger = TXC wins more
    # Bottom panel: paired bars showing absolute TXC vs SAE gAUC
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.7 * len(rows) + 2)))
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))

    # Left: paired absolute bars
    axes[0].barh(y - 0.2, [r["gauc_txc"] for r in rows], height=0.4,
                 color="#1f77b4", label="TXC-base T=5", edgecolor="black", linewidth=0.5)
    axes[0].barh(y + 0.2, [r["gauc_sae"] for r in rows], height=0.4,
                 color="#888888", label="TopK-SAE",     edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("gAUC at k_pos=1 (mean over seeds)", fontsize=11)
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(0.5, color="black", lw=0.5, alpha=0.3, ls=":")
    axes[0].set_title("Absolute global recovery (gAUC)", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.92)
    axes[0].grid(axis="x", alpha=0.3)

    # Right: gap bars (signed)
    gaps = [r["gauc_gap"] for r in rows]
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gaps]
    axes[1].barh(y, gaps, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("TXC − SAE  gAUC gap  (positive = TXC wins)", fontsize=11)
    axes[1].set_title("Per-setup gAUC gap", fontsize=12)
    axes[1].grid(axis="x", alpha=0.3)
    # Annotate bars with the numeric gap
    for i, g in enumerate(gaps):
        x_text = g + (0.02 if g >= 0 else -0.02)
        ha = "left" if g >= 0 else "right"
        axes[1].text(x_text, i, f"{g:+.2f}", va="center", ha=ha,
                     fontsize=9)

    fig.suptitle(
        "Cross-setup TXC vs TopK-SAE comparison (gAUC at k_pos=1)\n"
        "Same training protocol; the per-token SAE has the SAME dictionary capacity.",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
