"""Headline bar chart: best global recovery per arch on Setups B and D.

Two bars per x-tick (one per task), in the style of the EM and backtracking
headline bars. Setup B uses the single-latent global correlation
``sl_mean_global`` (max over T variants, k_pos, seeds) from
``denoising_probe_results.json`` — same metric the y-axis of the
single-latent scatter (\cref{fig:setup_b_denoising}) shows. Setup D uses
gAUC at maximum overlap (np=10), max over T variants and k_pos, from c2.md.

Refresh inputs after T-SAE-on-D runs land; we keep the function signatures
parametric so the merge is one dict update.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROBE_JSON = REPO / "notes" / "c2_synthetic" / "data" / "denoising_probe_results.json"
OUT = REPO / "figs" / "c2" / "c2_synth_global_headline.png"

# Architectures we report on the headline bar chart, in display order.
ARCH_DISPLAY = [
    ("topk_sae",    "TopK SAE"),
    ("tsae_paper",  "T-SAE"),
    ("txc_base",    "TXC-base"),
    ("txc_pro",     "TXC-pro"),
]

# ---------------------------------------------------------------------------
# Setup B: best single-latent r̄_global per arch (max over T-variants, k, seed)
# Same metric as the y-axis of the single-latent scatter — keeping LHS plot
# and headline bar consistent.
# ---------------------------------------------------------------------------
def setup_b_best_global() -> dict[str, tuple[float, str]]:
    """Return {arch: (best_corr, t_label)} with seed-averaging first.

    Group by (arch, t_label, k_pos), mean over seeds, then take max
    over (t_label, k_pos) per arch. This matches the points plotted in
    the single-latent scatter (\\cref{fig:setup_b_denoising}) where each
    marker is a seed-averaged cell.
    """
    import statistics
    rows = json.loads(PROBE_JSON.read_text())
    cell = defaultdict(list)
    for r in rows:
        key = (r["arch_name"], r.get("t_label", ""), int(r["k_pos"]))
        cell[key].append(float(r["sl_mean_global"]))
    best: dict[str, tuple[float, str]] = {}
    for (arch, t, _k), vs in cell.items():
        m = statistics.mean(vs)
        cur = best.get(arch, (-1.0, ""))
        if m > cur[0]:
            best[arch] = (m, t)
    return best

# ---------------------------------------------------------------------------
# Setup D: best gAUC per arch at maximum-overlap (np=10), max over T, k.
# Hard-coded from c2.md until the leaderboard JSON is mirrored locally.
# T-SAE entry intentionally None (runs in flight).
# ---------------------------------------------------------------------------
SETUP_D_GAUC_NP10: dict[str, float | None] = {
    "topk_sae":    0.915,   # default, k=1
    "tsae_paper":  0.990,   # default, k=1 — seed-mean over 3 seeds; saturated
    "txc_base":    0.990,   # T=5 default, k=1-2 (saturated)
    "txc_pro":     0.976,   # T=2, k=1
}


def main() -> None:
    setup_b = setup_b_best_global()  # {arch: (best_sl_global, t_label)}

    archs = [a for a, _ in ARCH_DISPLAY]
    labels = [d for _, d in ARCH_DISPLAY]
    b_vals = [setup_b.get(a, (np.nan, ""))[0] for a in archs]
    b_t    = [setup_b.get(a, (np.nan, ""))[1] for a in archs]
    d_vals = [SETUP_D_GAUC_NP10.get(a, np.nan) if SETUP_D_GAUC_NP10.get(a) is not None else np.nan for a in archs]

    x = np.arange(len(archs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    b_color = "#7E57C2"   # purple — Setup B
    d_color = "#E64A19"   # orange — Setup D

    bb = ax.bar(x - w/2, b_vals, w, label=r"Setup B  (single-latent $\bar r_{\mathrm{global}}$)", color=b_color, edgecolor="black", linewidth=0.5)
    bd = ax.bar(x + w/2, d_vals, w, label=r"Setup D  ($g\mathrm{AUC}$, $n_{\mathrm{parents}}{=}10$)", color=d_color, edgecolor="black", linewidth=0.5)

    # Hatch any missing Setup D bar (T-SAE before runs land).
    for i, v in enumerate(d_vals):
        if np.isnan(v):
            ax.text(x[i] + w/2, 0.04, "in flight", ha="center", va="bottom",
                    rotation=90, fontsize=7, color="gray")

    # Numbers on top of each bar.
    for rect, val in zip(bb, b_vals):
        if not np.isnan(val):
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    for rect, val in zip(bd, d_vals):
        if not np.isnan(val):
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Best global recovery")
    ax.set_title("Best global recovery per architecture", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=8, ncol=1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    print("Setup B best sl_mean_global per arch (value, T-variant):")
    for a in archs:
        v, t = setup_b.get(a, (float("nan"), ""))
        print(f"  {a:14} {v:.3f}  ({t})")
    print("Setup D best gAUC per arch (np=10):", SETUP_D_GAUC_NP10)


if __name__ == "__main__":
    main()
