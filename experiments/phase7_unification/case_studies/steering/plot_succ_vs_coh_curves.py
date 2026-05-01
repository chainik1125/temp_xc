"""Publication-quality success-vs-coherence curves for the unified Pareto.

Shows WHY T-SAE wins unconstrained but loses at every coherence threshold:
each cell's (succ, coh) curve across strengths sweeps from (low-s, low-succ,
high-coh) to (high-s, high-succ, low-coh). T-SAE k=20's curve dives below
the coh ≥ 1.5 floor *while still rising in succ*. TXC curves stay coherent
longer.

Outputs:
  results/case_studies/plots/succ_vs_coh_curves.png         — main paper figure
  results/case_studies/plots/succ_vs_coh_curves.thumb.png

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_succ_vs_coh_curves
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"


# Subset for clarity — only top performers and anchor.
INVENTORY = [
    ("T-SAE k=20 (anchor)", "blue", "o", "-", 3.0, [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
        ("steering_paper_normalised_seed1", "tsae_paper_k20", 1),
    ]),
    ("T=2 H8 shifts=(T,) PP — coh≥1.5 winner", "red", "^", "-", 2.5, [
        ("steering_paper_window_perposition",       "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 shifts=(T,) RE — coh≥1.75 winner", "darkred", "v", "-", 2.5, [
        ("steering_paper_normalised",       "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 bare PP — coh≥2.0 winner", "orange", "s", "-", 2.5, [
        ("steering_paper_window_perposition",       "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=5 bare k_win=20 PP — closest to anchor unc", "darkgreen", "D", "--", 1.5, [
        ("steering_paper_window_perposition", "txc_bare_antidead_t5_kwin20", 42),
    ]),
    ("T=2 T-SAE warm-start PP", "gold", "P", "--", 1.5, [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 42),
    ]),
    ("T=3 grown PP", "purple", "*", "--", 1.5, [
        ("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 42),
    ]),
]


def load_curve(subdir: str, arch_id: str) -> dict | None:
    g_path = BASE / subdir / arch_id / "generations.jsonl"
    r_path = BASE / subdir / arch_id / "grades.jsonl"
    if not g_path.exists() or not r_path.exists():
        return None
    gens = [json.loads(l) for l in g_path.open()]
    grads = [json.loads(l) for l in r_path.open()]
    if len(gens) != len(grads):
        return None
    by_s = defaultdict(lambda: {"succ": [], "coh": []})
    for g, r in zip(gens, grads):
        s = g.get("s_norm", g.get("strength"))
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        by_s[s]["succ"].append(r["success_grade"])
        by_s[s]["coh"].append(r["coherence_grade"])
    out = {}
    for s, d in sorted(by_s.items()):
        out[s] = (sum(d["succ"]) / len(d["succ"]), sum(d["coh"]) / len(d["coh"]))
    return out


def mean_curve(curves: list[dict]) -> dict:
    if not curves:
        return {}
    common = set(curves[0].keys())
    for c in curves[1:]:
        common &= set(c.keys())
    out = {}
    for s in sorted(common):
        succs = [c[s][0] for c in curves]
        cohs = [c[s][1] for c in curves]
        out[s] = (sum(succs) / len(succs), sum(cohs) / len(cohs))
    return out


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Shaded coherence threshold bands
    ax.axhspan(2.0, 3.0, color="lightgreen", alpha=0.15, label="_nolegend_")
    ax.axhspan(1.5, 2.0, color="lightyellow", alpha=0.30, label="_nolegend_")
    ax.axhspan(0.0, 1.5, color="lightcoral", alpha=0.15, label="_nolegend_")

    ax.text(0.02, 2.5, "mostly coherent", color="darkgreen", fontsize=9, alpha=0.7)
    ax.text(0.02, 1.7, "between somewhat\nand mostly coherent", color="darkgoldenrod", fontsize=8, alpha=0.7)
    ax.text(0.02, 0.5, "incoherent", color="darkred", fontsize=9, alpha=0.7)

    ax.axhline(1.5, color="black", linestyle=":", linewidth=1.0, alpha=0.6, label="prereg coh ≥ 1.5")
    ax.axhline(2.0, color="darkgreen", linestyle=":", linewidth=1.0, alpha=0.5)

    for label, color, marker, linestyle, lw, specs in INVENTORY:
        seeds = [load_curve(s, a) for s, a, _sd in specs]
        seeds = [c for c in seeds if c]
        mc = mean_curve(seeds)
        if not mc:
            continue
        s_norms = sorted(mc.keys())
        succs = [mc[s][0] for s in s_norms]
        cohs = [mc[s][1] for s in s_norms]
        n = len(seeds)
        ax.plot(succs, cohs, color=color, marker=marker, linestyle=linestyle,
                linewidth=lw, markersize=10, label=f"{label} (n={n})", alpha=0.85)
        # Annotate strength at each point
        for s, succ, coh in zip(s_norms, succs, cohs):
            ax.annotate(f"{s:g}", xy=(succ, coh), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, alpha=0.6)
        # Mark unconstrained peak
        peak_idx = int(np.argmax(succs))
        ax.scatter([succs[peak_idx]], [cohs[peak_idx]], s=200, marker="*",
                   facecolor="none", edgecolor=color, linewidth=2, zorder=5)

    ax.set_xlabel("mean success grade (peak per curve = ★)", fontsize=11)
    ax.set_ylabel("mean coherence grade", fontsize=11)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0.0, 3.05)
    ax.set_title("Success vs coherence: T-SAE's peak (1.80) is in the incoherent region (★ at coh=1.40);\n"
                 "TXC architectures stay coherent — at every readable-text threshold, TXC dominates.\n"
                 "Numbers next to markers = s_norm; ★ = unconstrained peak success.",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = PLOTS_DIR / "succ_vs_coh_curves.png"
    out_thumb = PLOTS_DIR / "succ_vs_coh_curves.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_thumb}")


if __name__ == "__main__":
    main()
