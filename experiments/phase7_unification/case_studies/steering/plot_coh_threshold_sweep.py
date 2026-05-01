"""Multi-coherence-threshold sweep — paper-grade reframing.

Y's GIGABRAIN insight (2026-04-30): T-SAE k=20's 1.80 unconstrained peak
occurs at coh=1.40 (between "somewhat coherent" and "mostly coherent",
i.e. essentially low-quality text). When we re-grade every cell at
*tighter* coherence thresholds, T-SAE's curve collapses — while TXC
architectures sustain high success past coh ≥ 1.75 and ≥ 2.0.

Outputs:
  results/case_studies/plots/coh_threshold_sweep.json     — per-cell peaks at every threshold
  results/case_studies/plots/coh_threshold_sweep.png      — bar chart (best TXC vs T-SAE per threshold)
  results/case_studies/plots/coh_threshold_sweep_full.png — full ranking grid

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_coh_threshold_sweep
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5]


# Inventory: (label, color, list of (subdir, arch_id, seed))
INVENTORY = [
    ("T-SAE k=20 (anchor)", "blue", [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
    ]),
    ("T=2 H8 shifts=(T,) PP", "red", [
        ("steering_paper_window_perposition",       "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 shifts=(T,) RE", "salmon", [
        ("steering_paper_normalised",       "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 bare PP", "orange", [
        ("steering_paper_window_perposition",       "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=2 bare RE", "gold", [
        ("steering_paper_normalised",       "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=2 T-SAE warm-start PP", "darkkhaki", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 42),
    ]),
    ("T=2 T-SAE warm-start RE", "olive", [
        ("steering_paper_normalised", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 42),
    ]),
    ("T=5 bare k_win=20 PP", "darkgreen", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t5_kwin20", 42),
    ]),
    ("T=5 bare k_win=20 RE", "lightgreen", [
        ("steering_paper_normalised", "txc_bare_antidead_t5_kwin20", 42),
    ]),
    ("T=3 H8 PP", "violet", [
        ("steering_paper_window_perposition", "txc_h8_t3_kpos20_shifts3", 42),
    ]),
    ("T=3 grown PP", "purple", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 42),
    ]),
    ("T=4 grown chain PP", "indigo", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t4_kpos20_grownChainFromT3", 42),
    ]),
    ("T=5 grown chain PP", "navy", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t5_kpos20_grownChainFromT4", 42),
    ]),
    ("T=5 H8 shifts=(T,) PP", "darkred", [
        ("steering_paper_window_perposition",       "txc_h8_t5_kpos20_shifts5", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t5_kpos20_shifts5", 1),
    ]),
    ("T=5 bare PP", "limegreen", [
        ("steering_paper_window_perposition",       "txc_bare_antidead_t5_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t5_kpos20", 1),
    ]),
    ("T=3 bare W-cellC PP", "cyan", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20", 42),
    ]),
    ("T=5 matry W-cellE PP", "teal", [
        ("steering_paper_window_perposition", "agentic_txc_02_kpos20", 42),
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


def peak_at_threshold(curve: dict, thr: float) -> tuple[float, float]:
    """Return (best_succ, s_norm_at_peak); 0.0/None if nothing meets threshold."""
    candidates = [(succ, s, coh) for s, (succ, coh) in curve.items() if coh >= thr]
    if not candidates:
        return 0.0, None
    succ, s, _ = max(candidates, key=lambda t: t[0])
    return succ, s


def main() -> None:
    out: list[dict] = []
    for label, color, specs in INVENTORY:
        seeds = [load_curve(sub, arch) for sub, arch, _sd in specs]
        seeds = [c for c in seeds if c]
        mc = mean_curve(seeds)
        if not mc:
            continue
        peak_unc = max(v[0] for v in mc.values())
        peaks = {f"coh_ge_{t:.2f}": peak_at_threshold(mc, t) for t in THRESHOLDS}
        peaks_simple = {f"coh_ge_{t:.2f}": peaks[f"coh_ge_{t:.2f}"][0] for t in THRESHOLDS}
        out.append({
            "label": label,
            "color": color,
            "n_seeds": len(seeds),
            "peak_unc": peak_unc,
            **peaks_simple,
            "curve": [(s, succ, coh) for s, (succ, coh) in mc.items()],
        })

    json_path = PLOTS_DIR / "coh_threshold_sweep.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"saved {json_path}")

    anchor = next(r for r in out if "anchor" in r["label"].lower())

    # Plot 1: best-TXC vs anchor at each threshold (single panel, 6 grouped bars)
    metrics = ["peak_unc"] + [f"coh_ge_{t:.2f}" for t in THRESHOLDS]
    metric_labels = ["unconstrained\n(any coh)"] + [f"coh ≥ {t}" for t in THRESHOLDS]

    txc_rows = [r for r in out if "anchor" not in r["label"].lower()]
    best_txc_per_metric = {}
    for m in metrics:
        best = max(txc_rows, key=lambda r: r[m])
        best_txc_per_metric[m] = (best["label"], best[m], best["n_seeds"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    w = 0.38
    anchor_vals = [anchor[m] for m in metrics]
    txc_vals = [best_txc_per_metric[m][1] for m in metrics]
    txc_labels = [best_txc_per_metric[m][0] for m in metrics]

    ax.bar(x - w/2, anchor_vals, w, color="blue", label="T-SAE k=20 (anchor)", edgecolor="black")
    ax.bar(x + w/2, txc_vals, w, color="red", label="best TXC at threshold", edgecolor="black")
    for i, (av, tv, tl) in enumerate(zip(anchor_vals, txc_vals, txc_labels)):
        ax.text(i - w/2, av + 0.04, f"{av:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tv + 0.04, f"{tv:.2f}", ha="center", fontsize=8, fontweight="bold", color="darkred")
        delta = tv - av
        col = "darkgreen" if delta > 0 else "red"
        ax.text(i, max(av, tv) + 0.18, f"Δ={delta:+.3f}", ha="center", fontsize=9, fontweight="bold", color=col)
        # Annotate which TXC won at this threshold
        ax.text(i + w/2, -0.12, tl.replace(" shifts=(T,)", "").replace("warm-start", "WS"),
                ha="center", fontsize=7, rotation=20, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("peak success grade")
    ax.set_title("T-SAE's unconstrained peak is at incoherent text; TXC dominates at every coh ≥ 1.5\n"
                 "(best TXC architecture per coh threshold, multi-seed mean-curve where available)")
    ax.set_ylim(0, max(2.2, max(txc_vals + anchor_vals) + 0.4))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = PLOTS_DIR / "coh_threshold_sweep.png"
    out_thumb = PLOTS_DIR / "coh_threshold_sweep.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")

    # Plot 2: all cells at each threshold (full grid)
    fig2, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for idx, m in enumerate(metrics):
        ax = axes[idx]
        rows_sorted = sorted(out, key=lambda r: r[m], reverse=True)
        names = [r["label"] for r in rows_sorted]
        vals = [r[m] for r in rows_sorted]
        cols = [r["color"] for r in rows_sorted]
        is_anchor = ["anchor" in r["label"].lower() for r in rows_sorted]
        edges = ["gold" if (idx > 0 and v - anchor[m] >= 0.27 and not anc)
                 else ("blue" if anc else "black")
                 for v, anc in zip(vals, is_anchor)]
        widths = [3.0 if (idx > 0 and v - anchor[m] >= 0.27 and not anc) else 1.0 for v, anc in zip(vals, is_anchor)]
        ax.barh(range(len(rows_sorted)), vals, color=cols, edgecolor=edges, linewidth=widths)
        ax.set_yticks(range(len(rows_sorted)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(anchor[m], color="blue", linestyle="--", alpha=0.6, label=f"anchor {anchor[m]:.2f}")
        ax.axvline(anchor[m] + 0.27, color="darkgreen", linestyle=":", alpha=0.6, label="WIN +0.27")
        ax.set_title(metric_labels[idx], fontsize=10)
        ax.set_xlabel("peak success")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(axis="x", alpha=0.3)
    fig2.suptitle("Full per-cell ranking at each coh threshold (3-seed mean-curve where available)\n"
                  "Gold edge = TXC cell crossing WIN threshold (anchor + 0.27)",
                  fontsize=12, y=1.0)
    fig2.tight_layout()
    out_png2 = PLOTS_DIR / "coh_threshold_sweep_full.png"
    out_thumb2 = PLOTS_DIR / "coh_threshold_sweep_full.thumb.png"
    fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
    fig2.savefig(out_thumb2, dpi=48, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out_png2}")


if __name__ == "__main__":
    main()
