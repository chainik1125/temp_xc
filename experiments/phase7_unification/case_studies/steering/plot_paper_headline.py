"""Paper-headline composite figure for the Phase 7 steering case study.

Single multi-panel figure summarising the GIGABRAIN reframe:
  (a) Top-left: succ-vs-coh curves with coherence bands, top-3 cells + anchor
  (b) Top-right: multi-coh-threshold ranking — best TXC vs anchor at each threshold
  (c) Bottom-left: Δ vs anchor at coh ≥ {1.5, 1.75, 2.0} with bootstrap CI bars
  (d) Bottom-right: AUC ranking — Δ AUC(1.5-3.0) bars per cell

Outputs:
  results/case_studies/plots/paper_headline.png
  results/case_studies/plots/paper_headline.thumb.png

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_paper_headline
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
RNG = np.random.default_rng(42)


CELLS = [
    ("T-SAE k=20 (anchor)", "blue", "o", "-", 3.0, [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
        ("steering_paper_normalised_seed1", "tsae_paper_k20", 1),
    ]),
    ("T=2 H8 PP — coh≥1.5 winner", "red", "^", "-", 2.5, [
        ("steering_paper_window_perposition", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 RE — coh≥1.75 winner", "darkred", "v", "-", 2.5, [
        ("steering_paper_normalised", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 bare PP — coh≥2.0 winner", "orange", "s", "-", 2.5, [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=2 bare RE — AUC winner", "gold", "D", "-", 2.5, [
        ("steering_paper_normalised", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
]


def load_curve(subdir, arch):
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists(): return None
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    if len(gens) != len(grads): return None
    by_s = defaultdict(lambda: {"succ": [], "coh": []})
    for gg, rr in zip(gens, grads):
        s = gg.get("s_norm", gg.get("strength"))
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None: continue
        by_s[s]["succ"].append(rr["success_grade"])
        by_s[s]["coh"].append(rr["coherence_grade"])
    out = {}
    for s, d in sorted(by_s.items()):
        if not d["succ"]: continue
        out[s] = (sum(d["succ"]) / len(d["succ"]), sum(d["coh"]) / len(d["coh"]))
    return out


def load_per_concept(subdir, arch):
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists(): return None
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    if len(gens) != len(grads): return None
    out = defaultdict(dict)
    for gg, rr in zip(gens, grads):
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None: continue
        s = gg.get("s_norm", gg.get("strength"))
        out[gg["concept_id"]][s] = (rr["success_grade"], rr["coherence_grade"])
    return dict(out)


def mean_curve(curves):
    if not curves: return {}
    common = set(curves[0].keys())
    for c in curves[1:]: common &= set(c.keys())
    out = {}
    for s in sorted(common):
        succs = [c[s][0] for c in curves]
        cohs = [c[s][1] for c in curves]
        out[s] = (sum(succs) / len(succs), sum(cohs) / len(cohs))
    return out


def average_per_concept(curves, min_seeds=None):
    if not curves: return {}
    if min_seeds is None: min_seeds = max(1, len(curves) - 1)
    cids = set()
    for c in curves: cids |= set(c.keys())
    out = {}
    for cid in cids:
        per_s = defaultdict(list)
        for c in curves:
            if cid not in c: continue
            for s, (succ, coh) in c[cid].items():
                per_s[s].append((succ, coh))
        avg = {}
        for s, pairs in per_s.items():
            if len(pairs) >= min_seeds:
                succs = [p[0] for p in pairs]
                cohs = [p[1] for p in pairs]
                avg[s] = (sum(succs)/len(succs), sum(cohs)/len(cohs))
        if avg: out[cid] = avg
    return out


def peak_at_threshold(curve, thr):
    eligible = [v[0] for v in curve.values() if v[1] >= thr]
    return max(eligible) if eligible else 0.0


def auc_succ_vs_coh(curve, coh_lo=1.5, coh_hi=3.0):
    if not curve: return 0.0
    pts = sorted(curve.values(), key=lambda v: v[1])
    succs = np.array([p[0] for p in pts])
    cohs = np.array([p[1] for p in pts])
    grid = np.linspace(coh_lo, coh_hi, 41)
    succ_interp = np.interp(grid, cohs, succs)
    return float(np.trapezoid(succ_interp, grid) / (coh_hi - coh_lo))


def bootstrap_ci_strength_uniform(per_concept_curve, anchor_per_concept, thr, n=1000):
    """Bootstrap 95% CI on Δ(cell - anchor) at strength-uniform peak15.

    For each bootstrap: resample 30 concepts with replacement, recompute
    per-cell strength-uniform peak, take Δ.
    """
    cids = sorted(set(per_concept_curve.keys()) & set(anchor_per_concept.keys()))
    if not cids: return (0.0, 0.0)
    cids_arr = np.asarray(cids)
    boot = []
    for _ in range(n):
        idx = RNG.integers(0, len(cids), len(cids))
        sampled = cids_arr[idx]
        # build by_s for sampled concepts
        by_s_cell = defaultdict(list)
        by_s_anc = defaultdict(list)
        for cid in sampled:
            for s, (succ, coh) in per_concept_curve[cid].items():
                by_s_cell[s].append((succ, coh))
            for s, (succ, coh) in anchor_per_concept[cid].items():
                by_s_anc[s].append((succ, coh))
        def peak(by_s):
            best = -1.0
            for s, items in by_s.items():
                if len(items) < len(sampled): continue
                mc = sum(it[1] for it in items) / len(items)
                if mc < thr: continue
                ms = sum(it[0] for it in items) / len(items)
                if ms > best: best = ms
            return max(best, 0.0)
        boot.append(peak(by_s_cell) - peak(by_s_anc))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def main():
    # Load curves and per-concept data
    cells_data = {}
    for label, color, marker, ls, lw, specs in CELLS:
        seeds_curves = [load_curve(s, a) for s, a, _ in specs]
        seeds_curves = [c for c in seeds_curves if c]
        seeds_pc = [load_per_concept(s, a) for s, a, _ in specs]
        seeds_pc = [c for c in seeds_pc if c]
        if not seeds_curves: continue
        cells_data[label] = {
            "color": color, "marker": marker, "ls": ls, "lw": lw,
            "n_seeds": len(seeds_curves),
            "mean_curve": mean_curve(seeds_curves),
            "per_concept": average_per_concept(seeds_pc),
        }
        print(f"loaded {label}: {len(seeds_curves)} seeds, {len(cells_data[label]['per_concept'])} concepts")

    anchor_label = "T-SAE k=20 (anchor)"
    anchor_curve = cells_data[anchor_label]["mean_curve"]
    anchor_pc = cells_data[anchor_label]["per_concept"]

    THRESHOLDS = [1.5, 1.75, 2.0]

    # === Build figure ===
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # (a) succ vs coh curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhspan(2.0, 3.0, color="lightgreen", alpha=0.15)
    ax1.axhspan(1.5, 2.0, color="lightyellow", alpha=0.30)
    ax1.axhspan(0.0, 1.5, color="lightcoral", alpha=0.15)
    ax1.text(0.05, 2.5, "mostly\ncoherent", color="darkgreen", fontsize=8, alpha=0.7)
    ax1.text(0.05, 1.7, "borderline", color="darkgoldenrod", fontsize=8, alpha=0.7)
    ax1.text(0.05, 0.6, "incoherent", color="darkred", fontsize=8, alpha=0.7)
    ax1.axhline(1.5, color="black", linestyle=":", alpha=0.6)
    ax1.axhline(2.0, color="darkgreen", linestyle=":", alpha=0.5)
    for label, d in cells_data.items():
        mc = d["mean_curve"]
        s_norms = sorted(mc.keys())
        succs = [mc[s][0] for s in s_norms]
        cohs = [mc[s][1] for s in s_norms]
        ax1.plot(succs, cohs, color=d["color"], marker=d["marker"], linestyle=d["ls"],
                 linewidth=d["lw"], markersize=8,
                 label=f"{label.split('—')[0].strip()} (n={d['n_seeds']})", alpha=0.85)
        peak_idx = int(np.argmax(succs))
        ax1.scatter([succs[peak_idx]], [cohs[peak_idx]], s=200, marker="*",
                    facecolor="none", edgecolor=d["color"], linewidth=2, zorder=5)
    ax1.set_xlabel("mean success grade")
    ax1.set_ylabel("mean coherence grade")
    ax1.set_xlim(-0.05, 2.05)
    ax1.set_ylim(0.0, 3.05)
    ax1.set_title("(a) Steering curves — T-SAE peak ★ in incoherent zone, TXC peaks ★ in coherent zones",
                  fontsize=10)
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(alpha=0.3)

    # (b) Multi-coh-threshold ranking — best TXC at each threshold
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ["unconstrained"] + [f"coh≥{t}" for t in THRESHOLDS] + ["AUC(1.5-3)"]
    txc_cells = [c for c in cells_data if c != anchor_label]
    anchor_vals_full = []
    best_txc_vals_full = []
    best_txc_labels = []
    for m in metrics:
        if m == "unconstrained":
            a_v = max(v[0] for v in anchor_curve.values())
            ts = [(c, max(v[0] for v in cells_data[c]["mean_curve"].values())) for c in txc_cells]
        elif m.startswith("AUC"):
            a_v = auc_succ_vs_coh(anchor_curve, 1.5, 3.0)
            ts = [(c, auc_succ_vs_coh(cells_data[c]["mean_curve"], 1.5, 3.0)) for c in txc_cells]
        else:
            thr = float(m.replace("coh≥", ""))
            a_v = peak_at_threshold(anchor_curve, thr)
            ts = [(c, peak_at_threshold(cells_data[c]["mean_curve"], thr)) for c in txc_cells]
        ts.sort(key=lambda x: x[1], reverse=True)
        anchor_vals_full.append(a_v)
        best_txc_vals_full.append(ts[0][1])
        best_txc_labels.append(ts[0][0].split('—')[0].strip())
    x = np.arange(len(metrics))
    w = 0.36
    ax2.bar(x - w/2, anchor_vals_full, w, color="blue", label="T-SAE k=20", edgecolor="black")
    ax2.bar(x + w/2, best_txc_vals_full, w, color="darkred", label="best TXC", edgecolor="black")
    for i, (av, tv, tl) in enumerate(zip(anchor_vals_full, best_txc_vals_full, best_txc_labels)):
        d = tv - av
        col = "darkgreen" if d > 0 else "red"
        ax2.text(i, max(av, tv) + 0.07, f"Δ={d:+.2f}", ha="center", fontsize=8, fontweight="bold", color=col)
        ax2.text(i, av - 0.07, f"{av:.2f}", ha="center", fontsize=7, va="top", color="white", fontweight="bold")
        ax2.text(i, tv - 0.07, f"{tv:.2f}", ha="center", fontsize=7, va="top", color="white", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=8, rotation=10)
    ax2.set_ylabel("metric value")
    ax2.set_title("(b) Best TXC vs anchor across all metrics — anchor only wins unconstrained (= incoherent text)",
                  fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, max(anchor_vals_full + best_txc_vals_full) * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    # (c) Δ at each threshold with bootstrap CIs
    ax3 = fig.add_subplot(gs[1, 0])
    print("computing bootstrap CIs (this can take ~20s per cell × 3 thresholds)...")
    bar_labels = []
    bar_deltas = []
    bar_cis = []
    bar_colors = []
    for thr in THRESHOLDS:
        for label in txc_cells:
            d = cells_data[label]
            cell_mc = d["mean_curve"]
            cell_pc = d["per_concept"]
            cell_peak = peak_at_threshold(cell_mc, thr)
            anc_peak = peak_at_threshold(anchor_curve, thr)
            delta = cell_peak - anc_peak
            ci = bootstrap_ci_strength_uniform(cell_pc, anchor_pc, thr, n=300)
            bar_labels.append(f"coh≥{thr}\n{label.split('—')[0].strip()}")
            bar_deltas.append(delta)
            bar_cis.append(ci)
            bar_colors.append(d["color"])
    yerr = [[d - ci[0] for d, ci in zip(bar_deltas, bar_cis)],
            [ci[1] - d for d, ci in zip(bar_deltas, bar_cis)]]
    x = np.arange(len(bar_labels))
    bars = ax3.bar(x, bar_deltas, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax3.errorbar(x, bar_deltas, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1)
    for i, (d, ci) in enumerate(zip(bar_deltas, bar_cis)):
        sig = ci[0] > 0
        marker = "***" if sig else "ns"
        ax3.text(i, d + max(0.02, ci[1] - d + 0.05), marker, ha="center", fontsize=7,
                 fontweight="bold", color="darkgreen" if sig else "gray")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axhline(0.27, color="darkgreen", linestyle="--", alpha=0.6, label="WIN threshold (+0.27)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha="right")
    ax3.set_ylabel("Δ vs anchor (peak success)")
    ax3.set_title("(c) Δ vs anchor with 95% bootstrap CI — *** = significant (CI lower bound > 0)",
                  fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # (d) AUC ranking — Δ AUC(1.5-3.0) per cell
    ax4 = fig.add_subplot(gs[1, 1])
    auc_anchor = auc_succ_vs_coh(anchor_curve, 1.5, 3.0)
    cell_aucs = [(label, auc_succ_vs_coh(cells_data[label]["mean_curve"], 1.5, 3.0))
                 for label in cells_data]
    cell_aucs.sort(key=lambda x: x[1], reverse=True)
    labels = [a[0].split('—')[0].strip() for a in cell_aucs]
    aucs = [a[1] for a in cell_aucs]
    colors = [cells_data[a[0]]["color"] for a in cell_aucs]
    bars2 = ax4.barh(range(len(labels)), aucs, color=colors, edgecolor="black", linewidth=0.5)
    ax4.invert_yaxis()
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.axvline(auc_anchor, color="blue", linestyle="--", alpha=0.6,
                label=f"anchor AUC={auc_anchor:.3f}")
    for i, v in enumerate(aucs):
        ax4.text(v + 0.005, i, f"{v:.3f}\n(Δ={v-auc_anchor:+.3f})",
                 va="center", fontsize=7)
    ax4.set_xlabel("AUC(1.5-3.0)  =  ∫ succ(coh) d(coh) / 1.5")
    ax4.set_title("(d) AUC ranking — TXC dominates Han's pre-stated alternative metric",
                  fontsize=10)
    ax4.set_xlim(0, max(aucs) * 1.25)
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(axis="x", alpha=0.3)

    fig.suptitle("Phase 7 steering case study — TXC dominates T-SAE k=20 across coherence-aware metrics\n"
                 "(matched per-token sparsity k_pos=20, Gemma-2-2b L12 anchor)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = PLOTS_DIR / "paper_headline.png"
    out_thumb = PLOTS_DIR / "paper_headline.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_thumb}")


if __name__ == "__main__":
    main()
