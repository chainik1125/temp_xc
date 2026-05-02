"""Auto-updating Phase 7 steering dashboard.

Discovers all cells in results/case_studies/ that have generations.jsonl +
grades.jsonl, builds a unified results table with bootstrap CIs, and
generates the headline plots. Run after any new cell is graded.

Outputs:
  results/case_studies/plots/auto_dashboard.json — full results
  results/case_studies/plots/auto_dashboard_ranking.png — multi-threshold ranking
  results/case_studies/plots/auto_dashboard_summary.md — markdown summary

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.steering.auto_dashboard
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
RNG = np.random.default_rng(42)
THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5]


# Known protocol → seed mapping (subdir naming convention)
PROTOCOL_PATTERNS = {
    "right-edge": [("steering_paper_normalised", 42),
                   ("steering_paper_normalised_seed1", 1),
                   ("steering_paper_normalised_seed2", 2)],
    "per-position": [("steering_paper_window_perposition", 42),
                     ("steering_paper_window_perposition_seed1", 1),
                     ("steering_paper_window_perposition_seed2", 2)],
    "tiled-broadcast": [("steering_paper_window_tiled_broadcast", 42),
                        ("steering_paper_window_tiled_broadcast_seed1", 1),
                        ("steering_paper_window_tiled_broadcast_seed2", 2)],
    "encoded-broadcast": [("steering_paper_window_encoded_broadcast", 42),
                          ("steering_paper_window_encoded_broadcast_seed1", 1),
                          ("steering_paper_window_encoded_broadcast_seed2", 2)],
}


def discover_cells():
    """Find all (arch_id, protocol, seed) cells with grades.jsonl on disk."""
    cells = defaultdict(list)  # (arch_id, protocol_label) -> [(subdir, seed)]
    for protocol_label, subdirs in PROTOCOL_PATTERNS.items():
        for subdir, seed in subdirs:
            sub_path = BASE / subdir
            if not sub_path.exists():
                continue
            for arch_dir in sub_path.iterdir():
                if not arch_dir.is_dir():
                    continue
                arch_id = arch_dir.name
                grades_path = arch_dir / "grades.jsonl"
                gens_path = arch_dir / "generations.jsonl"
                if grades_path.exists() and gens_path.exists():
                    n_grades = sum(1 for _ in grades_path.open())
                    n_gens = sum(1 for _ in gens_path.open())
                    if n_grades >= 200 and n_gens >= 200:
                        cells[(arch_id, protocol_label)].append((subdir, seed))
    return cells


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
                avg[s] = (sum(p[0] for p in pairs)/len(pairs),
                          sum(p[1] for p in pairs)/len(pairs))
        if avg: out[cid] = avg
    return out


def per_strength_curve(per_concept):
    by_s = defaultdict(lambda: {"succ": [], "coh": []})
    for cid, sd in per_concept.items():
        for s, (succ, coh) in sd.items():
            by_s[s]["succ"].append(succ)
            by_s[s]["coh"].append(coh)
    out = {}
    for s, d in by_s.items():
        if not d["succ"]: continue
        out[s] = (sum(d["succ"])/len(d["succ"]), sum(d["coh"])/len(d["coh"]))
    return out


def peak_at_threshold(curve, thr):
    eligible = [v[0] for v in curve.values() if v[1] >= thr]
    return max(eligible) if eligible else 0.0


def auc_succ_vs_coh(curve, lo=1.5, hi=3.0):
    if not curve: return 0.0
    pts = sorted(curve.values(), key=lambda v: v[1])
    succs = np.array([p[0] for p in pts])
    cohs = np.array([p[1] for p in pts])
    grid = np.linspace(lo, hi, 41)
    return float(np.trapezoid(np.interp(grid, cohs, succs), grid) / (hi - lo))


def bootstrap_ci_peak(per_concept_cell, per_concept_anchor, thr, n=500):
    """Proper bootstrap CI on Δ(cell-anchor) at strength-uniform peak."""
    cids = sorted(set(per_concept_cell.keys()) & set(per_concept_anchor.keys()))
    if not cids: return (0.0, 0.0)
    cids_arr = np.asarray(cids)
    boot = []
    for _ in range(n):
        idx = RNG.integers(0, len(cids), len(cids))
        sampled = cids_arr[idx]
        by_s_cell = defaultdict(list)
        by_s_anc = defaultdict(list)
        for cid in sampled:
            for s, (succ, coh) in per_concept_cell[cid].items():
                by_s_cell[s].append((succ, coh))
            for s, (succ, coh) in per_concept_anchor[cid].items():
                by_s_anc[s].append((succ, coh))
        n_sampled = len(sampled)
        def peak(by_s):
            best = -1.0
            for s, items in by_s.items():
                if len(items) < n_sampled: continue
                mc = sum(it[1] for it in items) / len(items)
                if mc < thr: continue
                ms = sum(it[0] for it in items) / len(items)
                if ms > best: best = ms
            return max(best, 0.0)
        boot.append(peak(by_s_cell) - peak(by_s_anc))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def main():
    cells = discover_cells()
    print(f"discovered {len(cells)} (arch, protocol) cells")

    # Build per-cell metrics
    results = {}
    pc_data = {}
    for (arch_id, protocol), specs in cells.items():
        seeds_pc = [load_per_concept(s, arch_id) for s, _ in specs]
        seeds_pc = [c for c in seeds_pc if c]
        if not seeds_pc:
            continue
        pc = average_per_concept(seeds_pc)
        curve = per_strength_curve(pc)
        pc_data[(arch_id, protocol)] = pc
        results[(arch_id, protocol)] = {
            "arch_id": arch_id,
            "protocol": protocol,
            "n_seeds": len(seeds_pc),
            "peak_unc": max(v[0] for v in curve.values()) if curve else 0.0,
            **{f"peak_coh_ge_{t:.2f}": peak_at_threshold(curve, t) for t in THRESHOLDS},
            "auc_1.5_3.0": auc_succ_vs_coh(curve, 1.5, 3.0),
            "auc_1.75_3.0": auc_succ_vs_coh(curve, 1.75, 3.0),
        }

    # Find anchor (T-SAE k=20)
    anchor_key = ("tsae_paper_k20", "right-edge")
    if anchor_key not in results:
        print("warning: anchor (T-SAE k=20 RE) missing")
        return

    # Compute deltas + bootstrap CIs for multi-seed cells
    anchor = results[anchor_key]
    anchor_pc = pc_data[anchor_key]
    for k, r in results.items():
        if k == anchor_key: continue
        r["delta_unc"] = r["peak_unc"] - anchor["peak_unc"]
        for t in THRESHOLDS:
            r[f"delta_peak_coh_ge_{t:.2f}"] = r[f"peak_coh_ge_{t:.2f}"] - anchor[f"peak_coh_ge_{t:.2f}"]
        r["delta_auc_1.5_3.0"] = r["auc_1.5_3.0"] - anchor["auc_1.5_3.0"]
        r["delta_auc_1.75_3.0"] = r["auc_1.75_3.0"] - anchor["auc_1.75_3.0"]
        # Bootstrap CI for multi-seed cells
        if r["n_seeds"] >= 2:
            for t in THRESHOLDS:
                ci = bootstrap_ci_peak(pc_data[k], anchor_pc, t, n=300)
                r[f"ci_peak_coh_ge_{t:.2f}"] = ci

    # Save JSON
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    json_path = PLOTS_DIR / "auto_dashboard.json"
    json_data = {f"{a}|{p}": r for (a, p), r in results.items()}
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"saved {json_path}")

    # Markdown summary — top 10 cells per metric
    lines = ["## Auto-updating dashboard — Phase 7 steering case study", "",
             f"All cells discovered with grades.jsonl + generations.jsonl ≥ 200 rows.", "",
             "### Anchor: T-SAE k=20", "",
             f"unc={anchor['peak_unc']:.3f}, "
             + ", ".join(f"≥{t}={anchor[f'peak_coh_ge_{t:.2f}']:.3f}" for t in THRESHOLDS)
             + f", AUC(1.5-3.0)={anchor['auc_1.5_3.0']:.3f}", ""]

    for metric, label in [("delta_peak_coh_ge_1.50", "coh ≥ 1.5 (prereg)"),
                          ("delta_peak_coh_ge_1.75", "coh ≥ 1.75"),
                          ("delta_peak_coh_ge_2.00", "coh ≥ 2.0"),
                          ("delta_auc_1.5_3.0",     "AUC(1.5-3.0)")]:
        lines.append(f"\n### Top 10 cells by Δ {label}\n")
        lines.append(f"| arch + protocol | n | Δ | base | seeds_data |")
        lines.append(f"|---|---:|---:|---:|---|")
        sorted_cells = sorted(
            ((a, p, r) for (a, p), r in results.items() if (a, p) != anchor_key),
            key=lambda x: x[2].get(metric, 0), reverse=True
        )[:10]
        for a, p, r in sorted_cells:
            base_metric = metric.replace("delta_", "")
            base = r.get(base_metric, 0)
            lines.append(f"| {a} {p} | {r['n_seeds']} | {r.get(metric, 0):+.3f} | {base:.3f} | n={r['n_seeds']} |")
    md_path = PLOTS_DIR / "auto_dashboard_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"saved {md_path}")

    # Ranking plot — best TXC at each threshold
    fig, ax = plt.subplots(figsize=(13, 6))
    metrics = ["unc"] + [f"coh ≥ {t}" for t in THRESHOLDS] + ["AUC(1.5-3.0)"]
    metric_keys = ["peak_unc"] + [f"peak_coh_ge_{t:.2f}" for t in THRESHOLDS] + ["auc_1.5_3.0"]
    txc_results = [(a, p, r) for (a, p), r in results.items() if (a, p) != anchor_key]
    anchor_vals = [anchor[k] for k in metric_keys]
    best_txc_vals = []
    best_txc_labels = []
    for k in metric_keys:
        best = max(txc_results, key=lambda x: x[2].get(k, 0))
        best_txc_vals.append(best[2].get(k, 0))
        # Make a short label
        a, p, _ = best
        short = re.sub(r"^txc_(?:bare_antidead_)?", "", a)
        short = re.sub(r"_kpos20.*$", "", short)
        best_txc_labels.append(f"{short}\n{p[:2]}\nn={best[2]['n_seeds']}")

    x = np.arange(len(metrics))
    w = 0.38
    ax.bar(x - w/2, anchor_vals, w, color="blue", label="T-SAE k=20", edgecolor="black")
    ax.bar(x + w/2, best_txc_vals, w, color="darkred", label="best TXC at metric", edgecolor="black")
    for i, (av, tv, tl) in enumerate(zip(anchor_vals, best_txc_vals, best_txc_labels)):
        d = tv - av
        col = "darkgreen" if d > 0 else "red"
        ax.text(i, max(av, tv) + 0.06, f"Δ={d:+.2f}", ha="center", fontsize=8, fontweight="bold", color=col)
        ax.text(i + w/2, -0.15, tl, ha="center", fontsize=6, rotation=20, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, rotation=10)
    ax.set_ylabel("metric value")
    ax.set_title(f"Phase 7 steering — auto-discovered best TXC across {len(results)} cells", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = PLOTS_DIR / "auto_dashboard_ranking.png"
    out_thumb = PLOTS_DIR / "auto_dashboard_ranking.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
