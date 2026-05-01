"""Build the definitive Phase 7 steering case-study results table.

Single source of truth covering all metrics × all 3-seed-verified cells.
Outputs JSON + markdown table for the paper appendix.

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.build_definitive_table
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
RNG = np.random.default_rng(42)

THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5]


# All cells at matched k_pos=20, with their multi-seed protocol+arch combinations
CELLS = [
    ("T-SAE k=20 (anchor, multi-seed)", [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
        ("steering_paper_normalised_seed1", "tsae_paper_k20", 1),
    ]),
    # T=2 H8 multi-seed
    ("T=2 H8 shifts=(T,) PP", [
        ("steering_paper_window_perposition", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 shifts=(T,) RE", [
        ("steering_paper_normalised", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    # T=2 bare multi-seed
    ("T=2 bare-antidead PP", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=2 bare-antidead RE", [
        ("steering_paper_normalised", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    # T=5 H8 / bare 2-seed
    ("T=5 H8 shifts=(T,) PP", [
        ("steering_paper_window_perposition", "txc_h8_t5_kpos20_shifts5", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t5_kpos20_shifts5", 1),
    ]),
    ("T=5 H8 shifts=(T,) RE", [
        ("steering_paper_normalised", "txc_h8_t5_kpos20_shifts5", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t5_kpos20_shifts5", 1),
    ]),
    ("T=5 bare-antidead PP", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t5_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t5_kpos20", 1),
    ]),
    ("T=5 bare-antidead RE", [
        ("steering_paper_normalised", "txc_bare_antidead_t5_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t5_kpos20", 1),
    ]),
    # Single-seed cells (in coh-threshold sweep but not yet multi-seed verified)
    ("T=3 H8 shifts=(T,) PP (1sd)", [("steering_paper_window_perposition", "txc_h8_t3_kpos20_shifts3", 42)]),
    ("T=3 grown PP (1sd)", [("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 42)]),
    ("T=4 grown chain PP (1sd)", [("steering_paper_window_perposition", "txc_bare_antidead_t4_kpos20_grownChainFromT3", 42)]),
    ("T=5 grown chain PP (1sd)", [("steering_paper_window_perposition", "txc_bare_antidead_t5_kpos20_grownChainFromT4", 42)]),
    # NEW 3-seed cells (T=3 grown, T-SAE WS, Galaxy 4, Galaxy 6)
    ("T=3 grown PP (3sd)", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 2),
    ]),
    ("T=3 grown RE (3sd)", [
        ("steering_paper_normalised", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t3_kpos20_grownFromT2sd42", 2),
    ]),
    ("T-SAE WS PP (3sd)", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 2),
    ]),
    ("T-SAE WS RE (3sd)", [
        ("steering_paper_normalised", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t2_kpos20_ws_tsae_encoder", 2),
    ]),
    ("Galaxy 4 hierarchical PP (3sd)", [
        ("steering_paper_window_perposition", "txc_galaxy4_t2_kw10_kp10", 42),
        ("steering_paper_window_perposition_seed1", "txc_galaxy4_t2_kw10_kp10", 1),
        ("steering_paper_window_perposition_seed2", "txc_galaxy4_t2_kw10_kp10", 2),
    ]),
    ("Galaxy 4 hierarchical RE (3sd)", [
        ("steering_paper_normalised", "txc_galaxy4_t2_kw10_kp10", 42),
        ("steering_paper_normalised_seed1", "txc_galaxy4_t2_kw10_kp10", 1),
        ("steering_paper_normalised_seed2", "txc_galaxy4_t2_kw10_kp10", 2),
    ]),
    ("Galaxy 6 max-pool PP (3sd)", [
        ("steering_paper_window_perposition", "txc_maxpool_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_maxpool_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_maxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 6 max-pool RE (3sd)", [
        ("steering_paper_normalised", "txc_maxpool_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_maxpool_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_maxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 8 soft-max-pool PP (3sd)", [
        ("steering_paper_window_perposition", "txc_softmaxpool_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_softmaxpool_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_softmaxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 8 soft-max-pool RE (3sd)", [
        ("steering_paper_normalised", "txc_softmaxpool_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_softmaxpool_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_softmaxpool_t2_kpos20", 2),
    ]),
    # NOTE: W's mystery archs labeled "MaxPool-merge"/"Contrastive-merge" below
    # use the canonical labels — Y's "W TXCMaxPoolMergeH8" entries deduped on rebase.
    ("T=5 bare k_win=20 PP (1sd)", [("steering_paper_window_perposition", "txc_bare_antidead_t5_kwin20", 42)]),
    ("T=3 bare W cellC PP (1sd)", [("steering_paper_window_perposition", "txc_bare_antidead_t3_kpos20", 42)]),
    ("T=5 matry W cellE PP (1sd)", [("steering_paper_window_perposition", "agentic_txc_02_kpos20", 42)]),
    # W's MYSTERY archs (n=3 multi-seed)
    ("MaxPool-merge RE n=3", [
        ("steering_paper_normalised", "txc_maxpool_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_maxpool_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_maxpool_h8_t2_kpos20_shifts2", 2),
    ]),
    ("MaxPool-merge PP n=3", [
        ("steering_paper_window_perposition", "txc_maxpool_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_maxpool_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_maxpool_h8_t2_kpos20_shifts2", 2),
    ]),
    ("Contrastive-merge RE n=3", [
        ("steering_paper_normalised", "txc_contrastive_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_contrastive_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_contrastive_h8_t2_kpos20_shifts2", 2),
    ]),
    ("Contrastive-merge PP n=3", [
        ("steering_paper_window_perposition", "txc_contrastive_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_contrastive_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_contrastive_h8_t2_kpos20_shifts2", 2),
    ]),
    ("Contrastive-merge V6 n=3", [
        ("steering_paper_window_dec_broadcast", "txc_contrastive_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_dec_broadcast_seed1", "txc_contrastive_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_dec_broadcast_seed2", "txc_contrastive_h8_t2_kpos20_shifts2", 2),
    ]),
]


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
                avg[s] = (sum(p[0] for p in pairs)/len(pairs), sum(p[1] for p in pairs)/len(pairs))
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


def bootstrap_ci_peak(per_concept_cell, per_concept_anchor, thr, n=1000):
    """Bootstrap 95% CI on Δ(cell - anchor) at strength-uniform peak.
    Resample concepts WITH replacement; preserve duplicates (don't dedupe via dict).
    """
    cids = sorted(set(per_concept_cell.keys()) & set(per_concept_anchor.keys()))
    if not cids: return (0.0, 0.0)
    cids_arr = np.asarray(cids)
    boot = []
    for _ in range(n):
        idx = RNG.integers(0, len(cids), len(cids))
        sampled = cids_arr[idx]
        # Accumulate (succ, coh) per s_norm directly from sampled (with duplicates)
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
                # Strict: only include strengths where all sampled concepts contributed
                if len(items) < n_sampled: continue
                mc = sum(it[1] for it in items) / len(items)
                if mc < thr: continue
                ms = sum(it[0] for it in items) / len(items)
                if ms > best: best = ms
            return max(best, 0.0)
        boot.append(peak(by_s_cell) - peak(by_s_anc))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def main():
    cells_data = {}
    for label, specs in CELLS:
        seeds_pc = [load_per_concept(s, a) for s, a, _ in specs]
        seeds_pc = [c for c in seeds_pc if c]
        if not seeds_pc:
            print(f"missing: {label}")
            continue
        cells_data[label] = {
            "n_seeds": len(seeds_pc),
            "per_concept": average_per_concept(seeds_pc),
        }
    print(f"loaded {len(cells_data)} cells")

    anchor_label = "T-SAE k=20 (anchor, multi-seed)"
    anchor_pc = cells_data[anchor_label]["per_concept"]

    # Compute all metrics
    results = {}
    for label, data in cells_data.items():
        pc = data["per_concept"]
        curve = per_strength_curve(pc)
        results[label] = {
            "n_seeds": data["n_seeds"],
            "peak_unc": max(v[0] for v in curve.values()) if curve else 0.0,
            **{f"peak_coh_ge_{t:.2f}": peak_at_threshold(curve, t) for t in THRESHOLDS},
            "auc_1.5_3.0": auc_succ_vs_coh(curve, 1.5, 3.0),
            "auc_1.0_3.0": auc_succ_vs_coh(curve, 1.0, 3.0),
            "auc_1.75_3.0": auc_succ_vs_coh(curve, 1.75, 3.0),
        }

    # Compute deltas vs anchor + bootstrap CIs for multi-seed cells
    for label, r in results.items():
        if anchor_label == label: continue
        anchor_r = results[anchor_label]
        r["delta_unc"] = r["peak_unc"] - anchor_r["peak_unc"]
        for t in THRESHOLDS:
            k = f"peak_coh_ge_{t:.2f}"
            r[f"delta_{k}"] = r[k] - anchor_r[k]
        r["delta_auc_1.5_3.0"] = r["auc_1.5_3.0"] - anchor_r["auc_1.5_3.0"]

        if cells_data[label]["n_seeds"] >= 2:
            for t in THRESHOLDS:
                ci = bootstrap_ci_peak(cells_data[label]["per_concept"], anchor_pc, t, n=1000)
                r[f"ci_peak_coh_ge_{t:.2f}"] = ci

    out_json = PLOTS_DIR / "definitive_table.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"saved {out_json}")

    # Markdown table
    md_path = PLOTS_DIR / "definitive_table.md"
    lines = ["## Phase 7 steering case study — definitive results table", "",
             "All cells at matched per-token sparsity (k_pos = 20). Multi-seed",
             "where available; mean-curve method.", "",
             f"### Peak success at coh thresholds (n = number of seeds)",
             "",
             f"| arch | n | unc | ≥1.5 | ≥1.75 | ≥2.0 | ≥2.25 | ≥2.5 | AUC(1.5-3.0) |",
             f"|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        thr_str = " | ".join(f"{r[f'peak_coh_ge_{t:.2f}']:.3f}" for t in THRESHOLDS)
        lines.append(f"| {label} | {r['n_seeds']} | {r['peak_unc']:.3f} | {thr_str} | {r['auc_1.5_3.0']:.3f} |")
    lines += ["", "### Δ vs anchor + 95% bootstrap CI (multi-seed cells only)", "",
              f"| arch | n | Δ unc | Δ ≥1.5 | CI ≥1.5 | Δ ≥1.75 | CI ≥1.75 | Δ ≥2.0 | CI ≥2.0 | Δ AUC(1.5-3.0) |",
              f"|---|---:|---:|---:|---|---:|---|---:|---|---:|"]
    for label, r in results.items():
        if anchor_label == label or r['n_seeds'] < 2: continue
        ci15 = r.get('ci_peak_coh_ge_1.50', (0,0))
        ci175 = r.get('ci_peak_coh_ge_1.75', (0,0))
        ci20 = r.get('ci_peak_coh_ge_2.00', (0,0))
        lines.append(f"| {label} | {r['n_seeds']} | {r.get('delta_unc', 0):+.3f} | "
                     f"{r.get('delta_peak_coh_ge_1.50', 0):+.3f} | [{ci15[0]:+.3f}, {ci15[1]:+.3f}] | "
                     f"{r.get('delta_peak_coh_ge_1.75', 0):+.3f} | [{ci175[0]:+.3f}, {ci175[1]:+.3f}] | "
                     f"{r.get('delta_peak_coh_ge_2.00', 0):+.3f} | [{ci20[0]:+.3f}, {ci20[1]:+.3f}] | "
                     f"{r.get('delta_auc_1.5_3.0', 0):+.3f} |")
    md_path.write_text("\n".join(lines))
    print(f"saved {md_path}")

    # Print summary to stdout
    print()
    multi_seed = [(l, r) for l, r in results.items() if r['n_seeds'] >= 2 and l != anchor_label]
    for thr in THRESHOLDS:
        print(f"\n=== TOP CELLS by coh ≥ {thr} (multi-seed only, proper bootstrap) ===")
        sorted_cells = sorted(multi_seed, key=lambda x: x[1].get(f'delta_peak_coh_ge_{thr:.2f}', 0), reverse=True)
        for label, r in sorted_cells[:6]:
            ci = r.get(f'ci_peak_coh_ge_{thr:.2f}', (0,0))
            sig = "YES" if ci[0] > 0 else "no"
            print(f"  {label:30s} n={r['n_seeds']}  Δ={r.get(f'delta_peak_coh_ge_{thr:.2f}', 0):+.3f}  CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  sig={sig}")

if __name__ == "__main__":
    main()
