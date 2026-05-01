"""Analyze K=2 multi-seed Lever B results vs K=1 baseline.

Run after grades are in for steering_paper_window_perposition_seed{1,2}_topk2.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")


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


def peak15(curve, thr):
    eligible = [v[0] for v in curve.values() if v[1] >= thr]
    return max(eligible) if eligible else 0.0


def auc(curve, lo=1.5, hi=3.0):
    if not curve: return 0.0
    pts = sorted(curve.values(), key=lambda v: v[1])
    succs = np.array([p[0] for p in pts])
    cohs = np.array([p[1] for p in pts])
    grid = np.linspace(lo, hi, 41)
    return float(np.trapezoid(np.interp(grid, cohs, succs), grid) / (hi - lo))


def main():
    # K=2 multi-seed
    k2_seeds = [
        ("sd=42", "steering_paper_window_perposition_topk2"),
        ("sd=1", "steering_paper_window_perposition_seed1_topk2"),
        ("sd=2", "steering_paper_window_perposition_seed2_topk2"),
    ]
    k2_curves = []
    for label, sub in k2_seeds:
        c = load_curve(sub, "txc_h8_t2_kpos20_shifts2")
        if c is None:
            print(f"K=2 {label}: missing — wait for grades")
            continue
        print(f"K=2 {label}: {len(c)} strengths")
        k2_curves.append(c)

    # K=1 multi-seed (baseline)
    k1_seeds = [
        ("sd=42", "steering_paper_window_perposition"),
        ("sd=1", "steering_paper_window_perposition_seed1"),
        ("sd=2", "steering_paper_window_perposition_seed2"),
    ]
    k1_curves = []
    for label, sub in k1_seeds:
        c = load_curve(sub, "txc_h8_t2_kpos20_shifts2")
        if c is None: continue
        k1_curves.append(c)

    # Anchor (T-SAE k=20)
    anchor = load_curve("steering_paper_normalised", "tsae_paper_k20")

    n_k2 = len(k2_curves)
    if n_k2 == 0:
        print("\nNo K=2 data yet"); return

    print(f"\n=== K=2 multi-seed ({n_k2} seeds) vs K=1 multi-seed ({len(k1_curves)} seeds) vs anchor ===")
    print(f"{'metric':18s} {'K=1 mean':>10s} {'K=2 mean':>10s} {'anchor':>8s} {'K=2 Δ vs K=1':>14s} {'K=2 Δ vs anc':>14s}")
    print('-' * 90)
    k2_mc = mean_curve(k2_curves)
    k1_mc = mean_curve(k1_curves)
    metrics = [
        ("unconstrained", lambda c: max(v[0] for v in c.values()) if c else 0),
        ("peak ≥ 1.5", lambda c: peak15(c, 1.5)),
        ("peak ≥ 1.75", lambda c: peak15(c, 1.75)),
        ("peak ≥ 2.0", lambda c: peak15(c, 2.0)),
        ("AUC(1.5-3.0)", lambda c: auc(c, 1.5, 3.0)),
        ("AUC(1.75-3.0)", lambda c: auc(c, 1.75, 3.0)),
    ]
    for name, fn in metrics:
        k1 = fn(k1_mc)
        k2 = fn(k2_mc)
        a = fn(anchor)
        print(f'{name:18s} {k1:>10.3f} {k2:>10.3f} {a:>8.3f} {k2-k1:>+14.3f} {k2-a:>+14.3f}')

    print(f"\n=== K=2 per-strength multi-seed mean-curve ({n_k2} seeds) ===")
    print(f'  {"s_norm":>8s}  {"succ":>6s}  {"coh":>6s}')
    for s, (succ, coh) in sorted(k2_mc.items()):
        print(f'  {s:>8.2f}  {succ:>6.3f}  {coh:>6.3f}')

    # Strongest claim
    print()
    if n_k2 >= 2:
        unc_k2 = max(v[0] for v in k2_mc.values())
        anchor_unc = max(v[0] for v in anchor.values())
        print(f"K=2 multi-seed unconstrained peak: {unc_k2:.3f}")
        print(f"T-SAE k=20 unconstrained peak: {anchor_unc:.3f}")
        if unc_k2 >= anchor_unc:
            print(f"\n🚀 GIGABRAIN MOMENT: K=2 multi-seed BEATS T-SAE on unconstrained peak by Δ=+{unc_k2-anchor_unc:.3f}!")
        elif unc_k2 + 0.05 >= anchor_unc:
            print(f"\n🎯 K=2 multi-seed CLOSE to T-SAE unc peak (gap = {anchor_unc-unc_k2:.3f}); within noise")
        else:
            print(f"\n📉 K=2 multi-seed unc peak still below T-SAE by {anchor_unc-unc_k2:.3f}")


if __name__ == "__main__":
    main()
