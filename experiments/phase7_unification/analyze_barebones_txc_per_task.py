"""Per-task analysis: where does barebones TXC do best / worst?

"Barebones TXC" here = `txcdr_t<T>` (vanilla TemporalCrosscoder, no
matryoshka, no contrastive, no anti-dead). The headline barebones cell
is `txcdr_t5` (T=5, k_win=500). Compared against:
  - `topk_sae` (vanilla per-token TopK SAE — most apples-to-apples)
  - `tsae_paper_k500` (T-SAE paper baseline at matched k_win)
  - `mlc` (multi-layer crosscoder per-token; the other "structural" arch)

For each of the 36 SAEBench tasks, compute 3-seed mean AUC for each
arch, then per-task Δ = txcdr_t5 − {topk_sae, tsae_paper_k500, mlc}.
Rank tasks by Δ vs topk_sae descending. Report at k_feat ∈ {5, 20}.

Also reports cluster aggregation so we can see whether barebones TXC
favours specific task families.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.analyze_barebones_txc_per_task
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR


PROBING_PATH = OUT_DIR / "probing_results.jsonl"
SEEDS = (1, 2, 42)
TXC_BASE = "txcdr_t5"
SAE_BASE = "topk_sae"
TSAE_BASE = "tsae_paper_k500"
MLC_BASE = "mlc"
ALL_ARCHS = (TXC_BASE, SAE_BASE, TSAE_BASE, MLC_BASE)


def task_cluster(name: str) -> str:
    if name.startswith("bias_in_bios"): return "bias_in_bios"
    if name.startswith("ag_news"):       return "ag_news"
    if name.startswith("amazon_reviews_cat"): return "amazon_cat"
    if "amazon_reviews_sentiment" in name: return "amazon_sentiment"
    if name.startswith("europarl"):     return "europarl"
    if name.startswith("github_code"):  return "github_code"
    if name in ("winogrande_correct_completion", "wsc_coreference"):
        return "coreference"
    return "other"


def load() -> dict:
    """Returns dict[k_feat][arch][task] -> 3-seed mean AUC."""
    by_seed = defaultdict(dict)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != 32 or r.get("seed") not in SEEDS: continue
            if r.get("k_feat") not in (5, 20): continue
            if "skipped" in r: continue
            if r.get("arch_id") not in ALL_ARCHS: continue
            key = (r["k_feat"], r["arch_id"], r["task_name"], r["seed"])
            by_seed[key] = r.get("test_auc_flip", r["test_auc"])
    out = defaultdict(lambda: defaultdict(dict))
    accum = defaultdict(list)
    for (kf, arch, task, seed), auc in by_seed.items():
        accum[(kf, arch, task)].append(auc)
    for (kf, arch, task), v in accum.items():
        out[kf][arch][task] = float(np.mean(v))
    return out


def main():
    data = load()

    for kf in (5, 20):
        print()
        print("=" * 110)
        print(f"BAREBONES TXC ({TXC_BASE}) per-task — k_feat={kf}, 3-seed mean")
        print("=" * 110)
        rows = []
        all_tasks = set(data[kf].get(TXC_BASE, {}).keys())
        for t in all_tasks:
            tx  = data[kf][TXC_BASE].get(t)
            sa  = data[kf][SAE_BASE].get(t)
            ts  = data[kf][TSAE_BASE].get(t)
            ml  = data[kf][MLC_BASE].get(t)
            if any(x is None for x in (tx, sa, ts, ml)): continue
            rows.append({
                "task": t, "cluster": task_cluster(t),
                "txc": tx, "sae": sa, "tsae": ts, "mlc": ml,
                "d_sae":  tx - sa,
                "d_tsae": tx - ts,
                "d_mlc":  tx - ml,
            })

        # Best for TXC: largest Δ vs SAE
        rows_sorted = sorted(rows, key=lambda r: -r["d_sae"])
        print(f"\nTasks where {TXC_BASE} BEATS {SAE_BASE} (top 10):")
        print(f"  {'task':40s} {'cluster':18s}  {'TXC':>6s}  {'SAE':>6s}  {'Δ_sae':>7s}  {'Δ_tsae':>7s}  {'Δ_mlc':>7s}")
        for r in rows_sorted[:10]:
            print(f"  {r['task']:40s} {r['cluster']:18s}  {r['txc']:.4f}  {r['sae']:.4f}  {r['d_sae']:+.4f}  {r['d_tsae']:+.4f}  {r['d_mlc']:+.4f}")

        print(f"\nTasks where {TXC_BASE} LOSES TO {SAE_BASE} (bottom 10):")
        print(f"  {'task':40s} {'cluster':18s}  {'TXC':>6s}  {'SAE':>6s}  {'Δ_sae':>7s}  {'Δ_tsae':>7s}  {'Δ_mlc':>7s}")
        for r in rows_sorted[-10:]:
            print(f"  {r['task']:40s} {r['cluster']:18s}  {r['txc']:.4f}  {r['sae']:.4f}  {r['d_sae']:+.4f}  {r['d_tsae']:+.4f}  {r['d_mlc']:+.4f}")

        # Cluster summary
        print(f"\nPer-cluster mean Δ (txcdr_t5 − baseline):")
        clusters = defaultdict(list)
        for r in rows:
            clusters[r["cluster"]].append(r)
        crows = []
        for c, ts in clusters.items():
            crows.append((
                c, len(ts),
                float(np.mean([r["d_sae"] for r in ts])),
                float(np.mean([r["d_tsae"] for r in ts])),
                float(np.mean([r["d_mlc"] for r in ts])),
                float(np.mean([r["txc"] for r in ts])),
                sum(1 for r in ts if r["d_sae"] > 0.001),
                sum(1 for r in ts if r["d_sae"] < -0.001),
            ))
        crows.sort(key=lambda r: -r[2])
        print(f"  {'cluster':18s} {'n':>3s}  {'Δ vs SAE':>10s}  {'Δ vs TSAE':>11s}  {'Δ vs MLC':>10s}  {'mean_TXC':>9s}  {'wins/loss':>10s}")
        for c, n, ds, dt, dm, tx, w, l in crows:
            print(f"  {c:18s} {n:>3d}  {ds:>+10.4f}  {dt:>+11.4f}  {dm:>+10.4f}  {tx:>9.4f}  {w:>5d}/{l:<3d}")


if __name__ == "__main__":
    main()
