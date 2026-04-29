"""Rank the 36 SAEBench tasks by discriminative power across archs.

Useful tasks have:
1. **High cross-arch range** (max - min AUC) — separates good from bad archs.
2. **High cross-arch SD** — same idea, smoother metric.
3. **Low intra-cluster correlation** — bias_in_bios_set1_prof11 vs prof2 are
   likely measuring the same skill axis, so 15 of them is over-counting.

Output:
- Per-task discrimination stats at k_feat ∈ {5, 20}, mean over the 13 paper
  leaderboard archs (3 seeds averaged where available).
- Recommended reduced set with one task per skill cluster + the highest
  discriminative tasks.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.analyze_task_importance
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR


PROBING_PATH = OUT_DIR / "probing_results.jsonl"
LEADERBOARD_ARCHS = [
    "topk_sae", "tsae_paper_k20", "tsae_paper_k500", "tfa_big",
    "mlc", "agentic_mlc_08", "mlc_contrastive_alpha100_batchtopk",
    "txcdr_t5", "txcdr_t16", "phase5b_subseq_h8",
    "txc_bare_antidead_t5", "phase57_partB_h8_bare_multidistance_t8",
    "hill_subseq_h8_T12_s5",
]
SEEDS = (1, 2, 42)


def load_per_arch_per_task() -> dict:
    """Returns dict[k_feat -> dict[task -> dict[arch -> mean_auc_across_seeds]]]."""
    by_seed = defaultdict(dict)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != 32 or r.get("seed") not in SEEDS: continue
            if r.get("k_feat") not in (5, 20): continue
            if "skipped" in r: continue
            if r.get("arch_id") not in LEADERBOARD_ARCHS: continue
            key = (r["k_feat"], r["task_name"], r["arch_id"], r["seed"])
            by_seed[key] = r.get("test_auc_flip", r["test_auc"])
    out = defaultdict(lambda: defaultdict(dict))
    for (kf, task, arch, seed), auc in by_seed.items():
        out[kf].setdefault(task, {}).setdefault(arch, []).append(auc)
    flat = {}
    for kf, tasks in out.items():
        flat[kf] = {}
        for task, archs in tasks.items():
            flat[kf][task] = {a: np.mean(v) for a, v in archs.items()}
    return flat


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


def main():
    data = load_per_arch_per_task()
    print("=" * 110)
    print("Per-task discriminative power across leaderboard archs (3-seed mean)")
    print("=" * 110)
    rows = []
    for kf in (5, 20):
        for task, arch_means in data[kf].items():
            vals = list(arch_means.values())
            if len(vals) < 5: continue
            rng = max(vals) - min(vals)
            sd  = float(np.std(vals, ddof=1))
            mean = float(np.mean(vals))
            rows.append((kf, task, mean, sd, rng, len(vals)))
    rows_k5  = sorted([r for r in rows if r[0] == 5],  key=lambda r: -r[3])
    rows_k20 = sorted([r for r in rows if r[0] == 20], key=lambda r: -r[3])

    for label, rs in [("k_feat=5", rows_k5), ("k_feat=20", rows_k20)]:
        print(f"\n{label}: tasks ranked by cross-arch SD (most discriminative first)")
        print(f"  {'task':40s}  {'mean':>7s}  {'SD':>7s}  {'range':>7s}  cluster")
        for kf, t, m, sd, rng, n in rs:
            print(f"  {t:40s}  {m:>7.4f}  {sd:>7.4f}  {rng:>7.4f}  {task_cluster(t)}")

    # Per-cluster summary
    print("\n" + "=" * 110)
    print("Per-cluster summary — mean SD and mean range across tasks in cluster")
    print("=" * 110)
    for kf in (5, 20):
        print(f"\n{f'k_feat={kf}'}")
        clusters = defaultdict(list)
        for r in rows:
            if r[0] != kf: continue
            clusters[task_cluster(r[1])].append(r)
        print(f"  {'cluster':18s}  {'n_tasks':>7s}  {'mean_SD':>9s}  {'mean_range':>11s}  {'mean_AUC':>9s}")
        crows = []
        for c, ts in clusters.items():
            mean_sd = float(np.mean([r[3] for r in ts]))
            mean_rng = float(np.mean([r[4] for r in ts]))
            mean_auc = float(np.mean([r[2] for r in ts]))
            crows.append((c, len(ts), mean_sd, mean_rng, mean_auc))
        for c, n, msd, mrng, mauc in sorted(crows, key=lambda r: -r[2]):
            print(f"  {c:18s}  {n:>7d}  {msd:>9.4f}  {mrng:>11.4f}  {mauc:>9.4f}")

    # Within-cluster correlation (do bias_in_bios prof11 and prof2 measure the same thing?)
    print("\n" + "=" * 110)
    print("Within-cluster mean pairwise correlation (high = redundant tasks)")
    print("=" * 110)
    for kf in (5, 20):
        print(f"\nk_feat={kf}")
        clusters = defaultdict(list)
        for r in rows:
            if r[0] != kf: continue
            clusters[task_cluster(r[1])].append(r[1])
        for c, tasks in sorted(clusters.items()):
            if len(tasks) < 2: continue
            # build per-task vectors of arch-AUCs
            vecs = []
            archs_present = sorted(LEADERBOARD_ARCHS)
            for t in tasks:
                v = []
                for a in archs_present:
                    if a in data[kf].get(t, {}):
                        v.append(data[kf][t][a])
                    else:
                        v.append(np.nan)
                vecs.append(v)
            arr = np.array(vecs, dtype=float)
            # pairwise correlation, ignoring nans pairwise
            corrs = []
            for i in range(len(tasks)):
                for j in range(i+1, len(tasks)):
                    a = arr[i]; b = arr[j]
                    mask = ~(np.isnan(a) | np.isnan(b))
                    if mask.sum() < 3: continue
                    c_ij = np.corrcoef(a[mask], b[mask])[0, 1]
                    corrs.append(c_ij)
            if corrs:
                print(f"  {c:18s}  n={len(tasks):>2d}  mean_pairwise_r={np.mean(corrs):>+.3f}  median={np.median(corrs):>+.3f}")


if __name__ == "__main__":
    main()
