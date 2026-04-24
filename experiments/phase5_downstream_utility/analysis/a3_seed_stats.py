"""A3 stats: per-task paired-t-test + Bonferroni correction across 29-arch bench.

Handover A3: "After, update the Δ columns in summary.md Figure 1/2 tables
with σ or with 95% CI, and add paired-t-test p-values with Bonferroni
correction across the 29-arch bench."

For each pair (A, baseline), we compute a per-task Δ_i = auc_A[task_i] −
auc_baseline[task_i], then a paired-t-test on those 36 per-task deltas.
Bonferroni correction = multiply p-values by n_comparisons.

Reads probing_results.jsonl, writes `results/a3_seed_stats.json`.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
OUT_JSON = REPO / "experiments/phase5_downstream_utility/results/a3_seed_stats.json"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
K_FEAT = 5


def load() -> dict:
    """Returns {(arch, seed, agg): {task: auc}}."""
    out: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("k_feat") != K_FEAT:
                continue
            rid = r.get("run_id", "")
            if "__seed" not in rid:
                continue
            arch, seed_str = rid.rsplit("__seed", 1)
            if not seed_str.isdigit():
                continue
            seed = int(seed_str)
            agg = r.get("aggregation")
            if agg not in ("last_position", "mean_pool"):
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            v = float(auc)
            if r["task_name"] in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[(arch, seed, agg)][r["task_name"]] = v
    return dict(out)


def paired_t(arch_A: str, arch_B: str, agg: str, data: dict,
             seeds: list[int] = None) -> dict | None:
    """Paired-t-test of (A - B) on per-task AUCs, averaged across seeds.

    If seeds is not None, average per-task AUCs across the given seeds
    for each arch before differencing. Otherwise use seed=42 alone.
    """
    if seeds is None:
        A_tasks = data.get((arch_A, 42, agg), {})
        B_tasks = data.get((arch_B, 42, agg), {})
    else:
        # Average per-task AUCs across seeds
        A_per_task = defaultdict(list)
        B_per_task = defaultdict(list)
        for s in seeds:
            for t, v in data.get((arch_A, s, agg), {}).items():
                A_per_task[t].append(v)
            for t, v in data.get((arch_B, s, agg), {}).items():
                B_per_task[t].append(v)
        A_tasks = {t: st.mean(vs) for t, vs in A_per_task.items() if vs}
        B_tasks = {t: st.mean(vs) for t, vs in B_per_task.items() if vs}
    common = sorted(set(A_tasks) & set(B_tasks))
    if len(common) < 10:
        return None
    diffs = np.array([A_tasks[t] - B_tasks[t] for t in common])
    t_stat, p_val = stats.ttest_rel(
        [A_tasks[t] for t in common],
        [B_tasks[t] for t in common],
    )
    return {
        "n_tasks": len(common),
        "mean_delta": float(diffs.mean()),
        "std_delta": float(diffs.std()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "wins_A": int((diffs > 0).sum()),
        "wins_B": int((diffs < 0).sum()),
    }


def main():
    data = load()

    # Focus: top agentic winners vs their non-agentic baselines + paired
    # TopK-vs-BatchTopK at matched arch.
    comparisons = [
        # Agentic vs baseline (handover's headline claim)
        ("agentic_mlc_08", "mlc"),
        ("agentic_mlc_08", "mlc_contrastive"),
        ("agentic_txc_02", "txcdr_t5"),
        ("agentic_txc_02", "matryoshka_t5"),
        # BatchTopK winners vs their TopK twin
        ("mlc_contrastive_alpha100_batchtopk", "mlc_contrastive_alpha100"),
        ("agentic_mlc_08_batchtopk", "agentic_mlc_08"),
        # Part-B α=1.0 vs default α=0.1
        ("mlc_contrastive_alpha100", "mlc_contrastive"),
        # Best-overall: top of lp table vs #2
        ("mlc_contrastive_alpha100_batchtopk", "agentic_mlc_08"),
    ]

    # For each comparison: seed=42 single-seed AND 3-seed average (where available)
    results = {}
    for A, B in comparisons:
        for agg in ("last_position", "mean_pool"):
            key = f"{A} vs {B} | {agg}"
            r42 = paired_t(A, B, agg, data)
            r3 = paired_t(A, B, agg, data, seeds=[1, 2, 42])
            entry = {"seed42": r42, "3seed_avg": r3}
            results[key] = entry

    # Bonferroni correction: multiply p_value by number of comparisons
    n_comp = len(comparisons) * 2  # × 2 aggregations
    for key, entry in results.items():
        for cond in ("seed42", "3seed_avg"):
            r = entry.get(cond)
            if r and r.get("p_value") is not None:
                r["p_bonferroni"] = min(1.0, r["p_value"] * n_comp)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")
    print("\n=== Paired-t summary (Bonferroni-corrected, n_comp={}) ===".format(n_comp))
    for key, entry in results.items():
        for cond, label in (("seed42", "s42"), ("3seed_avg", "3s")):
            r = entry.get(cond)
            if r is None:
                continue
            sig = "***" if r.get("p_bonferroni", 1) < 0.001 else "**" if r.get("p_bonferroni", 1) < 0.01 else "*" if r.get("p_bonferroni", 1) < 0.05 else " "
            print(f"{sig} {key} [{label}]: Δ={r['mean_delta']:+.4f} "
                  f"p_raw={r['p_value']:.4f} p_bonf={r.get('p_bonferroni',1):.4f} "
                  f"wins={r['wins_A']}/{r['wins_B']}")


if __name__ == "__main__":
    main()
