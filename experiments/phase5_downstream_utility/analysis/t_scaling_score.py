"""T-scaling score: (monotonicity, Δ(T_max − T_min)) per arch.

Used in Part B to evaluate whether a candidate TXC arch exhibits
AUC-scales-with-T under the fixed probing protocol.

Metrics:
- monotonicity_score: fraction of T-pairs (i<j) where auc(T_j) ≥ auc(T_i).
  Range [0, 1]; random ~0.5; strict monotone = 1.0.
- delta_maxmin: auc(T_max) − auc(T_min) (absolute AUC gain).

Targets from handover:
- monotonicity_score ≥ 0.8
- delta(T=30, T=5) > 0.02

Usage:
  from t_scaling_score import score_arch_family
  score_arch_family("txcdr_t", T_values=[5,10,15,20,30], seed=42)
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}


def _load_auc_by_T(family_prefix: str, T_values: list[int], seed: int = 42,
                   aggregation: str = "last_position", k_feat: int = 5,
                   sparsity_suffix: str = "") -> dict[int, dict[str, float]]:
    """Return {T: {task: auc}} for archs matching {family_prefix}{T}{suffix}.

    Example: family_prefix='txcdr_t', sparsity_suffix='' → txcdr_t5, txcdr_t10...
             family_prefix='txcdr_t', sparsity_suffix='_batchtopk' → txcdr_t5_batchtopk...
    """
    out: dict[int, dict[str, float]] = defaultdict(dict)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != aggregation or r.get("k_feat") != k_feat:
                continue
            rid = r.get("run_id", "")
            if not rid.endswith(f"__seed{seed}"):
                continue
            arch = rid.rsplit(f"__seed{seed}", 1)[0]
            if not arch.startswith(family_prefix) or not arch.endswith(sparsity_suffix):
                continue
            middle = arch[len(family_prefix):len(arch) - len(sparsity_suffix)
                          if sparsity_suffix else None]
            if not middle.isdigit():
                continue
            T = int(middle)
            if T not in T_values:
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            v = float(auc)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[T][r["task_name"]] = v
    return dict(out)


def score_arch_family(family_prefix: str, T_values: list[int], seed: int = 42,
                      aggregation: str = "last_position", k_feat: int = 5,
                      sparsity_suffix: str = "",
                      min_task_coverage: int = 30) -> dict:
    """Compute monotonicity + delta for an arch family T-sweep.

    Returns:
      {
        'T_values': [5, 10, 15, 20, 30],
        'per_T_mean_auc': [0.7752, ...],
        'monotonicity_score': 0.7,
        'delta_maxmin': 0.012,
        'delta_30m5': 0.008 (or None if T=5 or T=30 missing),
        'coverage': {T: n_tasks, ...},
      }
    """
    data = _load_auc_by_T(family_prefix, T_values, seed, aggregation,
                          k_feat, sparsity_suffix)
    covered_T = [T for T in T_values if len(data.get(T, {})) >= min_task_coverage]
    mean_auc = {T: st.mean(data[T].values()) for T in covered_T}

    # Monotonicity: for all T-pairs (i<j) in covered_T, count auc(T_j) >= auc(T_i).
    pairs = list(combinations(covered_T, 2))
    if not pairs:
        mono = float("nan")
    else:
        gains = sum(1 for a, b in pairs if mean_auc[b] >= mean_auc[a])
        mono = gains / len(pairs)

    delta_maxmin = (mean_auc[max(covered_T)] - mean_auc[min(covered_T)]
                    if len(covered_T) >= 2 else float("nan"))
    delta_30m5 = None
    if 30 in mean_auc and 5 in mean_auc:
        delta_30m5 = mean_auc[30] - mean_auc[5]

    return {
        "family": f"{family_prefix}<T>{sparsity_suffix}",
        "aggregation": aggregation,
        "seed": seed,
        "T_values": covered_T,
        "per_T_mean_auc": [round(mean_auc[T], 4) for T in covered_T],
        "monotonicity_score": round(mono, 4) if mono == mono else None,
        "delta_maxmin": round(delta_maxmin, 4) if delta_maxmin == delta_maxmin else None,
        "delta_30m5": round(delta_30m5, 4) if delta_30m5 is not None else None,
        "coverage": {T: len(data.get(T, {})) for T in T_values},
    }


def main():
    families = [
        ("txcdr_t", ""),
        ("txcdr_t", "_batchtopk"),
        ("agentic_txc_02_t", "_batchtopk"),
    ]
    T_values = [2, 3, 5, 8, 10, 15, 20, 24, 28, 32, 36]
    print("T-scaling score report")
    print("=" * 80)
    for prefix, suffix in families:
        for agg in ["last_position", "mean_pool"]:
            r = score_arch_family(prefix, T_values, aggregation=agg,
                                  sparsity_suffix=suffix)
            print(f"\nfamily: {r['family']}  |  agg: {agg}")
            print(f"  T values covered: {r['T_values']}")
            print(f"  per-T mean AUC:   {r['per_T_mean_auc']}")
            print(f"  monotonicity:     {r['monotonicity_score']}"
                  f" (target ≥ 0.8)")
            print(f"  Δ(T_max − T_min): {r['delta_maxmin']}")
            print(f"  Δ(T=30 − T=5):    {r['delta_30m5']} (target > 0.02)")


if __name__ == "__main__":
    main()
