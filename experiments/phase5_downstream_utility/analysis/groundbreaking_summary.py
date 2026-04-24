"""End-of-queue summary for the groundbreaking-results push.

Aggregates:
  1. H8 T-sweep monotonicity + Δ(T_max - T_min) at lp + mp.
  2. Shift-ablation curve at T=5: AUC vs shifts-set.
  3. MLC + anti-dead vs TXC + anti-dead deltas.
  4. H3 log-matryoshka T-sweep (if trained).

Reads probing_results.jsonl (filtered by k_feat=5, seed=42 unless noted).
Prints a single big table per section.
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

K_FEAT = 5


def _mean_auc_per_arch_seed(seed: int = 42, agg: str = "last_position",
                             k_feat: int = K_FEAT) -> dict[str, float]:
    """Return {arch: mean_test_auc_over_tasks} for one seed/agg/k_feat."""
    by_arch = defaultdict(list)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != k_feat:
                continue
            rid = r.get("run_id", "")
            if not rid.endswith(f"__seed{seed}"):
                continue
            arch = rid.rsplit(f"__seed{seed}", 1)[0]
            v = r.get("test_auc")
            if v is None:
                continue
            v = float(v)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            by_arch[arch].append(v)
    return {arch: st.mean(vs) for arch, vs in by_arch.items() if len(vs) >= 30}


def section_h8_tsweep():
    """H8 T-sweep: train at T=5, 10, 15, 20, 30. Monotonicity + Δ(T-T_min)."""
    print("\n" + "=" * 80)
    print("§1  H8 T-SWEEP: phase57_partB_h8_bare_multidistance × T")
    print("=" * 80)
    archs_for_T = {
        5: "phase57_partB_h8_bare_multidistance",
        10: "phase57_partB_h8_bare_multidistance_t10",
        15: "phase57_partB_h8_bare_multidistance_t15",
        20: "phase57_partB_h8_bare_multidistance_t20",
        30: "phase57_partB_h8_bare_multidistance_t30",
    }
    for agg in ("last_position", "mean_pool"):
        means = _mean_auc_per_arch_seed(agg=agg)
        present_T = [T for T, a in archs_for_T.items() if a in means]
        if not present_T:
            print(f"  {agg}: no H8 T-sweep data yet.")
            continue
        per_T = {T: means[archs_for_T[T]] for T in present_T}
        pairs = list(combinations(sorted(present_T), 2))
        mono = (sum(1 for a, b in pairs if per_T[b] >= per_T[a]) / len(pairs)
                if pairs else float("nan"))
        delta = per_T[max(present_T)] - per_T[min(present_T)] if len(present_T) >= 2 else float("nan")
        print(f"  {agg}:")
        for T in sorted(present_T):
            print(f"    T={T:2d}: AUC={per_T[T]:.4f}")
        print(f"    monotonicity={mono:.3f} (target ≥ 0.80)")
        print(f"    Δ(T_max−T_min)={delta:+.4f} (target > +0.020)")
        if 5 in present_T and 30 in present_T:
            print(f"    Δ(T=30−T=5)={per_T[30] - per_T[5]:+.4f}")


def section_shift_ablation():
    """Shift-ablation at T=5: how does AUC depend on shifts-set."""
    print("\n" + "=" * 80)
    print("§2  H8 SHIFT-ABLATION at T=5: AUC vs shifts-set")
    print("=" * 80)
    archs = {
        "{1}": "phase57_partB_h8a_shifts1",
        "{1,2} = H8": "phase57_partB_h8_bare_multidistance",
        "{1,2,3}": "phase57_partB_h8a_shifts123",
        "{1,2,3,4}": "phase57_partB_h8a_shifts1234",
        "{1,2,4}": "phase57_partB_h8a_shifts124",
        "{2}": "phase57_partB_h8a_shifts2",
        "{4}": "phase57_partB_h8a_shifts4",
        "{1,2,3} uniform": "phase57_partB_h8a_shifts123_uniform",
    }
    for agg in ("last_position", "mean_pool"):
        means = _mean_auc_per_arch_seed(agg=agg)
        print(f"\n  {agg}:")
        print(f"    {'label':<20s} {'arch':<55s} {'AUC':>7s}")
        for label, arch in archs.items():
            auc = means.get(arch)
            s = f"{auc:.4f}" if auc is not None else "—"
            print(f"    {label:<20s} {arch:<55s} {s:>7s}")


def section_mlc_antidead():
    """MLC + anti-dead vs TXC + anti-dead — the fairness check."""
    print("\n" + "=" * 80)
    print("§3  MLC + ANTI-DEAD vs TXC + ANTI-DEAD (fairness)")
    print("=" * 80)
    pairs = [
        ("recon-only",
         "txc_bare_antidead (NOT TRAINED in 5.7)",  # TXC equivalent of mlc_bare_antidead — never trained
         "mlc_bare_antidead"),
        ("matryoshka + 1-shift contr",
         "phase62_c3 (Phase 6.2 0.7834 lp)",  # external reference
         "mlc_bare_matryoshka_contrastive_antidead"),
        ("matryoshka + multi-scale contr (= H7)",
         "phase57_partB_h7_bare_multiscale",
         "mlc_bare_multiscale_antidead"),
    ]
    for agg in ("last_position", "mean_pool"):
        means = _mean_auc_per_arch_seed(agg=agg)
        print(f"\n  {agg}:")
        print(f"    {'recipe':<35s} {'TXC AUC':>9s}  {'MLC AUC':>9s}  Δ")
        for label, txc_arch, mlc_arch in pairs:
            txc = means.get(txc_arch.split()[0]) if txc_arch and " " not in txc_arch else None
            # Some entries are external refs — pull from constant
            if "0.7834" in txc_arch:
                txc = 0.7834 if agg == "last_position" else None
            mlc = means.get(mlc_arch)
            if txc is not None and mlc is not None:
                delta = mlc - txc
                print(f"    {label:<35s} {txc:>9.4f}  {mlc:>9.4f}  {delta:+.4f}")
            else:
                t = f"{txc:.4f}" if txc is not None else "—"
                m = f"{mlc:.4f}" if mlc is not None else "—"
                print(f"    {label:<35s} {t:>9s}  {m:>9s}  —")


def section_log_matryoshka_tsweep():
    print("\n" + "=" * 80)
    print("§4  H3 LOG-MATRYOSHKA T-SWEEP")
    print("=" * 80)
    archs = {T: f"log_matryoshka_t{T}" for T in (5, 10, 15, 20, 30)}
    for agg in ("last_position", "mean_pool"):
        means = _mean_auc_per_arch_seed(agg=agg)
        present = [T for T, a in archs.items() if a in means]
        if not present:
            print(f"  {agg}: no log-matryoshka data yet.")
            continue
        per_T = {T: means[archs[T]] for T in present}
        pairs = list(combinations(sorted(present), 2))
        mono = (sum(1 for a, b in pairs if per_T[b] >= per_T[a]) / len(pairs)
                if pairs else float("nan"))
        delta = per_T[max(present)] - per_T[min(present)] if len(present) >= 2 else float("nan")
        print(f"  {agg}:")
        for T in sorted(present):
            print(f"    T={T:2d}: AUC={per_T[T]:.4f}")
        print(f"    monotonicity={mono:.3f} (target ≥ 0.80)")
        print(f"    Δ(T_max−T_min)={delta:+.4f} (target > +0.020)")


def section_top10_overall():
    print("\n" + "=" * 80)
    print("§5  TOP 10 ARCHS OVERALL (seed 42)")
    print("=" * 80)
    for agg in ("last_position", "mean_pool"):
        means = _mean_auc_per_arch_seed(agg=agg)
        sorted_archs = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"\n  {agg}:")
        for i, (arch, auc) in enumerate(sorted_archs, 1):
            print(f"    {i:2d}. {arch:<55s} {auc:.4f}")


def main():
    section_h8_tsweep()
    section_shift_ablation()
    section_mlc_antidead()
    section_log_matryoshka_tsweep()
    section_top10_overall()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
