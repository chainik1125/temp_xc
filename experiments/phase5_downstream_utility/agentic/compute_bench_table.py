"""Compute apples-to-apples bench tables from probing_results.jsonl.

For each aggregation × arch, uses LAST-WRITE-WINS per (arch, task) to
get the most-recent probe value (probing is non-deterministic so older
rows are stale). Computes mean AUC across 36 tasks. Prints markdown
table sorted by AUC desc.

Called after test-set eval + baseline re-probe chain completes.
"""

import json
from collections import defaultdict
from pathlib import Path
import statistics

JSONL = Path("experiments/phase5_downstream_utility/results/probing_results.jsonl")

BASELINES = {"baseline_attn_pool", "baseline_last_token_lr"}
AGENTIC_WINNERS = {"agentic_txc_02", "agentic_mlc_08"}
PART_B_WINNERS = {
    "matryoshka_txcdr_contrastive_t5_alpha100",
    "mlc_contrastive_alpha100",
}

def load_last_write(agg, k_feat=5, seed=42):
    """Return {arch: {task: auc}} using last-write-wins per (arch, task)."""
    # rows in file order; later rows overwrite earlier
    by_rid_task_auc = {}
    by_rid_task_acc = {}
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != k_feat:
                continue
            rid = r["run_id"]
            if rid.startswith("baseline"):
                # baseline rows keyed differently; synthesize
                pass
            # For seeded rids, enforce seed match via the __seedN suffix
            if rid.endswith(f"__seed{seed}") or rid.startswith("baseline"):
                by_rid_task_auc[(rid, r["task_name"])] = r.get("test_auc")
                by_rid_task_acc[(rid, r["task_name"])] = r.get("test_acc")
    # pivot
    per_arch_auc = defaultdict(dict)
    per_arch_acc = defaultdict(dict)
    for (rid, task), auc in by_rid_task_auc.items():
        if auc is None:
            continue
        # Strip seed suffix to get "arch" name; baselines stay as-is
        if rid.startswith("baseline"):
            arch = rid
        elif f"__seed{seed}" in rid:
            arch = rid.rsplit(f"__seed{seed}", 1)[0]
        else:
            continue
        per_arch_auc[arch][task] = auc
        per_arch_acc[arch][task] = by_rid_task_acc.get((rid, task), 0.0)
    return per_arch_auc, per_arch_acc


def table(agg, k_feat=5, seed=42):
    per_auc, per_acc = load_last_write(agg, k_feat, seed)
    rows = []
    for arch, tasks in per_auc.items():
        if len(tasks) < 36:
            # Skip archs without full 36-task coverage
            continue
        aucs = list(tasks.values())
        mean_auc = statistics.mean(aucs)
        std_auc = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        accs = list(per_acc[arch].values())
        mean_acc = statistics.mean(accs)
        rows.append((arch, mean_auc, std_auc, mean_acc, len(aucs)))
    rows.sort(key=lambda r: -r[1])
    return rows


def mark(arch):
    if arch in BASELINES: return "**%s**" % arch
    if arch in AGENTIC_WINNERS: return "🆕 **%s**" % arch
    if arch in PART_B_WINNERS: return "🆕 %s" % arch
    return arch


def print_table(agg, seed=42):
    rows = table(agg, seed=seed)
    print(f"\n### {agg} × AUC × full (k=5, seed={seed})\n")
    print("| arch | mean AUC | std | n |")
    print("|---|---|---|---|")
    for arch, mean_auc, std_auc, mean_acc, n in rows:
        print(f"| {mark(arch)} | {mean_auc:.4f} | {std_auc:.4f} | {n} |")


def three_seed(agg, arch):
    """Report per-seed AUC + 3-seed mean/std for an agentic winner."""
    import statistics
    per_seed = {}
    for seed in [42, 1, 2]:
        per_auc, _ = load_last_write(agg, seed=seed)
        tasks = per_auc.get(arch, {})
        if len(tasks) == 36:
            per_seed[seed] = statistics.mean(tasks.values())
    if per_seed:
        aucs = list(per_seed.values())
        print(f"\n{arch} @ {agg} across seeds:")
        for s, v in per_seed.items():
            print(f"  seed={s}: {v:.4f}")
        if len(aucs) > 1:
            print(f"  mean ± σ = {statistics.mean(aucs):.4f} ± {statistics.stdev(aucs):.4f}")


if __name__ == "__main__":
    for agg in ["last_position", "mean_pool"]:
        print_table(agg)
    print("\n\n=== 3-seed variance on agentic winners ===")
    for agg in ["last_position", "mean_pool"]:
        for arch in ["agentic_txc_02", "agentic_mlc_08"]:
            three_seed(agg, arch)
