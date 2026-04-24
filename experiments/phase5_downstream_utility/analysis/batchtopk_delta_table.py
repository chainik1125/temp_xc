"""Build full 21-arch TopK-vs-BatchTopK Δ table from probing_results.jsonl.

For each BatchTopK arch, find the TopK counterpart (same name minus
`_batchtopk`), join on (task, aggregation, k_feat=5, seed=42), compute
per-task Δ AUC, aggregate to mean ± std.

Writes `results/batchtopk_delta_table.json` (for plotting) and emits a
Markdown table for summary.md copy/paste.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
OUT_JSON = REPO / "experiments/phase5_downstream_utility/results/batchtopk_delta_table.json"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
SEED = 42
K_FEAT = 5


def load_auc() -> dict:
    """Returns {(arch, agg): {task: auc}}."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("run_id", "")
            if not rid.endswith(f"__seed{SEED}"):
                continue
            if r.get("k_feat") != K_FEAT:
                continue
            agg = r.get("aggregation")
            if agg not in ("last_position", "mean_pool"):
                continue
            arch = rid.rsplit(f"__seed{SEED}", 1)[0]
            auc = r.get("test_auc")
            if auc is None:
                continue
            v = float(auc)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[(arch, agg)][r["task_name"]] = v
    return out


def build_table(data: dict) -> list[dict]:
    rows = []
    bt_archs = sorted({a for a, _ in data if a.endswith("_batchtopk")})
    for bt in bt_archs:
        base = bt.removesuffix("_batchtopk")
        r: dict = {"arch": base, "batchtopk_arch": bt}
        for agg in ("last_position", "mean_pool"):
            bt_tasks = data.get((bt, agg), {})
            tk_tasks = data.get((base, agg), {})
            if not bt_tasks or not tk_tasks:
                r[f"{agg}_topk"] = None
                r[f"{agg}_batchtopk"] = None
                r[f"{agg}_delta"] = None
                r[f"{agg}_n"] = 0
                continue
            common = sorted(set(bt_tasks) & set(tk_tasks))
            if not common:
                r[f"{agg}_topk"] = None
                r[f"{agg}_batchtopk"] = None
                r[f"{agg}_delta"] = None
                r[f"{agg}_n"] = 0
                continue
            tk_mean = st.mean(tk_tasks[t] for t in common)
            bt_mean = st.mean(bt_tasks[t] for t in common)
            r[f"{agg}_topk"] = round(tk_mean, 4)
            r[f"{agg}_batchtopk"] = round(bt_mean, 4)
            r[f"{agg}_delta"] = round(bt_mean - tk_mean, 4)
            r[f"{agg}_n"] = len(common)
        rows.append(r)
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| arch | lp_TopK | lp_BatchTopK | Δ_lp | mp_TopK | mp_BatchTopK | Δ_mp | n_lp | n_mp |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def fmt(x, p=4):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) else "-"
        lines.append(
            f"| {r['arch']} "
            f"| {fmt(r['last_position_topk'])} "
            f"| {fmt(r['last_position_batchtopk'])} "
            f"| {fmt(r['last_position_delta'])} "
            f"| {fmt(r['mean_pool_topk'])} "
            f"| {fmt(r['mean_pool_batchtopk'])} "
            f"| {fmt(r['mean_pool_delta'])} "
            f"| {r['last_position_n']} "
            f"| {r['mean_pool_n']} |"
        )
    return "\n".join(lines)


def main():
    data = load_auc()
    rows = build_table(data)
    summary = {
        "rows": rows,
        "by_agg_summary": {},
    }
    for agg in ("last_position", "mean_pool"):
        deltas = [r[f"{agg}_delta"] for r in rows if r[f"{agg}_delta"] is not None]
        if deltas:
            summary["by_agg_summary"][agg] = {
                "n_archs": len(deltas),
                "mean_delta": round(st.mean(deltas), 4),
                "median_delta": round(st.median(deltas), 4),
                "n_positive": sum(1 for d in deltas if d > 0),
                "n_negative": sum(1 for d in deltas if d < 0),
                "n_small": sum(1 for d in deltas if abs(d) < 0.01),  # "within noise"
            }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")
    print("\nSummary:")
    for agg, s in summary["by_agg_summary"].items():
        print(f"  {agg}: {s}")
    print("\n" + to_markdown(rows))


if __name__ == "__main__":
    main()
