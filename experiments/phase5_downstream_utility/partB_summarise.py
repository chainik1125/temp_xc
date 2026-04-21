"""Part B analysis: α sensitivity + k-sweep cross-summary.

Reads probing_results.jsonl and produces a clean per-variant table
with:
  - mean val AUC across 36 probing tasks (last_position_val, k=5)
  - Δ vs vanilla base (txcdr_t5 / matryoshka_t5)
  - Δ vs the α=0.1 / k=500 reference (txcdr_contrastive_t5 /
    matryoshka_txcdr_contrastive_t5) — measures "how much did we
    gain by moving off the default"

Writes to partB_summary.json and prints a markdown table that can
be pasted into the results doc.

Usage:
    .venv/bin/python -m experiments.phase5_downstream_utility.partB_summarise \\
        [--k 5] [--metric auc]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path("/workspace/temp_xc")
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS / "probing_results.jsonl"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

# Part B variant map — (arch, family, config_label, hyperparams).
# family groups variants that share a vanilla base (for Δ vs vanilla).
# ref_arch is the α=0.1, k=500 "default" variant we compare against.
FAMILIES = {
    "A2": {
        "vanilla_base": "txcdr_t5",
        "ref": "txcdr_contrastive_t5",
        "variants": [
            # (arch, label, alpha, k_win)
            ("txcdr_contrastive_t5_alpha003", "α=0.03, k=500", 0.03, 500),
            ("txcdr_contrastive_t5", "α=0.10, k=500 (ref)", 0.10, 500),
            ("txcdr_contrastive_t5_alpha100", "α=1.00, k=500", 1.00, 500),
            ("txcdr_contrastive_t5_k2x", "α=0.10, k=1000", 0.10, 1000),
        ],
    },
    "A3": {
        "vanilla_base": "matryoshka_t5",
        "ref": "matryoshka_txcdr_contrastive_t5",
        "variants": [
            ("matryoshka_txcdr_contrastive_t5_alpha003", "α=0.03, k=500", 0.03, 500),
            ("matryoshka_txcdr_contrastive_t5", "α=0.10, k=500 (ref)", 0.10, 500),
            ("matryoshka_txcdr_contrastive_t5_alpha100", "α=1.00, k=500", 1.00, 500),
            ("matryoshka_txcdr_contrastive_t5_k2x", "α=0.10, k=1000", 0.10, 1000),
        ],
    },
}


def _val_per_task(rows, arch, seed, k, metric):
    """Return {task: AUC_or_acc} for last_position_val rows of (arch, seed, k)."""
    out = {}
    rid_target = f"{arch}__seed{seed}"
    key = f"test_{metric}"
    for r in rows:
        if r.get("error") or r.get(key) is None:
            continue
        if r.get("aggregation") != "last_position_val":
            continue
        if r.get("k_feat") != k:
            continue
        if r.get("run_id") != rid_target:
            continue
        task = r.get("task_name")
        v = float(r[key])
        if task in FLIP_TASKS:
            v = max(v, 1.0 - v)
        out[task] = v
    return out


def _paired_stats(cand_pt, base_pt):
    """Return (mean, stderr, t, wins, losses, n) for paired Δ."""
    common = sorted(set(cand_pt) & set(base_pt))
    if not common:
        return None
    cand_vals = np.array([cand_pt[t] for t in common])
    base_vals = np.array([base_pt[t] for t in common])
    d = cand_vals - base_vals
    mean = float(d.mean())
    stderr = float(d.std() / np.sqrt(len(d))) if len(d) > 1 else 0.0
    t = mean / stderr if stderr > 0 else 0.0
    wins = int((d > 0.005).sum())
    losses = int((d < -0.005).sum())
    return {
        "mean": mean, "stderr": stderr, "t": t,
        "wins": wins, "losses": losses, "n": len(common),
    }


def _records():
    rows = []
    with JSONL.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--metric", choices=["auc", "acc"], default="auc")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = _records()

    out = {"k": args.k, "metric": args.metric, "seed": args.seed, "families": {}}

    print()
    print("## Part B results  (last_position_val, k=5, seed=42)")
    print()

    for family_name, family in FAMILIES.items():
        vanilla_pt = _val_per_task(
            rows, family["vanilla_base"], args.seed, args.k, args.metric,
        )
        ref_pt = _val_per_task(
            rows, family["ref"], args.seed, args.k, args.metric,
        )
        print(f"### {family_name} ({family['ref']} family)")
        print()
        print("| variant | mean val AUC | Δ vs vanilla {} | t | wins/losses | Δ vs α=0.1 ref | t | wins/losses |".format(family["vanilla_base"]))
        print("|---|---|---|---|---|---|---|---|")
        family_out = {
            "vanilla_base": family["vanilla_base"],
            "ref": family["ref"],
            "variants": [],
        }
        for arch, label, alpha, k_win in family["variants"]:
            cand_pt = _val_per_task(rows, arch, args.seed, args.k, args.metric)
            if not cand_pt:
                print(f"| **{label}** | NO DATA (arch={arch}) | | | | | | |")
                family_out["variants"].append({
                    "arch": arch, "label": label, "alpha": alpha, "k_win": k_win,
                    "mean": None, "vs_vanilla": None, "vs_ref": None,
                })
                continue
            mean_cand = float(np.mean(list(cand_pt.values())))
            d_van = _paired_stats(cand_pt, vanilla_pt)
            d_ref = _paired_stats(cand_pt, ref_pt)
            van_str = (f"{d_van['mean']:+.4f} ±{d_van['stderr']:.4f}"
                        if d_van else "—")
            ref_str = (f"{d_ref['mean']:+.4f} ±{d_ref['stderr']:.4f}"
                        if d_ref else "—")
            t_van = f"{d_van['t']:+.2f}" if d_van else "—"
            t_ref = f"{d_ref['t']:+.2f}" if d_ref else "—"
            wl_van = f"{d_van['wins']}/{d_van['losses']}" if d_van else "—"
            wl_ref = f"{d_ref['wins']}/{d_ref['losses']}" if d_ref else "—"
            print(f"| {label} | {mean_cand:.4f} | {van_str} | {t_van} | {wl_van} | {ref_str} | {t_ref} | {wl_ref} |")
            family_out["variants"].append({
                "arch": arch, "label": label, "alpha": alpha, "k_win": k_win,
                "mean": mean_cand,
                "vs_vanilla": d_van,
                "vs_ref": d_ref,
            })
        out["families"][family_name] = family_out
        print()

    (RESULTS / "partB_summary.json").write_text(json.dumps(out, indent=2))
    print(f"JSON: {RESULTS / 'partB_summary.json'}")


if __name__ == "__main__":
    main()
