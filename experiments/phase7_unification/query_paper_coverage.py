"""Derived view over (paper_archs.json + results_manifest.json + probing_results.jsonl)
that prints the canonical paper coverage:

  - leaderboard cells: which (subject_model, arch_id, k_win, seed) are trained,
    probed under current methodology, with mean AUC at k_feat=5/20.
  - barebones T-sweep cells: same, per T.
  - hill-climbed T-sweep cells: same, per T.

This is a LENS over the canonical files, not a new source of truth. The
canonical files are:

  experiments/phase7_unification/paper_archs.json        (spec)
  experiments/phase7_unification/results/results_manifest.json
                                                         (coverage)
  experiments/phase7_unification/results/probing_results.jsonl
                                                         (raw AUCs)
  experiments/phase5_downstream_utility/results/probing_results.jsonl
                                                         (legacy raw, old methodology)

Re-run any time after a probing pass to refresh the view. Does not
write any file by default; pass --json <path> to dump the structured
result for downstream consumers.

Usage:
  .venv/bin/python -m experiments.phase7_unification.query_paper_coverage
  .venv/bin/python -m experiments.phase7_unification.query_paper_coverage --json /tmp/paper_view.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER_SPEC = REPO / "experiments/phase7_unification/paper_archs.json"
MANIFEST   = REPO / "experiments/phase7_unification/results/results_manifest.json"
P7_PROBE   = REPO / "experiments/phase7_unification/results/probing_results.jsonl"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}


def load_p7_aucs() -> dict:
    """Index Phase 7 probing rows by (arch_id, seed, k_win, k_feat) → list of (task, auc)."""
    idx = defaultdict(list)
    with P7_PROBE.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("S") != 32: continue
            if r.get("skipped"): continue
            auc = r.get("test_auc_flip") if r["task_name"] in FLIP_TASKS else r["test_auc"]
            key = (r["arch_id"], r["seed"], r.get("k_win"), r["k_feat"])
            idx[key].append((r["task_name"], auc))
    return idx


def auc_summary(rows: list[tuple[str, float]]) -> dict | None:
    if not rows: return None
    aucs = [a for _, a in rows]
    return {"n_tasks": len(rows), "mean_auc": round(statistics.mean(aucs), 4),
            "min_auc": round(min(aucs), 4), "max_auc": round(max(aucs), 4)}


def status_for_cell(p7_aucs: dict, manifest_cells: list,
                    arch_id: str, k_win: int) -> dict:
    """For a (arch_id, k_win) cell across both subject models + all seeds,
    return a structured status row."""
    out = {"arch_id": arch_id, "k_win": k_win, "by_subject_model": {}}
    by_sm = defaultdict(list)
    for c in manifest_cells:
        if c["_subject_model"] != "google/gemma-2-2b" and c["_subject_model"] != "google/gemma-2-2b-it":
            continue
        if c["arch_id"] == arch_id and c["k_win"] == k_win:
            by_sm[c["_subject_model"]].append(c)
    for sm, cells in by_sm.items():
        seeds = sorted({c["seed"] for c in cells})
        ckpt_seeds = sorted({c["seed"] for c in cells
                             if any(t.get("ckpt_hf_present") for t in c["training"])})
        cur_seeds = sorted({c["seed"] for c in cells
                            if any(p["methodology_class"] == "current" for p in c["probing"])})
        old_seeds = sorted({c["seed"] for c in cells
                            if any(p["methodology_class"] == "old" for p in c["probing"])})
        # mean AUCs (only Phase 7 base side has AUCs in p7_aucs)
        aucs_by_seed = {}
        for s in cur_seeds:
            aucs_by_seed[s] = {
                "k=5":  auc_summary(p7_aucs.get((arch_id, s, k_win, 5),  [])),
                "k=20": auc_summary(p7_aucs.get((arch_id, s, k_win, 20), [])),
            }
        out["by_subject_model"][sm] = {
            "seeds_with_ckpt": ckpt_seeds,
            "seeds_probed_current": cur_seeds,
            "seeds_probed_old":     old_seeds,
            "auc_by_seed": aucs_by_seed,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="optional: write structured output JSON", default=None)
    args = ap.parse_args()

    spec = json.loads(PAPER_SPEC.read_text())
    manifest = json.loads(MANIFEST.read_text())
    p7_aucs = load_p7_aucs()

    # Flatten manifest cells with subject_model attached for easy filtering
    cells = []
    for sm, sm_data in manifest["subject_models"].items():
        for c in sm_data["cells"]:
            c2 = dict(c); c2["_subject_model"] = sm
            cells.append(c2)

    out = {"version": spec["version"], "paper_archs": [],
           "tsweep_barebones": [], "tsweep_hillclimbed": []}

    # leaderboard
    for a in spec["leaderboard_archs"]:
        out["paper_archs"].append({
            "paper_id": a["paper_id"],
            **status_for_cell(p7_aucs, cells, a["arch_id"], a["k_win"]),
        })

    # barebones T-sweep
    for T in spec["tsweep_barebones"]["T_values"]:
        arch_id = spec["tsweep_barebones"]["arch_template"].replace("<T>", str(T))
        out["tsweep_barebones"].append({
            "T": T,
            **status_for_cell(p7_aucs, cells, arch_id, spec["tsweep_barebones"]["k_win"]),
        })

    # hill-climbed T-sweep
    if spec.get("tsweep_hillclimbed"):
        for T in spec["tsweep_hillclimbed"]["T_values"]:
            arch_id = spec["tsweep_hillclimbed"]["arch_template"].replace("<T>", str(T))
            out["tsweep_hillclimbed"].append({
                "T": T,
                **status_for_cell(p7_aucs, cells, arch_id, spec["tsweep_hillclimbed"]["k_win"]),
            })

    # human-readable print
    SMS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
    def fmt_seeds(meta, k):
        if not meta: return f'{"":>10}'
        a = meta.get(k)
        if not a: return f'{"":>10}'
        return f'{a["mean_auc"]:.4f}'

    def fmt_status(seeds_dict):
        cur = seeds_dict.get("seeds_probed_current", [])
        old = seeds_dict.get("seeds_probed_old", [])
        c = seeds_dict.get("seeds_with_ckpt", [])
        return f'ckpt={c}  cur={cur}  old={old}'

    print("=" * 110)
    print("LEADERBOARD CELLS")
    print("=" * 110)
    print(f'{"paper_id":<22} {"arch_id":<42} {"k_win":>5}  base sd42 sd1  IT methodology')
    for entry in out["paper_archs"]:
        base = entry["by_subject_model"].get(SMS[0], {})
        it   = entry["by_subject_model"].get(SMS[1], {})
        b42_aucs = base.get("auc_by_seed", {}).get(42, {})
        b1_aucs  = base.get("auc_by_seed", {}).get(1,  {})
        it_meth  = "current" if it.get("seeds_probed_current") else ("old" if it.get("seeds_probed_old") else "—")
        it_seeds = (it.get("seeds_with_ckpt") or [])
        bk5_42  = (b42_aucs.get("k=5",  {}) or {}).get("mean_auc", "—")
        bk20_42 = (b42_aucs.get("k=20", {}) or {}).get("mean_auc", "—")
        print(f'{entry["paper_id"]:<22} {entry["arch_id"]:<42} {entry["k_win"]:>5}  '
              f'k5={bk5_42!s:<6}k20={bk20_42!s:<6}  IT:{it_meth}/sd{it_seeds}')

    print()
    print("=" * 110)
    print("BAREBONES T-SWEEP — txcdr_t<T>")
    print("=" * 110)
    print(f'{"T":>3}  {"base k=5 sd42":>14} {"base k=5 sd1":>14}  {"base k=20 sd42":>16} {"base k=20 sd1":>14}  IT')
    for entry in out["tsweep_barebones"]:
        b = entry["by_subject_model"].get(SMS[0], {})
        it = entry["by_subject_model"].get(SMS[1], {})
        b42 = b.get("auc_by_seed", {}).get(42, {})
        b1  = b.get("auc_by_seed", {}).get(1, {})
        b42k5  = (b42.get("k=5",  {}) or {}).get("mean_auc", "—")
        b1k5   = (b1.get("k=5",   {}) or {}).get("mean_auc", "—")
        b42k20 = (b42.get("k=20", {}) or {}).get("mean_auc", "—")
        b1k20  = (b1.get("k=20",  {}) or {}).get("mean_auc", "—")
        it_meth  = "current" if it.get("seeds_probed_current") else ("old" if it.get("seeds_probed_old") else "—")
        it_seeds = it.get("seeds_with_ckpt") or []
        print(f'{entry["T"]:>3}  {b42k5!s:>14} {b1k5!s:>14}  {b42k20!s:>16} {b1k20!s:>14}  {it_meth}/sd{it_seeds}')

    print()
    print("=" * 110)
    print("HILL-CLIMBED T-SWEEP — phase57_partB_h8_bare_multidistance_t<T>")
    print("=" * 110)
    feas = spec["tsweep_hillclimbed"]["a40_feasibility"]
    def feas_for(T):
        if T in feas["comfortable_at_b4096"]: return "ok"
        if T in feas["tight_at_b4096_use_b2048"]: return "b2048"
        if T in feas["needs_streaming_refactor"]: return "STREAM"
        return "?"
    print(f'{"T":>3}  {"A40":>6}  {"base k=5 sd42":>14} {"base k=5 sd1":>14}  {"base k=20 sd42":>16} {"base k=20 sd1":>14}  IT')
    for entry in out["tsweep_hillclimbed"]:
        b = entry["by_subject_model"].get(SMS[0], {})
        it = entry["by_subject_model"].get(SMS[1], {})
        b42 = b.get("auc_by_seed", {}).get(42, {})
        b1  = b.get("auc_by_seed", {}).get(1, {})
        b42k5  = (b42.get("k=5",  {}) or {}).get("mean_auc", "—")
        b1k5   = (b1.get("k=5",   {}) or {}).get("mean_auc", "—")
        b42k20 = (b42.get("k=20", {}) or {}).get("mean_auc", "—")
        b1k20  = (b1.get("k=20",  {}) or {}).get("mean_auc", "—")
        it_meth  = "current" if it.get("seeds_probed_current") else ("old" if it.get("seeds_probed_old") else "—")
        it_seeds = it.get("seeds_with_ckpt") or []
        f = feas_for(entry["T"])
        print(f'{entry["T"]:>3}  {f:>6}  {b42k5!s:>14} {b1k5!s:>14}  {b42k20!s:>16} {b1k20!s:>14}  {it_meth}/sd{it_seeds}')

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f'\nWrote structured JSON to {args.json}')


if __name__ == "__main__":
    main()
