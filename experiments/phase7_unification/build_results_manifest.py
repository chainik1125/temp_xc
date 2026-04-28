"""Build the Phase 7 results manifest — source of truth for what's been
trained and probed across all phases (5, 5B, 7), both subject models
(gemma-2-2b base and -it), and both layers (L12, L13).

The manifest is a single JSON keyed by `(subject_model, arch_id, seed,
k_win)` with per-cell coverage of:
  - which training ckpt exists, and where (local path / HF repo)
  - which probing aggregations exist, and how many tasks each covers

The leaderboard artefact and the T-sweep artefact are filtered subsets
of this manifest.

Re-runnable. Reads:
  - experiments/phase{5,5b,7}_*/results/{training_index,probing_results}.jsonl
  - HF repo file lists (han1823123123/{txcdr,txcdr-base})
Writes:
  - experiments/phase7_unification/results/results_manifest.json

Usage:
  .venv/bin/python -m experiments.phase7_unification.build_results_manifest
"""
from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/phase7_unification/results/results_manifest.json"


# ────────────────────────────── arch_id renaming
# Phase 5/5B used different names for some archs that Phase 7 renamed in
# canonical_archs.json. We rewrite to Phase 7's canonical names ONLY where
# the architectural shape matches (T_max, t_sample, k_win), so that
# variants like subseq_h8 with t_sample=8 don't get incorrectly aliased
# to phase5b_subseq_h8 (which is t_sample=5).
def alias_to_canonical(phase: str, arch: str, row: dict) -> str:
    """Return the Phase 7 canonical arch_id for a row from another phase."""
    if phase == "phase5b_t_scaling_explore":
        T_max = row.get("T_max")
        t_sample = row.get("t_sample")
        k_win = row.get("k_win")
        # Canonical row 13: phase5b_subseq_h8 = subseq_h8 (T_max=10, t_sample=5, k_win=500)
        if (arch == "subseq_h8" and T_max == 10
                and t_sample == 5 and k_win == 500):
            return "phase5b_subseq_h8"
        # Canonical row 12: phase5b_subseq_track2 = subseq_track2 (T_max=10, t_sample=5, k_win=500)
        if (arch == "subseq_track2" and T_max == 10
                and t_sample == 5 and k_win == 500):
            return "phase5b_subseq_track2"
    # No rename — keep the original name (may differ by k_win etc)
    return arch

# ────────────────────────────── data sources by phase

PHASE_SPECS = [
    {
        "phase": "phase7_unification",
        "subject_model": "google/gemma-2-2b",
        "anchor_layer": 12,
        "training_index": REPO / "experiments/phase7_unification/results/training_index.jsonl",
        "probing": REPO / "experiments/phase7_unification/results/probing_results.jsonl",
        "training_logs_dir": REPO / "experiments/phase7_unification/results/training_logs",
        "ckpt_repo": "han1823123123/txcdr-base",
        "ckpt_repo_prefix": "ckpts/",
        # row schemas
        "ti_arch_key": "arch_id",
        "ti_kwin_inference": None,  # k_win is explicit
        "probe_arch_key": "arch_id",
        "probe_seed_explicit": True,
        "probe_S_default": None,   # S is explicit
        "probe_agg_key": None,     # implicitly mean_pool with first_real masking
        "probe_agg_default": "mean_pool_S",
        # methodology — see "methodology classification" below
        "methodology_class": "current",
        "methodology_tag": "phase7_S32_first_real_meanpool",
    },
    {
        "phase": "phase5_downstream_utility",
        "subject_model": "google/gemma-2-2b-it",
        "anchor_layer": 13,
        "training_index": REPO / "experiments/phase5_downstream_utility/results/training_index.jsonl",
        "probing": REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl",
        "training_logs_dir": None,
        "ckpt_repo": "han1823123123/txcdr",
        "ckpt_repo_prefix": "ckpts/",
        "ti_arch_key": "arch",
        # Phase 5 didn't distinguish k_win from k_pos — k_pos field IS the
        # TopK k passed to the model. MLC's shared TopK is also just k.
        "ti_kwin_inference": "kpos_is_kwin",
        "probe_arch_key": "arch",
        "probe_seed_explicit": False,  # extracted from run_id
        "probe_S_default": 20,         # Phase 5 default tail = 20
        "probe_agg_key": "aggregation",
        "probe_agg_default": None,
        "methodology_class": "old",
        "methodology_tag": "phase5_S20_pre_first_real",
    },
    {
        "phase": "phase5b_t_scaling_explore",
        "subject_model": "google/gemma-2-2b-it",
        "anchor_layer": 13,
        "training_index": REPO / "experiments/phase5b_t_scaling_explore/results/training_index.jsonl",
        "probing": REPO / "experiments/phase5b_t_scaling_explore/results/probing_results.jsonl",
        "training_logs_dir": None,
        "ckpt_repo": "han1823123123/txcdr",
        "ckpt_repo_prefix": "phase5b_ckpts/",
        "ti_arch_key": "arch",
        "ti_kwin_inference": None,
        "probe_arch_key": "arch",
        "probe_seed_explicit": True,
        "probe_S_default": 20,
        "probe_agg_key": "aggregation",
        "probe_agg_default": None,
        "methodology_class": "old",
        "methodology_tag": "phase5b_S20_pre_first_real",
    },
]

# methodology classification:
# - "current" = Phase 7's S=32 mean-pool with per-example first_real
#   masking, FLIP convention on cross-token tasks. The methodology used
#   for the canonical Phase 7 leaderboard.
# - "old" = Phase 5/5B's S=20 with last_position / mean_pool / full_window
#   aggregations, no first_real masking, FLIP convention applied at
#   analysis time. Useful as historical context but NOT comparable to
#   "current" methodology numerically.


def _seed_from_run_id(run_id: str) -> int | None:
    """Phase 5 run_id format: '<arch>__seed<n>' or 'BASELINE_*' or anonymous."""
    if "__seed" in run_id:
        try:
            return int(run_id.split("__seed")[-1])
        except ValueError:
            return None
    return None


def _infer_kwin(row: dict, mode: str | None) -> int | None:
    """k_win = the integer passed as TopK k to the SAE constructor.

    Phase 7 records k_win explicitly. Phase 5 records it as `k_pos`
    (Phase 5 didn't yet distinguish window vs position — the model's
    actual TopK budget IS k_pos). MLC has a single shared TopK across
    its L-layer-window encoder, so k_win = k (the value actually
    passed to MultiLayerCrosscoder.__init__) — NOT k_pos × n_layers.
    Same for TXCDR/TXC variants in Phase 5.
    """
    if row.get("k_win") is not None:
        return row["k_win"]
    if mode == "kpos_is_kwin":
        kp = row.get("k_pos")
        if kp is not None:
            return int(kp)
    return None


def _normalize_agg(spec: dict, row: dict) -> str:
    """Return a canonical short tag for the (S, aggregation) combination."""
    agg_key = spec["probe_agg_key"]
    if agg_key:
        agg = row.get(agg_key)
        if agg in ("last_position", "last_position_val"):
            return f"last_position_S{spec['probe_S_default']}"
        if agg == "mean_pool":
            return f"mean_pool_S{spec['probe_S_default']}"
        if agg == "full_window":
            return f"full_window_S{spec['probe_S_default']}"
        return f"{agg}_S{spec['probe_S_default']}"
    # Phase 7 default
    S = row.get("S")
    return f"mean_pool_S{S}"


def _seed_from_row(spec: dict, row: dict) -> int | None:
    if spec["probe_seed_explicit"]:
        return row.get("seed")
    return _seed_from_run_id(row.get("run_id", ""))


def _list_hf_ckpts(repo: str) -> set[str]:
    """Returns set of file basenames (e.g. 'topk_sae__seed42.pt') present
    under any prefix in the repo. Prefix is encoded into the lookup."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo)
        return set(files)
    except Exception as e:
        print(f"  [warn] HF list failed for {repo}: {e}")
        return set()


def build() -> dict:
    # ── HF ckpt inventory ─────────────────────────────────────────────
    print("Listing HF ckpt repos...")
    hf_files: dict[str, set[str]] = {}
    for spec in PHASE_SPECS:
        repo = spec["ckpt_repo"]
        if repo not in hf_files:
            hf_files[repo] = _list_hf_ckpts(repo)
            print(f"  {repo}: {len(hf_files[repo])} files")

    # ── per-cell aggregation ──────────────────────────────────────────
    # cells[(subject_model, arch, seed, k_win)] = record
    cells: dict[tuple, dict] = {}

    for spec in PHASE_SPECS:
        phase = spec["phase"]
        sm = spec["subject_model"]
        layer = spec["anchor_layer"]
        ckpt_repo = spec["ckpt_repo"]
        ckpt_prefix = spec["ckpt_repo_prefix"]

        # training-index entries
        ti_path = spec["training_index"]
        if not ti_path.exists():
            print(f"[warn] missing TI {ti_path}")
            continue
        ti_rows = [json.loads(l) for l in ti_path.open()]

        for row in ti_rows:
            arch_raw = row.get(spec["ti_arch_key"])
            seed = row.get("seed")
            if arch_raw is None or seed is None:
                continue
            kwin = _infer_kwin(row, spec["ti_kwin_inference"])
            kpos = row.get("k_pos")
            arch = alias_to_canonical(phase, arch_raw, row)

            key = (sm, arch, int(seed), kwin)
            run_id = row.get("run_id", f"{arch}__seed{seed}")
            ckpt_name = f"{run_id}.pt"
            ckpt_hf_relpath = f"{ckpt_prefix}{ckpt_name}"
            ckpt_hf_present = ckpt_hf_relpath in hf_files.get(ckpt_repo, set())

            log_path = (spec["training_logs_dir"] / f"{run_id}.json"
                        if spec["training_logs_dir"] else None)
            log_local_present = log_path.exists() if log_path else False

            entry = cells.setdefault(key, {
                "arch_id": arch,
                "seed": int(seed),
                "subject_model": sm,
                "anchor_layer": layer,
                "k_win": kwin,
                "k_pos": kpos,
                "T": row.get("T"),
                "T_max": row.get("T_max"),
                "t_sample": row.get("t_sample"),
                "training": [],
                "probing": defaultdict(lambda: defaultdict(set)),  # tag -> k_feat -> set(tasks)
            })
            entry["training"].append({
                "phase": phase,
                "run_id": run_id,
                "ckpt_hf_repo": ckpt_repo,
                "ckpt_hf_relpath": ckpt_hf_relpath,
                "ckpt_hf_present": ckpt_hf_present,
                "training_log": (str(log_path.relative_to(REPO))
                                 if log_local_present else None),
                "final_step": row.get("final_step"),
                "converged": row.get("converged"),
                "elapsed_s": row.get("elapsed_s"),
                "layer_field_in_ti": row.get("layer") or row.get("layers"),
            })

        # probing rows
        pp_path = spec["probing"]
        if not pp_path.exists():
            print(f"[warn] missing probing {pp_path}")
            continue
        pp_rows = [json.loads(l) for l in pp_path.open()]
        n_skipped_no_seed = 0
        n_skipped_no_match = 0
        for row in pp_rows:
            arch_raw = row.get(spec["probe_arch_key"])
            if arch_raw is None:
                continue
            arch = alias_to_canonical(phase, arch_raw, row)
            seed = _seed_from_row(spec, row)
            if seed is None:
                n_skipped_no_seed += 1
                continue
            kwin = row.get("k_win")
            if kwin is None:
                # try training-index lookup for this (arch, seed) on this phase
                # fallback: assume the same kwin as the training_index entry above
                # if kwin remained None at training, leave it None — match anything
                pass
            agg_tag = _normalize_agg(spec, row)
            kf = row.get("k_feat")

            # match key — k_win can be None on either side; do flexible match
            key_candidates = [(sm, arch, int(seed), kwin)]
            if kwin is None:
                # try the same arch+seed at any k_win
                key_candidates = [k for k in cells if k[0] == sm and k[1] == arch and k[2] == int(seed)]
            matched = [k for k in key_candidates if k in cells]
            if not matched:
                # cell with no training-index entry — create a probe-only cell
                key = (sm, arch, int(seed), kwin)
                matched = [key]
                cells.setdefault(key, {
                    "arch_id": arch,
                    "seed": int(seed),
                    "subject_model": sm,
                    "anchor_layer": layer,
                    "k_win": kwin,
                    "k_pos": row.get("k_pos"),
                    "T": row.get("T"),
                    "T_max": row.get("T_max"),
                    "t_sample": row.get("t_sample"),
                    "training": [],
                    "probing": defaultdict(lambda: defaultdict(set)),
                })

            for k in matched:
                # tag is now a tuple (methodology_class, methodology_tag, agg_tag)
                # so we keep current/old separated even if agg names happen to collide
                full_tag = (spec["methodology_class"],
                            spec["methodology_tag"],
                            agg_tag)
                cells[k]["probing"][full_tag][kf if kf is not None else "no_kfeat"].add(row["task_name"])

        if n_skipped_no_seed:
            print(f"  [{phase}] {n_skipped_no_seed} probing rows skipped (no seed in run_id)")

    # ── format for output ─────────────────────────────────────────────
    out_models = defaultdict(list)
    for key, c in cells.items():
        sm = c["subject_model"]
        # collapse defaultdicts → list of probing-aggregation entries,
        # each with explicit methodology fields
        probing_out = []
        for (mclass, mtag, agg_tag), by_k in c["probing"].items():
            probing_out.append({
                "methodology_class": mclass,        # "current" or "old"
                "methodology_tag": mtag,            # phase7_S32_first_real_meanpool, etc.
                "aggregation": agg_tag,             # mean_pool_S32, last_position_S20, ...
                "by_k_feat": {
                    str(kf): {"n_tasks": len(tasks), "tasks": sorted(tasks)}
                    for kf, tasks in by_k.items()
                },
            })
        # sort: current methodology first, then by aggregation name
        probing_out.sort(key=lambda p: (p["methodology_class"] != "current",
                                        p["aggregation"]))
        c_out = {
            "arch_id": c["arch_id"],
            "seed": c["seed"],
            "k_win": c["k_win"],
            "k_pos": c["k_pos"],
            "T": c["T"],
            "T_max": c["T_max"],
            "t_sample": c["t_sample"],
            "training": c["training"],
            "probing": probing_out,
        }
        out_models[sm].append(c_out)

    # sort cells inside each model
    for sm in out_models:
        out_models[sm].sort(key=lambda c: (c["arch_id"], c["seed"]))

    manifest = {
        "version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "description": "Phase 7 results manifest — source of truth for what's "
                       "been trained and probed across phases 5, 5B, 7. "
                       "Each cell is keyed by (subject_model, arch_id, seed, "
                       "k_win). The leaderboard and T-sweep artefacts are "
                       "filtered subsets of this manifest.",
        "subject_models": {
            sm: {
                "anchor_layer": next((s["anchor_layer"] for s in PHASE_SPECS
                                       if s["subject_model"] == sm), None),
                "phase_origins": sorted({s["phase"] for s in PHASE_SPECS
                                          if s["subject_model"] == sm}),
                "n_cells": len(cells_list),
                "cells": cells_list,
            }
            for sm, cells_list in sorted(out_models.items())
        },
    }
    return manifest


def main() -> None:
    manifest = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT}")
    # summary
    for sm, sm_data in manifest["subject_models"].items():
        print(f"\n{sm} (L{sm_data['anchor_layer']}): {sm_data['n_cells']} cells")
        # break down by seed availability
        archs = {}
        for c in sm_data["cells"]:
            archs.setdefault(c["arch_id"], []).append(c["seed"])
        # how many archs have N seeds with ckpts?
        with_ckpt = {}
        for c in sm_data["cells"]:
            has_ckpt = any(t.get("ckpt_hf_present") for t in c["training"])
            if has_ckpt:
                with_ckpt.setdefault(c["arch_id"], set()).add(c["seed"])
        print(f"  unique archs: {len(archs)}")
        from collections import Counter
        seed_counts = Counter(len(set(s)) for s in archs.values())
        print(f"  seed coverage in TI:  {dict(seed_counts)}")
        ckpt_counts = Counter(len(s) for s in with_ckpt.values())
        print(f"  seed coverage on HF:  {dict(ckpt_counts)}")


if __name__ == "__main__":
    main()
