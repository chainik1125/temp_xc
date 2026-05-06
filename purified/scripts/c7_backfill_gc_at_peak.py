"""Backfill gc_at_peak_mag + gc_at_baseline into existing C7 leaderboard rows.

Reads each canonical cell's ``results/runs/<eval_key>/judge_outputs.jsonl``
and re-aggregates the genuine-backtracking counts via
:func:`compute_delta_gc`. Patches:

  - ``results/leaderboard.jsonl``  (the row's metrics)
  - ``results/runs/<eval_key>/metrics.json``

CPU-only, no checkpoint loading needed — finishes in seconds per cell.

Idempotent: cells that already have ``gc_at_peak_mag`` are skipped.

Usage::

    .venv/bin/python -m scripts.c7_backfill_gc_at_peak           # patch all
    .venv/bin/python -m scripts.c7_backfill_gc_at_peak --dry-run # just print
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from temp_bench.case_studies.backtracking import compute_delta_gc
from temp_bench.config import checkpoint_dir, purified_root, run_dir

PURIFIED = purified_root()
LEADERBOARD = PURIFIED / "results" / "leaderboard.jsonl"


def _train_cfg(row: dict) -> dict:
    cfg_path = checkpoint_dir(row["train_key"]) / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text()).get("training_cfg", {})


def _is_canonical(row: dict) -> bool:
    if row.get("component") != "c7":
        return False
    if row.get("eval_cfg", {}).get("_extended_mags"):
        return False
    return _train_cfg(row).get("n_steps") == 300_000


def _has_gc(row: dict) -> bool:
    return "gc_at_peak_mag" in row.get("metrics", {})


def _patch_leaderboard(eval_key: str, new_metrics: dict[str, float]) -> bool:
    if not LEADERBOARD.exists():
        return False
    lines = LEADERBOARD.read_text().splitlines()
    out: list[str] = []
    patched = False
    for line in lines:
        s = line.strip()
        if not s:
            out.append(line)
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if r.get("eval_key") == eval_key:
            r.setdefault("metrics", {}).update(new_metrics)
            out.append(json.dumps(r))
            patched = True
        else:
            out.append(line)
    if patched:
        LEADERBOARD.write_text("\n".join(out) + "\n")
    return patched


def _patch_metrics_json(eval_key: str, new_metrics: dict[str, float]) -> bool:
    p = run_dir(eval_key) / "metrics.json"
    if not p.exists():
        return False
    blob = json.loads(p.read_text())
    metrics = blob.get("metrics", blob)
    metrics.update(new_metrics)
    if metrics is not blob:
        blob["metrics"] = metrics
    p.write_text(json.dumps(blob, indent=2))
    return True


def main(*, dry_run: bool = False) -> None:
    if not LEADERBOARD.exists():
        print(f"no leaderboard at {LEADERBOARD}")
        return
    rows = [
        json.loads(line) for line in LEADERBOARD.read_text().splitlines()
        if line.strip()
    ]
    canonical = [r for r in rows if _is_canonical(r)]
    todo = [r for r in canonical if not _has_gc(r)]
    print(f"[gc-backfill] {len(canonical)} canonical c7 rows, "
          f"{len(canonical) - len(todo)} already have gc_at_peak_mag, "
          f"{len(todo)} to backfill")
    if not todo:
        return

    for r in todo:
        eval_key = r["eval_key"]
        bs = _train_cfg(r).get("batch_size", "?")
        arch_name = r["arch"]
        jpath = run_dir(eval_key) / "judge_outputs.jsonl"
        if not jpath.exists():
            print(f"[gc-backfill] SKIP {arch_name} bs={bs}: no judge_outputs at {jpath}")
            continue
        rows_j = [json.loads(line) for line in jpath.read_text().splitlines() if line.strip()]
        # compute_delta_gc handles label parsing + cohort filtering internally.
        delta = compute_delta_gc(rows_j)
        gc_peak = float(delta["gc_at_peak"].get(arch_name, 0.0))
        gc_base = float(delta["gc_at_baseline"].get(arch_name, 0.0))
        # Sanity check: delta_gc_peak from this re-aggregation should match
        # the leaderboard's stored value within 1e-3.
        peak_mag, peak_delta = delta["peak"].get(arch_name, (None, None))
        stored_peak = r["metrics"].get("delta_gc_peak")
        if peak_delta is not None and stored_peak is not None and abs(peak_delta - stored_peak) > 1e-3:
            print(f"  WARN {arch_name} bs={bs}: stored Δgc_peak={stored_peak:.4f} "
                  f"recomputed={peak_delta:.4f} (drift > 1e-3)")
        new_metrics = {
            "gc_at_peak_mag": gc_peak,
            "gc_at_baseline": gc_base,
        }
        print(f"[gc-backfill] {arch_name} bs={bs}: peak_mag={peak_mag} "
              f"gc_at_peak={gc_peak:.3f}, gc_at_baseline={gc_base:.3f}, "
              f"lift={gc_peak - gc_base:+.3f}")
        if dry_run:
            print("  (dry-run; not patched)")
            continue
        _patch_metrics_json(eval_key, new_metrics)
        _patch_leaderboard(eval_key, new_metrics)
        print("  patched")
    print("[gc-backfill] done")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    cli()
