"""Compute net_saves + 2x2 contingency at peak Δgc magnitude per arch.

Reads the per-cell artifacts produced by ``eval_optimal_mag.py``:

    <workspace>/steered_phase2_optimal.jsonl  — per-row text + parsed answer
    <workspace>/judge_outputs.jsonl           — backtracking COUNT label
    <workspace>/coherence_judge.jsonl         — coherence 0-3 grade

For each (arch, bs) cell that has these artifacts, computes:

  * ``net_saves`` (baseline-corrected): # rescues − # regressions, where
      rescue     := unsteered_wrong AND steered_correct
      regression := unsteered_correct AND steered_wrong
    Computed at the optimal mag and at mag=0; reports both the raw
    Δnet_at_peak and the baseline-corrected Δnet_corr =
      (rescues_peak − rescues_0) − (regressions_peak − regressions_0).

  * 2x2 contingency at the optimal mag:
      coherent_AND_backtrack   |  coherent_AND_no_backtrack
      incoherent_AND_backtrack |  incoherent_AND_no_backtrack
    where coherent := (coherence_grade >= 2) and
          backtrack := (genuine_count >= 1).

Writes a markdown table (one row per cell) to the supplied output
path, ready to drop into the paper bundle.

Usage::

    .venv/bin/python -m experiments.c7_backtracking.analyze_optimal \\
        --output /workspace/aniket/temp_xc_paper/purified/docs/components/c7_optimal_mag_results.md
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from temp_bench.config import (
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    load_arch,
    load_datasource,
    purified_root,
    run_dir,
)
from temp_bench.schemas import TrainingConfig

from experiments.c7_backtracking.run import COMPONENT, DATASOURCE, EVAL_PROTOCOL_VERSION

log = logging.getLogger("c7.analyze_optimal")

# (arch, bs, n_steps, seed) cells we expect to find.
DEFAULT_CELLS: list[tuple[str, int, int, int]] = [
    ("txc_base",   256,  300_000, 42),
    ("txc_base",   1024, 300_000, 42),
    ("txc_pro",    256,  300_000, 42),
    ("txc_pro",    1024, 300_000, 42),
    ("topk_sae",   1024, 300_000, 42),
    ("tsae_paper", 1024, 300_000, 42),
    ("mlc",        1024, 300_000, 42),
]

PAPER_ARCH_LABEL = {
    "topk_sae":   "TopK SAE",
    "tsae_paper": "T-SAE",
    "mlc":        "MLC",
    "txc_base":   "TXC-base",
    "txc_pro":    "TXC-pro",
}


def _cell_workspace(arch: str, bs: int, n_steps: int, seed: int
                    ) -> tuple[Path, float | None]:
    """Resolve the optimal-mag eval workspace by re-deriving its eval_key.

    Returns (workspace_path, peak_mag) or (None, None) if no canonical
    leaderboard row exists yet (cell still training).
    """
    spec = load_arch(arch, component=COMPONENT)
    ds = load_datasource(DATASOURCE)
    act_cache_key = compute_act_cache_key(ds)
    training_cfg = TrainingConfig(n_steps=n_steps, batch_size=bs)
    train_key = compute_train_key(
        arch=spec, seed=seed, training_cfg=training_cfg,
        act_cache_key=act_cache_key,
    )
    # Find the canonical leaderboard row to read the peak magnitude.
    lb = purified_root() / "results" / "leaderboard.jsonl"
    peak_mag: float | None = None
    with lb.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("component") == COMPONENT
                    and r.get("train_key") == train_key
                    and not r.get("eval_cfg", {}).get("_extended_mags")):
                peak_mag = float(r["metrics"]["delta_gc_peak_magnitude"])
                break
    if peak_mag is None:
        return Path("/dev/null"), None
    eval_cfg = {
        "magnitudes": [0.0, peak_mag],
        "cut_fraction": 0.25,
        "_optimal_mag": True,
        "_peak_mag": peak_mag,
    }
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=eval_cfg,
    )
    return run_dir(eval_key), peak_mag


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _index_by_qid_mag(rows: list[dict], grade_field: str) -> dict[tuple[str, float], int]:
    """Return {(qid, mag): grade} for the latest row per (qid, mag).
    Skips rows where ``grade_field`` is missing or < 0 (parse failures)."""
    by_key: dict[tuple[str, float], dict] = {}
    for r in rows:
        key = (r.get("transcript_id") or r.get("unique_id"),
               float(r["magnitude"]))
        if key[0] is None:
            continue
        if r.get(grade_field, -1) < 0:
            continue
        cur = by_key.get(key)
        if cur is None or r.get("ts", "") > cur.get("ts", ""):
            by_key[key] = r
    return {k: int(v[grade_field]) for k, v in by_key.items()}


def analyze_one_cell(arch: str, bs: int, n_steps: int, seed: int
                     ) -> dict[str, Any] | None:
    """Compute net_saves + 2x2 contingency for one cell. Returns None
    if any artifact is missing (e.g. cell still in flight)."""
    workspace, peak_mag = _cell_workspace(arch, bs, n_steps, seed)
    if peak_mag is None:
        log.warning("[%s bs=%d] no canonical leaderboard row yet — skip",
                    arch, bs)
        return None
    steered = _load_jsonl(workspace / "steered_phase2_optimal.jsonl")
    bt_rows = _load_jsonl(workspace / "judge_outputs.jsonl")
    coh_rows = _load_jsonl(workspace / "coherence_judge.jsonl")
    if not steered or not bt_rows or not coh_rows:
        log.warning(
            "[%s bs=%d] missing artefacts (steered=%d, bt=%d, coh=%d) at %s — skip",
            arch, bs, len(steered), len(bt_rows), len(coh_rows), workspace,
        )
        return None

    # Index judge labels by (qid, mag) for fast lookup.
    bt_by_key = _index_by_qid_mag(bt_rows, "label")
    coh_by_key = _index_by_qid_mag(coh_rows, "grade")

    # Group steered rows by magnitude.
    by_mag: dict[float, list[dict]] = defaultdict(list)
    for r in steered:
        by_mag[float(r["magnitude"])].append(r)

    # Rescue / regression counts per magnitude.
    counts_per_mag: dict[float, dict[str, int]] = {}
    for mag, mrows in by_mag.items():
        rescues = 0; regressions = 0
        for r in mrows:
            uns = bool(r["unsteered_correct"])
            stg = bool(r["steered_correct"])
            if (not uns) and stg: rescues += 1
            if uns and (not stg): regressions += 1
        counts_per_mag[mag] = {
            "rescues": rescues,
            "regressions": regressions,
            "n": len(mrows),
        }

    # 2x2 contingency at the OPTIMAL mag (peak_mag).
    contingency = {
        "coh_bt":   0,  # coherent + backtrack
        "coh_nobt": 0,  # coherent + no-backtrack
        "inc_bt":   0,  # incoherent + backtrack
        "inc_nobt": 0,  # incoherent + no-backtrack
        "missing":  0,  # missing one/both labels
        "n_total":  0,
    }
    for r in by_mag.get(peak_mag, []):
        contingency["n_total"] += 1
        bt = bt_by_key.get((r["unique_id"], peak_mag))
        coh = coh_by_key.get((r["unique_id"], peak_mag))
        if bt is None or coh is None:
            contingency["missing"] += 1
            continue
        is_coh = coh >= 2
        is_bt = bt >= 1
        if is_coh and is_bt: contingency["coh_bt"] += 1
        elif is_coh and not is_bt: contingency["coh_nobt"] += 1
        elif not is_coh and is_bt: contingency["inc_bt"] += 1
        else: contingency["inc_nobt"] += 1

    peak_counts = counts_per_mag.get(peak_mag, {})
    base_counts = counts_per_mag.get(0.0, {})
    delta_net_at_peak = (
        peak_counts.get("rescues", 0) - peak_counts.get("regressions", 0)
    )
    delta_net_at_base = (
        base_counts.get("rescues", 0) - base_counts.get("regressions", 0)
    )
    delta_net_corr = delta_net_at_peak - delta_net_at_base

    return {
        "arch": arch,
        "bs": bs,
        "n_steps": n_steps,
        "seed": seed,
        "peak_mag": peak_mag,
        "rescues_peak": peak_counts.get("rescues", 0),
        "regressions_peak": peak_counts.get("regressions", 0),
        "rescues_base": base_counts.get("rescues", 0),
        "regressions_base": base_counts.get("regressions", 0),
        "n_eval_peak": peak_counts.get("n", 0),
        "delta_net_at_peak": delta_net_at_peak,
        "delta_net_at_base": delta_net_at_base,
        "delta_net_corr": delta_net_corr,
        **{k: contingency[k] for k in
           ("coh_bt", "coh_nobt", "inc_bt", "inc_nobt", "missing", "n_total")},
    }


def write_markdown(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# C7 — Backtracking case study: optimal-magnitude analysis")
    md.append("")
    md.append("_Generated from each cell's `steered_phase2_optimal.jsonl`,")
    md.append("`judge_outputs.jsonl`, and `coherence_judge.jsonl`. For each")
    md.append("trained cell, evaluation runs at exactly two magnitudes:")
    md.append("`{0, peak Δgc magnitude}`. The mag=0 column is used for")
    md.append("baseline-corrected `Δnet_corr`. Coherence is the 0–3")
    md.append("Sonnet rubric (port of Aniket's wasteland `grade_sonnet.py`)")
    md.append("with `coherent := grade >= 2`. Backtracking is")
    md.append("`Sonnet COUNT >= 1`._")
    md.append("")

    # Net-saves table.
    md.append("## Net saves at optimal magnitude (baseline-corrected)")
    md.append("")
    md.append("`Δnet_corr = (rescues_peak − regressions_peak) − "
              "(rescues_0 − regressions_0)`. Larger is better — positive "
              "means steering rescued more questions than the cut-and-continue "
              "noise floor would by itself.")
    md.append("")
    md.append("| Arch | bs | peak mag | rescues@peak | regr@peak | "
              "rescues@0 | regr@0 | Δnet@peak | Δnet@0 | **Δnet_corr** |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        md.append("| " + " | ".join(str(x) for x in [
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            r["bs"],
            f"{r['peak_mag']:+g}",
            r["rescues_peak"], r["regressions_peak"],
            r["rescues_base"], r["regressions_base"],
            f"{r['delta_net_at_peak']:+d}",
            f"{r['delta_net_at_base']:+d}",
            f"**{r['delta_net_corr']:+d}**",
        ]) + " |")
    md.append("")

    # Contingency table.
    md.append("## 2×2 contingency at optimal magnitude")
    md.append("")
    md.append("Each cell of the cohort (n=61) classified along ")
    md.append("{coherent (grade≥2) vs incoherent} × {backtracking (count≥1) "
              "vs no-backtracking} at the cell's peak Δgc magnitude.")
    md.append("")
    md.append("| Arch | bs | peak mag | coh+bt | coh+no-bt | inc+bt | "
              "inc+no-bt | missing | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        md.append("| " + " | ".join(str(x) for x in [
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            r["bs"],
            f"{r['peak_mag']:+g}",
            r["coh_bt"], r["coh_nobt"], r["inc_bt"],
            r["inc_nobt"], r["missing"], r["n_total"],
        ]) + " |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("_Regenerated from cell-level artifacts every time")
    md.append("`analyze_optimal.py` runs. Source per cell: see the")
    md.append("`workspace` paths logged in the eval_optimal_mag.py output._")

    out_path.write_text("\n".join(md))
    log.info("[c7.analyze_optimal] wrote %s (%d cells)", out_path, len(results))


def main(*, output: Path, cells: list[tuple[str, int, int, int]] | None = None
         ) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    cells = cells or DEFAULT_CELLS
    results: list[dict] = []
    for arch, bs, n_steps, seed in cells:
        r = analyze_one_cell(arch, bs, n_steps, seed)
        if r is not None:
            results.append(r)
    if not results:
        log.warning("[c7.analyze_optimal] no cells with complete artifacts; "
                    "writing empty report")
    write_markdown(results, output)
    return 0


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raise SystemExit(main(output=args.output))


if __name__ == "__main__":
    cli()
