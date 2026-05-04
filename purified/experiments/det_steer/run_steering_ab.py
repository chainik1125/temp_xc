"""V0 / V1 / V2 / V4 steering A/B for one (arch, feature) on the C7 cohort.

This is the **agent-runnable** wrapper that agent_back invokes inside
their existing C7 sweep to produce the steering A/B numbers
documented in ``docs/cross_component/det_steer_steering.md``.

Re-uses everything from :mod:`temp_bench.case_studies.backtracking`
(``StageA``, ``Cohort``, ``SonnetBacktrackingJudge``, ``run_phase1_unsteered``,
``generate_continuation_panels``, ``compute_delta_gc``) — only the
steering-vector hook is swapped from the legacy ``SteeringHook``
(constant ``d_in`` vector × magnitudes) to
:class:`temp_bench.eval.steering_hooks.TXCSteeringHook` (mode-parameterised).

Pipeline per (arch, feature, mode):

  1. Load arch + cohort + Stage A (cached identical to existing C7 path).
  2. Mine top features once per arch via ``mine_top_features`` (existing).
  3. For each ``mode ∈ {v0, v1, v2, v4}``:
     a. Build the hook via :func:`temp_bench.eval.steering_hooks.build_hook`.
     b. For V1 only: sweep ``cycle_phase ∈ [0, T)`` and pick the best
        per-feature peak Δgc.
     c. Run the existing ``generate_continuation_panels`` over the full
        25-magnitude grid × cohort; persist Sonnet judge calls as usual
        to ``judge_outputs.jsonl``.
  4. Compute Δgc per (arch, mode, mag) via existing ``compute_delta_gc``.
  5. Tabulate peak Δgc + peak magnitude + stability per mode + Δ vs V0.

Cost: ~4× the existing single-feature C7 cell (V1 sweeps T phases).
The judge's mandatory-persistence design ensures partial cells resume
cleanly — re-running on a partial workspace skips already-judged
panels.

This script is **the** integration point: agent_back imports our
hook, runs this driver, and lands the results in
``experiments/c7_backtracking/results/`` — owning the sweep continues
to live in agent_back's territory; the steering hook just becomes a
parameter.

NOTE — this script intentionally does NOT execute on this branch's
checkout because:
  * the steering pipeline imports
    ``temp_bench.case_studies.backtracking.run_phase1_unsteered`` etc.
    which are present on ``final`` but not yet on ``final-aniket``;
  * loading R1-Distill-Llama needs `meta-llama/...` HF auth which
    agent_back has provisioned but is not part of this branch's
    smoke test.

Agent_back: copy this file unchanged into ``experiments/c7_backtracking/``
or import it from here once the branches are merged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# det_steer src on PYTHONPATH.
_DETSTEER_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_DETSTEER_SRC) not in sys.path:
    sys.path.insert(0, str(_DETSTEER_SRC))


def _build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, help="arch name in locked_archs.yaml (txc_base, txc_pro)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_id", type=int, default=None,
                   help="if omitted, picks top D+/D- selectivity feature via mine_top_features")
    p.add_argument("--modes", default="v0,v1,v2,v4")
    p.add_argument("--magnitudes", default=None,
                   help="comma-separated magnitudes (defaults to DEFAULT_MAGNITUDE_GRID)")
    p.add_argument("--cycle_phases", default=None,
                   help="V1 cycle_phase values to sweep (default: 0..T-1)")
    p.add_argument("--workspace", required=True, help="results/runs/<eval_key>/ for this cell")
    p.add_argument("--no_sqrt_t", action="store_true", help="disable √T energy correction")
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = _build_args()

    import torch

    from temp_bench.architectures import load_arch  # type: ignore  # supplied by final
    from temp_bench.case_studies.backtracking import (  # type: ignore  # supplied by final
        DEFAULT_MAGNITUDE_GRID,
        StageA, build_cohort, load_stage_a, load_reasoning_lm,
        SonnetBacktrackingJudge, _build_prompt,
        cut25_token_position, generate_continuation_panels,
        run_phase1_unsteered, mine_top_features, extract_labeled_sentence_acts,
        split_pos_neg, compute_delta_gc,
    )
    from temp_bench.eval.steering_hooks import (
        ALL_MODES, build_hook, TXCSteeringHook,
    )
    import asyncio

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    modes = tuple(m.strip() for m in args.modes.split(","))
    for m in modes:
        assert m in ALL_MODES, f"unknown mode {m!r}; allowed {ALL_MODES}"

    if args.magnitudes is None:
        mags = DEFAULT_MAGNITUDE_GRID
    else:
        mags = tuple(float(m) for m in args.magnitudes.split(","))

    # ── 1. Setup ──
    stage_a = load_stage_a()
    cohort = build_cohort(stage_a)
    arch = load_arch(args.arch, component="c7", seed=args.seed).cuda().eval()
    judge = SonnetBacktrackingJudge(workspace=workspace)

    # ── 2. Mine top feature (or accept caller's choice) ──
    if args.feature_id is None:
        sa = extract_labeled_sentence_acts()
        pn = split_pos_neg(sa)
        mined = mine_top_features(arch, pos_activations=pn["pos"], neg_activations=pn["neg"], top_k=1)
        feature_id = mined[0].feature_id
    else:
        feature_id = int(args.feature_id)

    # ref_norm from the dom-base-union vector (existing C7 convention).
    ref_norm = float(stage_a.dom_vectors["base"]["union"].norm().item())

    # ── 3. Phase 1 unsteered (cached identically to existing pipeline) ──
    rmodel, rtok = load_reasoning_lm()
    phase1 = run_phase1_unsteered(
        cohort, workspace=workspace,
        max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size,
        model=rmodel, tok=rtok,
    )
    by_qid = {r["unique_id"]: r for r in phase1}

    # ── 4. Per-mode (and per-cycle_phase for V1) generation ──
    layer = 10
    layer_module = rmodel.model.layers[layer]
    cycle_phases_default = list(range(arch.W_dec.shape[1]))
    cycle_phases = (
        [int(c) for c in args.cycle_phases.split(",")]
        if args.cycle_phases else cycle_phases_default
    )

    panels: list[tuple[str, list[float], list[list[int]], list[float], list[int]]] = []
    for qid in cohort.all:
        row = by_qid[qid]
        cut_pos = cut25_token_position(row["unsteered_token_ids"])
        prefix = row["unsteered_token_ids"][:cut_pos]
        remaining = max(64, len(row["unsteered_token_ids"]) - cut_pos)
        for m in mags:
            panels.append((qid, _build_prompt(row["problem"]), prefix, float(m), remaining))

    cell_results: dict[str, dict] = {}
    for mode in modes:
        phases = cycle_phases if mode == "v1" else [0]
        for phase in phases:
            cell_key = f"{mode}_phase{phase}" if mode == "v1" else mode
            print(f"[run_steering_ab] {args.arch} feat={feature_id} {cell_key}")
            hook = build_hook(
                arch, feature_id=feature_id, mode=mode,
                ref_norm=ref_norm, cycle_phase=phase,
                sqrt_t_correction=not args.no_sqrt_t,
            )
            handle = layer_module.register_forward_hook(hook)
            try:
                outs = generate_continuation_panels(
                    rmodel, rtok, hook,
                    problem_prompts=[p[1] for p in panels],
                    prefix_token_ids=[p[2] for p in panels],
                    mags_per_panel=[p[3] for p in panels],
                    max_new_per_panel=[p[4] for p in panels],
                    batch_size=args.gen_batch_size,
                )
            finally:
                handle.remove()

            judge_rows = [
                (qid, m, f"{args.arch}.{cell_key}", args.seed, prompt, gen)
                for (qid, prompt, _, m, _), gen in zip(panels, outs)
            ]
            asyncio.run(judge.judge_many(judge_rows))

            # Reload all judge rows + restrict to this (arch, cell_key) tag.
            all_rows = []
            with judge._jsonl.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("arch") == f"{args.arch}.{cell_key}":
                        all_rows.append(rec)
            delta = compute_delta_gc(all_rows)
            peak = delta["peak"].get(f"{args.arch}.{cell_key}", (0.0, 0.0))
            cell_results[cell_key] = {
                "mode": mode,
                "cycle_phase": phase,
                "feature_id": int(feature_id),
                "peak_delta_gc": float(peak[1]),
                "peak_magnitude": float(peak[0]),
                "stability": delta["stability"].get(f"{args.arch}.{cell_key}", "?"),
                "by_mag": {
                    str(m): float(d) for (a, m), d in delta["by_arch_mag"].items()
                    if a == f"{args.arch}.{cell_key}"
                },
            }

    # ── 5. Pick best V1 phase (if swept) and write results ──
    if "v1" in modes and any(k.startswith("v1_phase") for k in cell_results):
        v1_keys = [k for k in cell_results if k.startswith("v1_phase")]
        best_v1 = max(v1_keys, key=lambda k: cell_results[k]["peak_delta_gc"])
        cell_results["v1_best"] = {**cell_results[best_v1], "from": best_v1}

    out = {
        "arch": args.arch,
        "seed": int(args.seed),
        "feature_id": int(feature_id),
        "ref_norm": ref_norm,
        "magnitudes": list(mags),
        "modes": list(modes),
        "sqrt_t_correction": not args.no_sqrt_t,
        "cells": cell_results,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (workspace / "steering_ab.json").write_text(json.dumps(out, indent=2))
    print(f"[run_steering_ab] wrote {workspace / 'steering_ab.json'}")
    for k, v in cell_results.items():
        print(f"  {k:20s}  peak Δgc={v.get('peak_delta_gc', float('nan')):+.3f}"
              f"  @ mag={v.get('peak_magnitude', float('nan')):+.1f}"
              f"  stab={v.get('stability', '?')}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
