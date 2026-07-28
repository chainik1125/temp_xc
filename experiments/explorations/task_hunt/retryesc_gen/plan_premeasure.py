"""PLAN-TIME premeasure for `retryesc_gen` — $0, runs before any API call.

Executes the cheap half of GENERATION_CARD s6 step 1. Four gates, all
computable from the plan alone, i.e. before a single token is
generated or a cent is spent:

  1. CLOCK        projected gap median inside GAP_RANGE (card s2.2a)
  2. VOCABULARY   event rate independent of task (card s4) -- the gate
                  `retryesc` failed, at unigram 0.689-0.716 vs 0.60
  3. POSITION     strategy pool must not exhaust mid-episode, else late
                  failures are FORCED repeats and event-status becomes
                  a function of position (the `reask_hr` confound)
  4. CORPUS CLOCK projected tokens/doc vs the bar `dharm` died on (155.6)

Gate 1's projection assumes realised assistant length ~= 0.8x the
sampled cap. THAT ASSUMPTION IS NOT VERIFIED HERE -- the pilot measures
`claim_zone` directly and re-tunes `P_REPEAT`. This script exists to
kill a bad design before the pilot, not to replace it.

Run:
  .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.plan_premeasure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labels"))
import retryesc_gen_lib as rg          # noqa: E402

N_DOCS = 400
SEED = 0
REALISED_FRAC = 0.8        # assumed realised/cap ratio -- pilot verifies
ENV_OVERHEAD = 22          # env text + directive tokens per pair
DHARM_FATAL_TOK_PER_DOC = 155.6
OUT = Path(__file__).resolve().parent / "results" / "plan_premeasure.json"


def main() -> None:
    rng = np.random.default_rng(SEED)
    plans = rg.plan(rng, rg.TASKS, N_DOCS)

    n_ev = sum(p.meta["n_events"] for p in plans)
    n_pair = sum(p.meta["n_pairs"] for p in plans)
    cap = float(np.mean([q.max_new for p in plans for q in p.pairs]))
    tok_per_pair = cap * REALISED_FRAC + ENV_OVERHEAD
    rate = n_ev / n_pair
    gap = tok_per_pair / rate
    tok_per_doc = tok_per_pair * float(
        np.mean([p.meta["n_pairs"] for p in plans]))

    si = rg.schedule_independence_check(plans)
    ex = rg.exhaustion_check(plans)

    gates = {
        # ⚠ SUPERSEDED by dry_run.py, which measures `f` directly instead
        # of projecting it from a gap. The projection assumed uniform
        # probe positions; eligible tokens are assistant tokens offset
        # from events by the masked environment turn, so f is much lower
        # at a given gap. Kept as a descriptive receipt, NOT a gate.
        "1_clock_SUPERSEDED": {
            "projected_gap_median": round(gap, 1),
            "superseded_by": "dry_run.py -> claim_zone frac_in_window T64",
            "why": "projection assumed uniform probe positions; "
                   "eligible tokens sit in assistant turns offset from "
                   "the event by the masked environment turn",
            "pass": True},
        "2_vocabulary": {
            "event_rate_cv_across_tasks": round(si["cv"], 4),
            "spread": round(si["spread_max_minus_min"], 4),
            "bar_proposed": si["cv_bar_proposed"],
            "pass": not si["stop"]},
        "3_position": {
            "frac_docs_exhausting_pool":
                round(ex["frac_docs_exhausting_pool"], 4),
            "pool_size": ex["pool_size"],
            "event_rate_rise_first_to_last_third": round(ex["rise"], 4),
            "pass": bool(ex["frac_docs_exhausting_pool"] < 0.05)},
        "4_corpus_clock": {
            "projected_tok_per_doc": round(tok_per_doc),
            "dharm_fatal": DHARM_FATAL_TOK_PER_DOC,
            "multiple_of_fatal": round(tok_per_doc / DHARM_FATAL_TOK_PER_DOC, 1),
            "pass": bool(tok_per_doc > 10 * DHARM_FATAL_TOK_PER_DOC)},
    }
    all_pass = all(g["pass"] for g in gates.values())

    print(f"PLAN-TIME PREMEASURE — {N_DOCS} docs, seed {SEED}, "
          f"P_REPEAT={rg.P_REPEAT}\n")
    print(f"  events {n_ev}  pairs {n_pair}  event rate {rate:.3f} "
          f"(one every {1 / rate:.2f} pairs)")
    for name, g in gates.items():
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {name}: "
              + "  ".join(f"{k}={v}" for k, v in g.items() if k != "pass"))
    print(f"\n{'ALL PLAN-TIME GATES PASS' if all_pass else 'GATE FAILURE'}"
          f" — {'proceed to pilot' if all_pass else 'do NOT generate'}")
    print("\nGates 2-4 are genuinely plan-side. The CLOCK is no longer "
          "gated here: dry_run.py measures `f` directly on a built "
          "stream, and the PILOT re-measures it on real prose.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "n_docs": N_DOCS, "seed": SEED, "p_repeat": rg.P_REPEAT,
        "assumed_realised_frac": REALISED_FRAC,
        "event_rate": round(rate, 4), "pairs_per_event": round(1 / rate, 2),
        "gates": gates, "all_pass": all_pass,
        "schedule_independence": si, "exhaustion": ex,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")


if __name__ == "__main__":
    main()
