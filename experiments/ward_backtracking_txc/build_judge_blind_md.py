"""Convert blind_pairs.csv to a GitHub-renderable markdown for human scoring.

Hides the LLM judge's verdict and the before-correct fact (so scoring is
truly blind), one transcript per section, with a fillable score block at
the top of each. Aniket reads on GitHub, fills the scores in his copy of
the markdown (or the CSV), and the validate-judge-kappa script later
loads the merged scores.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


HEADER = """---
author: Aniket Deshpande
date: 2026-05-03
tags:
  - guide
  - in-progress
  - ward-backtracking
---

## TL;DR

20 randomly-sampled steered continuations from the Stage B cut-25%
backtracking sweep. **Blind judge κ validation set.** Read each transcript
and score it 0–3 for coherence, 0/1 for backtracking-present, 0/1 for
looping-present, *without* peeking at the LLM judge's verdict (which is
hidden in the CSV columns `judge_rescued` and `before_correct`).

After scoring, save your scores into
`results/ward_backtracking_txc/judge_validation/blind_pairs_aniket.csv`
(copy of the original with `human_*` columns filled). I'll then run
`validate_judge_kappa.py` to compute Cohen's κ + raw agreement vs the
LLM judge.

Strata: 10 transcripts from "near peak" magnitudes (\\|mag\\| ∈ [3, 8])
and 10 from "extreme" magnitudes (\\|mag\\| ∈ {12, 16}), random seed 42.
You'll see a mix of "steering does its thing" and "steering breaks the
model" cases.

## Scoring rubric

| Field | Scale | Definition |
|---|---|---|
| `coherence` | 0 (incoherent / loop) — 1 (mostly nonsense) — 2 (mostly coherent w/ issues) — 3 (fully coherent) | Holistic readability + logical flow |
| `backtracking_present` | 0 / 1 | Does the steered continuation contain GENUINE backtracking — error-catching, missing-constraint detection, approach-rejection, assumption re-evaluation? **Filler ("Hmm, let me think") and pseudo-backtracking (same conclusion restated) do NOT count.** |
| `looping_present` | 0 / 1 | Does the continuation loop — sentence-level repetition for ≥3 consecutive sentences? |

Targets: ≥80% raw agreement and Cohen's κ ≥ 0.6 between Aniket and the
LLM judge would validate the judge for camera-ready. Below that, refine
the judge prompt once and re-test.

## Note on what's NOT shown

To keep the read truly blind:

- The LLM judge's coherence/backtracking-present/looping verdicts are NOT shown here — they live in the original CSV's `judge_rescued` column (which means "did the math answer flip from incorrect to correct" — a downstream metric that depends on the LLM judge). Score blind, then merge.
- The unsteered (mag=0) baseline outcome is also not shown — `before_correct` in the CSV. Don't peek; score the steered transcript on its own.

## How to fill scores

Either:

1. Edit `results/ward_backtracking_txc/judge_validation/blind_pairs.csv`
   in place — fill `human_coherence_0_3`, `human_backtracking_present`,
   `human_looping_present` for each row. **Hide `judge_rescued` and
   `before_correct` columns first** in your spreadsheet so you don't
   bias on them.

2. OR jot scores in a notebook and I'll wire them in.

---

"""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("results/ward_backtracking_txc/judge_validation/blind_pairs.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/aniket/experiments/ward_backtracking/judge_blind_validation.md"))
    args = p.parse_args(argv)

    rows = list(csv.DictReader(args.csv.open()))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    parts = [HEADER]
    for r in rows:
        # Don't show judge_rescued or before_correct — those bias scoring.
        # Hide arch + magnitude only if they'd give too much away — but
        # arch/mag are study metadata (helpful for stratifying mental
        # model), not LLM-judge output, so OK to show.
        parts.append(f"## Transcript #{r['id']}\n")
        parts.append(f"- **arch:** `{r['arch']}`")
        parts.append(f"- **magnitude:** `{r['magnitude']}`")
        parts.append(f"- **question:** `{r['question_id']}`\n")
        parts.append("**Your scores** (fill in):\n")
        parts.append("```yaml")
        parts.append("human_coherence_0_3:        # 0=loop, 1=mostly nonsense, 2=mostly coherent, 3=fully coherent")
        parts.append("human_backtracking_present: # 0/1 — genuine backtracking?")
        parts.append("human_looping_present:      # 0/1 — sentence loop ≥3 sentences?")
        parts.append("```\n")
        parts.append("**Steered continuation:**\n")
        parts.append("```text")
        # Sanitize backticks in transcript so we don't break the code fence
        text = (r["transcript"] or "").replace("```", "''' ")
        parts.append(text.strip())
        parts.append("```\n")
        parts.append("---\n")

    args.out.write_text("\n".join(parts))
    print(f"[saved] {args.out} ({len(rows)} transcripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
