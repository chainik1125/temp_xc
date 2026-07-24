---
status: active
created: 2026-07-24
for: runpod
venue: runpod (32C CPU)
---

# Rebuttal pack — paper-scale cost accounting + print-ready receipts

**You are `runpod`** (32C). Everything here is CPU-complete and feeds
the humans typing responses before **2026-07-27** (team check-in
Sunday 10:00 PT; your deliverables wanted by Saturday morning PT).
HARD RULE: no reviewer/meeting quotes or paraphrases in any tracked
file — organize by neutral topic labels (e.g. "parameter accounting",
"temporal isolation", "seed robustness"), never by reviewer. Output
directory: `docs/rebuttal_pack/` with a README index. Scripts
committed before their outputs, as always.

## 1. Paper-scale parameter + inference-cost table (the highest-value item)

The existing param table (STORY.md § 6) is at toy scale (d_in = 128).
Redo it at the PAPER's real configurations so the numbers are directly
quotable: for each real-world task config — sparse probing
(gemma-2-2b-it L13), backtracking (R1-Distill-Llama-8B L10), EM
(Qwen-2.5-7B-Instruct L15) — instantiate the registered arch classes
(per-token TopK/BatchTopK, T-SAE, Stacked, TXC-base/pre/post, MLC and
TFA if registered; skip with a note if not) at the paper's
d_in/d_sae/T/k and report: exact parameter counts (formula + number),
training-memory note where relevant, and inference MACs/token under
the tiled protocol (+ the sliding-deployment ×T caveat and the
window-buffering latency note, per STORY § 6). Pull the exact paper
configs from the repo history (`git show origin/final:purified/...`
configs; read-only) and the paper appendix conventions; DISCLOSE every
assumption where a config is ambiguous. Include the dictionary-size
accounting for each arch at matched window budget (why nominal knobs
differ across families while realized atoms/token match — the Part II
fairness convention, stated plainly). Deliverable:
`docs/rebuttal_pack/param_costs.md` + `param_costs.py`.

## 2. Print-ready receipt set (consolidation, not new science)

One directory, one README that maps artifact → topic (neutral labels).
Re-render from canonical sources only (leaderboard, story_figs,
committed results JSONs — never hand-typed numbers):

- temporal isolation: the regime-2 vs regime-3 Stacked boundary
  (Stacked ≈ TXC at 0.95 on backtracking-mirror; Stacked ≈ 0 where
  mixing wins) — one small table + the existing isolation figure;
- seed robustness: the suite's 3-seeds+untrained convention in one
  paragraph + the Stage-2 variance-receipt headline numbers
  (`support_stats/stage2_variance.json` — cite, don't recompute);
- the λ̂ Stage-2 figure (variance-aware render, matched-only variant)
  and the shuffle-receipt figure/table — copied from their canonical
  locations with pointers back;
- component dissection: the one-page component table
  (`loss_dissection/results/dissection_table.md` distilled) + the § 7
  sentence;
- probing corollary: the regime-1 explanation of tight probing
  clusters + flat window-length response (STORY § 4), with the paper's
  own numbers cited from the paper, clearly marked as paper values.

## Acceptance gate — stop for review

`docs/rebuttal_pack/` complete with README index; scripts before
outputs; no reviewer/meeting references; STATUS rewritten. Briefing
stays until mac-local review.
