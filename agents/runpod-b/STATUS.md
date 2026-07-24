# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 — **TASK-HUNT PREP SESSION COMPLETE
(`briefings/task-hunt-prep.md`), acceptance gate reached; briefing left
in place until mac-local review.** No task in flight.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
Freqbench meter unchanged: **$1.63 / $25** ($0 this session — label
engineering only, no training, no judges).

## Task-hunt prep outcome (all committed + pushed)

Everything under `experiments/explorations/task_hunt/` (LOG has the
prep entry with receipts):

- **`labels/`** — `lib.py` (pure label logic, 10 sanity tests in
  `tests/test_task_hunt_labels.py`; suite 190 green) + `wardmap.py`
  (shared Ward-grid broadcast; round-trip validity 0.9997098… matches
  committed ward_stream_stats digit-for-digit) + four builders
  (committed pre-run) and their artifacts:
  `replag_fineweb_{gpt2,gemma2,llama31}.npz` (~5.4 MB each; exact
  token_ids = the alignment contract; within-doc-shuffle Δ null),
  `ward_lambda.npz` (causal mirror λ̂ + terciles + is_bt control; event
  rate by tercile 0.053/0.081/0.256), `proofops.npz` (op / time-in-run /
  boundary + **clock bridge: median 16 tok/sentence ⇒ 2-sentence
  windows need T ≥ 32**), `confidence.npz` (hedge state + slope4/8;
  slope8 support ≈ 128 tokens). Stats JSONs alongside; traces.json
  re-ported at build time per ATTRIBUTION (deliberately uncommitted).
- **Cards:** `proofops/CARD.md`, `confidence/CARD.md` frozen (science
  sections; running agents append screen-cell tables). `forbidden_word/`
  + `emotional_instability/` **CARD.DRAFT.md** staged (running agents
  freeze their own).
- **Replag card NOT written by me:** runpod-e froze `replag/CARD.md`
  with its own inline labels first (briefings updated 2026-07-24: labels
  are a parallel convenience, not a gate). Scheme differences recorded
  in the LOG entry; my artifacts stand as cross-check + n=2 reserve.
  Label-side receipt: short-lag bigram repeats ≈13× the frequency-only
  null; unigrams ≈1.3× (n=2 is the sharper order task).

## Items for mac-local review
- Clock-bridge consequence worth a decision: arm-B candidate 2's
  primary target (slope8) has ≈128-token support — at the mandated
  screen T ≤ 64 only slope4 reaches full coverage. The card freezes
  "timescale unreachable" as a valid kill; runpod-e may reasonably jump
  to its candidate 3.
- `confidence.npz` / `proofops.npz` and the two frozen cards are mine —
  check the KEEP/KILL thresholds (0.02/0.05 AUC, borrowed from
  runpod-e's frozen replag card) before any screen consumes them.

## Operational notes
- 5 agents on `arxiv`: runpod (loss dissection), runpod-c (EM redo),
  runpod-d (hunt arm A), runpod-e (hunt arm B), me. Pull-rebase before
  every push (one LOG.md conflict already resolved in runpod-e's
  favor); cite commit SUBJECTS not SHAs.
- Rewrite this file before any compact.
