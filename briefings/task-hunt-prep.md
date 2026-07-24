---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod
---

# Task-hunt prep — label mining + candidate specs for the hunt

**You are `runpod-b`** (32C). You feed the hunt pods — `runpod-d`
(`briefings/task-hunt.md`: λ̂ + proof-op) and `runpod-e`
(`briefings/task-hunt-b.md`: repetition-lag + confidence trend) — read
both first: your job is CPU-side label engineering and candidate
mini-card drafting so the GPU pods never wait on labels. Shared-branch rules apply; commit label artifacts under
`experiments/explorations/task_hunt/labels/` with a build script each
(committed before its outputs). Fail fast; check-in 2026-07-26.

## Deliverables, in priority order

1. **Repetition-lag Δ labels (candidate 2 — exact, zero-API).** A
   builder that takes a tokenized corpus slice (fineweb sample in
   `expansion/data/`, or re-pull; also build for the Ward stream's
   tokenizer) and emits per-position labels: Δ = distance to the
   previous occurrence of the current token n-gram (n ∈ {1, 2}),
   bucketed Δ ∈ {≤4, ≤8, ≤16, none}, plus balanced probe-row
   manifests per Δ-bucket. Include the null: shuffled-window Δ
   distribution. Ship with 5 sanity tests.
2. **Backtracking-intensity targets (candidate 1).** From the Ward
   stage-A event labels (`results/c7_backtracking/stage_a/`, read-only)
   + the fitted Hawkes mirror (`backtracking/results/` kernel params):
   per-position λ̂ targets on the Ward stream token grid (the same
   exponential-kernel convolution the synthetic mirror uses — reuse
   `backtracking/measure.py` machinery). Regression targets + binned
   classification variant; manifest format identical to (1).
3. **Proof-operation run features (candidate 3).** From the expansion
   corpus labels: per-sentence operation phase → token-grid
   time-in-run / run-boundary targets, with the sentence→token clock
   bridge stated explicitly (audit item 6) — flag tokens-per-sentence
   stats so the T range is chosen honestly.
4. **Mini-cards** for candidates 2–4 (candidate 1's card is
   runpod-c's, using your targets): one page each per the task-hunt
   Stage-1 format — label definition, non-ambience argument, predicted
   T-pattern (STORY.md § 7 taxonomy), falsifier. Frozen by commit
   before the corresponding screen runs.

## Acceptance gate

Labels 1–2 + cards pushed within ~6 h (runpod-d/e block on them);
label 3 + card 4 after. STATUS rewritten. No reviewer/meeting quotes
in tracked files. Briefing stays until mac-local review.
