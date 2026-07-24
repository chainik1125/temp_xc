---
status: active
created: 2026-07-24
for: runpod
venue: runpod (32C CPU)
---

# Candidate factory, broad corpus — QUANTITY MODE (Han directive)

**You are `runpod`.** Program directive (Han, 2026-07-24 evening):
**quantity over quality — maximize screen-ready case-study
candidates.** runpod-b owns the Ward-trace batch
(`candidate-factory-traces.md` — read it; its bundle format, triage
kill authority, and masking discipline govern you identically). You
own everything OUTSIDE the Ward corpus. GPU economics: runpod-e's
volume holds fineweb caches for gpt2/gemma-2-2b/llama-8b — candidates
on fineweb text screen for minutes; candidates on a NEW corpus need
one cheap caching pass (~minutes/model on an H100) — still fine, but
say so in the bundle. **Ship incrementally by Saturday morning PT.**

## 1. The idea ledger FIRST (~an hour, the quantity artifact)

Commit `experiments/explorations/task_hunt/CANDIDATES.md`: 10–20
candidate ideas, one paragraph each, each vetted against the four
round-1 lessons — (a) **conversion risk**: does the latent help
next-token prediction? (if yes the model has likely linearized it —
the round-1 graveyard); (b) **label-side per-token proxy**: is the
label readable from the current token's identity? (c) **clock
feasibility** at panel T; (d) regime shape + predicted T-pattern.
Verdict per idea: BUILD / PARK / DEAD with one-line reasons. Ideas
that die in the ledger are deliverables too — the next hunter starts
from this page.

## 2. Build the top vetted bundles (target ≥ 3)

Seeds for the ledger (vet, don't assume — and add your own):

- **Interleave `tss` (finish it — labels exist).** Turn runpod-b's
  committed interleave artifacts into a screen-ready bundle with
  `tss` PRIMARY (unigram 0.55, near-blind) and source-identity
  explicitly demoted to a disclosed anchor (0.66 = expected regime-1).
  This is the anti-conversion class's one prepared shot; the parked
  status is LIFTED for screening under quantity mode.
- **Vocabulary-novelty rate on fineweb** (fraction of window tokens
  unseen earlier in the doc — a topic-drift intensity; exact).
- **List/enumeration density trend on fineweb** (markers exact; mask
  marker tokens per the traces-briefing discipline).
- **NEW-corpus intensity candidates** (CPU-downloadable, exact
  labels; models must be ones with existing caches or cheap to cache):
  e.g. OpenWebMath equation-density intensity; dialogue corpora
  turn-length / speaker-switch rate (a grounded cousin of interleave
  `tss`); news chronology density. Pick by ledger vet, not by list
  order.
- AVOID (already dead, recorded): bracket/indentation state-tracking,
  repetition detection, forbidden-word onset, emotional onset,
  anything whose primary is a rollout-level boolean.

Per bundle: same format + triage + CARD_DRAFT as the traces briefing.
New-corpus bundles must include the tokenized corpus artifact (or an
exact re-pull script) so a GPU pod caches without decisions.

## Acceptance gate — stop for review

CANDIDATES.md ledger committed; ≥ 3 bundles shipped or honestly
triage-killed; LOG line per bundle/kill; STATUS rewritten; no
reviewer/meeting quotes. Briefing stays until mac-local review.
