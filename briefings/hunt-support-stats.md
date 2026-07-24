---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod (32C CPU)
---

# Hunt support (stats + prep) — variance receipts, renderer, round-3 optionality

**You are `runpod-b`** (32C, no GPU). Round-1 review is APPROVED
(`task_hunt/LOG.md`, mac-local review entry — its note 2 is your item
1's mandate; note 3 is your item 2's). Deliverables under
`experiments/explorations/task_hunt/support_stats/` (+ label/corpus
artifacts under `task_hunt/labels/` as before), builders committed
before outputs, LOG entries for anything verdict-shaped. **Items 1–2
by Saturday morning PT** (they gate the rebuttal figure); 3–4 after.

## 1. Variance receipts for the Stage-2 λ̂ panel (gates the rebuttal text)

R1's core complaint about the PAPER is single-seed results with
overlapping CIs — the new result must not repeat that sin. From the
84 committed leaderboard rows (datasource `ward_real_lambda_base_l12`)
+ `lambda_intensity/results/stage2_summary.json`, compute and commit
`support_stats/stage2_variance.json` + a short markdown section:

- per-seed values for every (arch, T) cell (the leaderboard rows carry
  them; the summary only has mean/std);
- **paired-by-seed** TXC-pre − T-SAE and TXC-pre − per-token
  differences at each T, with exact permutation p-values and BCa
  bootstrap CIs (n = 3 is tiny — say so honestly; the paired design is
  the point);
- a trend statistic across T = 2→8 (is the RISE itself significant,
  pooling seeds?) and the trained−untrained margin CI;
- a **power calc**: how many additional seeds would bound the
  TXC-pre-vs-T-SAE margin at 95 % given observed spread? If the answer
  is ≤ 4 extra seeds, say which cells (pre + tsae only, T ∈ {4, 8} —
  ~12 GPU cells) so runpod-d can append them cheaply to its round-2
  run. Post your recommendation as a LOG entry addressed to runpod-d.

## 2. The variance-aware Stage-2 renderer (you own it; runpod-d re-renders)

Upgrade the Stage-2 renderer (provenance: RECORD § 3b figure) so that:
(a) every arch line carries its realized-l0 range in the legend, with
TXC-post's collapse explicit (review note 3 — MANDATORY before any
external use); (b) whiskers are seed-based CIs from item 1, not ±std;
(c) an optional "budget-matched only" variant omits non-matched lines.
Coordinate note: round-2 assigns runpod-d the re-render once its
budget-matched cells land — your renderer is the implementation; note
in the LOG when it's merged so d doesn't duplicate.

## 3. Anti-conversion candidate prep (round-3 optionality, CPU-complete)

The parked candidate class (arm-B closure: conversion kills any latent
with generative training signal) needs its data side built so a GPU
pod can screen in ~2 h if one frees up Saturday. Build: an
interleaved-document corpus (two-source interleave over fineweb docs,
block lengths jittered; per-token labels = source identity +
time-since-switch), with the controls that make it survivable —
lexical-overlap matching between sources (else source identity is
vocabulary detection = ambient), a shuffled-block null, and the
per-token-first triage numbers computed on labels alone. Builder +
5 sanity tests + stats + a DRAFT mini-card (the running agent freezes
its own; state the frozen prior: per-token HIGH on source identity is
the expected kill risk — the lexical control is what the candidate
lives or dies by).

## 4. Hedging-LEVEL card support for runpod-e

Your confidence labels + clock bridge already cover most of it; ship a
DRAFT card for the hedging-LEVEL Stage-2 (label = anchor hedge level,
window-mean framing, aggregation-claimed with shuffle-IMMUNITY as the
mechanism receipt; per-tile readout convention sentence included) so
runpod-e only sharpens and freezes.

## Acceptance gate — stop for review

Items 1–2 pushed with LOG entries (incl. the seeds recommendation to
runpod-d); 3–4 as reached; STATUS rewritten; no reviewer/meeting
quotes in tracked files. Briefing stays until mac-local review.
