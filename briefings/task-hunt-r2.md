---
status: active
created: 2026-07-24
for: runpod-d, runpod-e
venue: runpod (GPU)
---

# Task hunt, round 2 — sharpen the positives (post-review follow-ups)

Round 1 is REVIEWED & APPROVED (verdicts + binding review notes:
`experiments/explorations/task_hunt/LOG.md`; methods `RECORD.md`).
This round exists to sharpen the two positives before the **Sunday
2026-07-26 10:00 PT check-in**; results wanted by **Saturday morning
PT**. Prime directive unchanged: a sound verdict, never a win. Frozen
cards/amendments before every run; append to the shared LOG; canonical
runner + 0-dup-key hygiene; no reviewer/meeting quotes in tracked
files.

**New hunt convention (adopted at review, binds all future screens):
per-token-first triage.** Before any window grid, run the per-token
linear probe alone; a candidate whose per-token ceiling is already
high (≫ chance) on its primary target is presumptively converted —
escalate to the full grid only with a card-stated reason. All five
round-1 kills were visible in that one number.

## runpod-d — the budget-matched TXC-post re-run (highest-leverage cheap run in the program)

1. **Re-run TXC-post on `ward_real_lambda_base_l12` with realized-l0
   matching.** Round 1's post cells collapsed to l0 = 3.4 → 0.49 as T
   grew (the post-squash `k_win // T` correction), so its monotone
   rise to 0.255 at T = 16 is budget-confounded. Freeze a short
   amendment card (target realized l0 ≈ 7–8 at every T — raise nominal
   k accordingly; state the per-T nominal k you compute), then run
   post × T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} + untrained (~24 cells).
   Two pre-registered readings: (a) the rise survives matching ⇒ the
   money plot upgrades from TXC-pre-peaks-at-8 to a monotone
   matched-budget line through T = 16 — a materially stronger rebuttal
   figure; (b) it does not ⇒ the 0.255 was sparsity-starvation
   behavior, recorded, and TXC-pre remains the headline. Either way
   the panel gains its missing cell.
2. **Figure hygiene (review note 3, mandatory):** amend the Stage-2
   renderer so every arch line carries its realized-l0 range in the
   legend (or annotate TXC-post's collapse directly). The current
   figure visually crowns a non-budget-matched line; it must not leave
   the repo that way.
3. Parked (do NOT run): proof-op Stage-2 on distill L12 — the raw
   contrast (+0.017…+0.042) is too thin to clear a trained panel by
   Saturday; post-rebuttal.

## runpod-e — both items runnable now (caches already on your volume)

1. **Hedging-trend LEVEL Stage-2 (program decision: an
   aggregation-framed win is ACCEPTED).** Your candidate-2 seed —
   window-readable, per-token-blind, monotone-in-T (distill mean-probe
   0.52 → 0.57 vs tok 0.47) — goes to a head-to-head panel on a
   **FRESH card** (the killed confidence card is not its
   confirmation). Frame: regime-2 aggregation latent; **shuffle
   IMMUNITY is disclosed as the mechanism receipt** (order-free
   pooling is the claim, not order). Reuse the candidate-1 Stage-2
   pattern: plugin datasource, single scarce anchor, 5 archs ×
   T ladder × 3 seeds + untrained, matched realized l0, per-tile
   readout. Deliverable: a second real-task T-scaling figure with the
   same convention sentence.
2. **Early-layer addendum (~2–3 h, zero new data):** g_order(ℓ) for
   lag4 and g_agg(ℓ) for slope8 across the cached alternate layers
   (gpt2 hs4, gemma hs8, llama hs8; 17 Ward capture points). Question:
   does the temporal signal GROW at pre-conversion depths? Either
   direction extends the conversion-depth story.

## Parked for post-rebuttal (recorded so nobody re-derives)

gpt2-scale order cell (lag-value Stage-2 + pythia scale ladder);
anti-conversion candidate class (latents with no generative training
signal — source identity in interleaved documents, externally-annotated
states); proof-op Stage-2.

## Acceptance gate — stop for review

Amendment/fresh cards frozen pre-run; LOG verdicts; figures + records;
leaderboard hygiene (0 dup keys, no null metrics); STATUS rewritten.
Briefing stays until mac-local review.
