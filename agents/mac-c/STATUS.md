# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ⚑ **THE WHOLE TASK HUNT, END TO END** (`hunt-mac-c-takeover.md`),
re-aimed by `briefings/hunt-safety-gold-clew.md` (active, owner mac-c)
**Last update:** 2026-07-28 20:15 BST (stamped from `date` at write time)

---

# RESUME HERE

## State in one paragraph

**Item 7 still has no KEEP, and I no longer think the bottleneck is task
choice.** Tonight closed two directions with receipts and then found
something structural: **the benchmark's arm has almost no room to beat
its own floor, for geometric reasons.** The floor sees `T + w` tokens
(~89); every probed token only ever saw **64–128** tokens of context,
because the activation cache chunks the stream into independent
128-token sequences. So the arm's entire structural advantage is the
band ~89→128 — **a factor of 1.4.** Everything else tonight follows from
that. **$0 spent all night. 0 mac-c pods (API-verified 19:11).**

## Prior lane, for context

**`retryesc_gen` CLOSED — WEAK 3/3** (`retryesc_gen/RESULT.md`), ≈$22
($21 gen + $0.68 screen), pod terminated + API-verified. Gain cleared on
all three legs (+0.063…+0.069); **the floor clause killed all three.**
The retryesc family is closed at **attempt 2 of 2**; the calibrated
re-aim I floated at 16:05 is **dropped, not deferred**.

## What is on record tonight (all pushed, all $0)

1. **`floor_excess ≡ f` is NOT a law** — it is a low-density
   approximation. The floor's second feature is
   `dose_window_count(**event_mask**, T)`, so its effective window is
   `T + w` (`w` = masked event-TURN width; evalage 13, retryesc_gen 25).
   Replacing `f` with **`P(any masked token in window)`**: mean |resid|
   0.0391 → **0.0056** (max 0.0075) across **6 legs / 2 corpora**,
   densities 0.048–0.289. `retryesc_gen/floor_predictor_test.py`.
   **My class-balancing hypothesis was REFUTED** (`manifest_f_test.py`:
   the row walk moves `f` 0.1853 → 0.1875, 2.8% of the gap).
2. **clew sweep** — 8 concept-level queries + `cited-by` on 3 nodes, all
   listed in LOG for reproducibility. Citation graph surfaced *Regime
   Leakage* and *Why Safety Probes Catch Liars But Miss Fanatics*,
   unreachable by keyword. Registry's safety states are
   **cumulative/regime**, not recency. *Alignment Faking* `cited-by`
   returned `corpus_n 0` over `fetched_n 0` = **UNMEASURED**, not
   evidence of absence.
3. **Cumulative face: arm at CHANCE** (`facecmp/arm_test.py`).
   `rate_H512` gain **+0.0024**; the foreign-context null *beat* the real
   arm. **Positive control on the identical local pipeline reproduced the
   pod** (+0.0596 vs pod's +0.0645), so the negative is about the LABEL,
   not instrumentation. **Refuted my own bar-first reasoning**: both bars
   were lower and it bought nothing, because the arm fell further than
   the bar did.
4. **Face battery, 6 faces** (`facecmp/face_battery.py`): **gain vs
   floor_excess Pearson +0.871 / Spearman +0.886; 0/6 beat their floor.**
   `ewma_tau128` **clears gain (+0.0552)** — the graded accumulator DOES
   read, so `rate_H512` failed on discretisation+horizon, not on
   cumulative structure.
5. **⚑ TWO STACKED CEILINGS** (`facecmp/ceiling_test.py`,
   `facecmp/run_seq512.py`):
   - **Apparatus:** chunks are independent 128-token sequences. Recency
     terciles sit at ages **121/286**, so 2 of 3 classes are "no event in
     my context" — per_class `[0.516, 0.394, 0.403]`. Re-tercileing
     INSIDE the context (ages 46/72) nearly triples gain
     (+0.0596 → **+0.1707**) and makes per_class uniform
     `[0.845, 0.873, 0.839]`.
   - **Representational:** rebuilding the cache at **SEQ_LEN=512** moved
     recency gain only +0.0596 → **+0.0928** (per_class
     `[0.492, 0.433, 0.416]`), and `rate_H512` stayed at chance *with its
     full horizon present*. **The floor held as pre-registered**
     (0.5859 → 0.5932 — it depends only on `T + w`).
   - **I over-inferred "raise SEQ_LEN and the arm unlocks"; the 512 run
     corrected me within the hour.** It buys ~+0.03, not +0.11.

## ⚠ The open caveat, and what is running

**Everything in (5) is gpt2 ONLY — the weakest leg.** `gemma2_2b` /
`llama31_8b` screen at layer 14 and are plausibly far better at
long-range integration. **Not extrapolating.**

**DONE (20:21) — scale step 124M → 355M, `facecmp/scale_test.py` +
`scale_test_seq512.py`, results in `facecmp/results/scale/`.
**A 3× scale step does NOT extend the readable horizon** — gpt2-medium
is slightly *worse* than small at **both** context lengths (@128 +0.0389
vs +0.0596; @512 +0.0754 vs +0.0928). **My prediction was wrong.** The
within-context face reproduces to ~0.02 per-class across two models and
two context lengths, and **the floor never moved** (0.5859/0.5932) — the
`T + w` prediction has now held across 2 models, 2 context lengths and 2
corpora. ⚠ **Disclosed design error:** the @128 arm of that test could
not answer the representational question at all (out-of-context classes
are absent from the input for *any* model); caught on reading the output
and re-run at 512.

## ⚑ THE SYNTHESIS — hand this over first

**Floor horizon = `T + w` ≈ 89 tokens** at the screened setting.
**Model readable horizon ≈ 100 tokens** — both gpt2 scales, both context
lengths, per-class collapsing to ~0.39–0.43 beyond it. **These are the
same number.** The arm's entire structural opportunity is the band
`(T+w, readable_horizon)` = **(89, 100), a factor of 1.12.** That one
fact predicts the scissors, ρ +0.871, 0/6 beating their floor, and
`arm − floor` falling monotonically with T (T16 −0.017, T32 −0.077,
T64 −0.148; T4 +0.025 is the only positive on record).

**Untested inference:** shrink the FLOOR's horizon rather than chase the
model's — **T=16 with narrow events (w≈4)** gives ~20 vs ~100, a factor
of **5**. `retryesc_gen` violated all three (T=64, w=25, terciles
121/286 — the label's whole range outside the readable horizon). **Not
frozen into a card**: the last two bar-side designs did not carry the
arm.

## Next actions, in order

1. **Read `scratchpad/scale_test.log`**, copy artifacts into
   `facecmp/results/`, LOG the scaling result whichever way it falls.
2. **The 2b/8b ceiling test decides whether ceiling 2 is a gpt2 fact or a
   program fact.** Needs a weights download or a pod. **Highest-value
   next experiment.**
3. **Do NOT freeze another generation card until (2) resolves.** If the
   readable horizon is short on every leg, the fix is apparatus/model
   geometry, not another task from the registry — and a $21 corpus aimed
   into a factor-1.4 band is wasted money.
4. `briefings/hunt-safety-gold-clew.md` **stays active** — the sweep is
   not exhausted. Untested leads: belief-drift under accumulating
   context, deception-maintenance/commitment, steering-injection age.

## Standing constraints (unchanged)

- **RunPod (Dmitry's key, BINDING):** keychain `dmitrys-runpod-api-key`,
  **mac agents only**, never seeded to a pod, env-inject only, never
  echoed/filed/argv'd. **$10/hr max per agent**; **terminate the moment
  it is unused, prefer TERMINATE, verify by API after**; **never modify
  pods you did not spin up**; name `mac-c-<purpose>-<mmdd>`; ledger at
  spin-up AND termination.
- **Generation** = shared `dmitry-mats-claude-api-key` ($300 shared cap,
  GENERATION ledger), mac-only, never on pods.
- **clew READ-ONLY**: never `sync`/`register`/`seed add`/`clip`, **never
  `--refresh`**. S2: env-only, **never argv**, 1 req/s CUMULATIVE.
- **Never `set -x` in a script that touches a secret** (my own rule after
  leaking a GitHub token into a pod log; contained + verified).
- **Attempt caps are per face-family.** `retryesc` closed 2/2. The
  cumulative family is at attempt 1 and I would call it spent given (3).
- **Re-arm the listener after every wake** —
  `zsh <scratchpad>/listener.sh` as a background task. It fires on my own
  pushes; expected, just re-arm.
- **Stamp from `date` at write time.** Three stamps corrected today.

## Error tally (mine, today) — the pattern matters more than the count

"capped near 1/3" → `K=0.63` → uniform-position gap map → the
class-balancing hypothesis → "raise SEQ_LEN and the arm unlocks".
**Every one was a model checked against itself rather than against the
instrument that sets the bar.** Rules earned, in order: *measure, don't
model* → *check instrument and bar use the same ROWS* → **check they use
the same EVENTS** → **and check the model was actually SHOWN the thing
you are asking it to encode.**

_Recorded-by: claude-opus-5 (mac-c)_
