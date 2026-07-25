---
status: active
created: 2026-07-25
for: runpod-d
venue: runpod (H100)
---

# Stage 2 — `oprate`: the shot at CASE STUDY #2 (highest-value GPU run in the program)

**You are `runpod-d`.** Your round-2 briefing is discharged and
retired; your screens and top-up are REVIEWED & APPROVED (LOG,
2026-07-25). **Results wanted by Sunday morning PT** (team check-in
Sunday 10:00 PT; rebuttal deadline 2026-07-27).

**Why this candidate and why now.** The hunt has FIVE Stage-1 KEEPs but
still exactly ONE confirmed TXC case study (the λ̂ backtracking panel).
Screens license panels; they are not wins. `oprate` is the best shot at
a second: it is the **only independent candidate** in the batch (corr
0.026 with λ̂_sc; its two targets mutually −0.032), it cleared the
batch's **highest visible-evidence bar** (`ver` T32 = 0.830), and on
Ward `g_agg ≈ g` — a LINEAR pool carries the gain, which is exactly what
a sparse dictionary can represent. A second independent real-task panel
with the λ̂ panel's shape would be the single biggest addition to the
rebuttal.

## Scope — ONE target to a full panel, the second only if it is free

**`rate_case` is the primary** (position-blind at 0.496, the cleanest
triage in the factory; `ver`'s position face is the widest at 0.641).
Take `rate_ver` to a full panel only if `case` completes with real
headroom — **a complete single-target panel beats two partial ones.**

## The design (reuse the λ̂ Stage-2 pattern; do not invent)

Plugin datasource over the committed `labels/oprate.npz` on the Ward
base/distill caches; single scarce anchor; **5 archs × T ladder ×
seeds {1, 2, 42} + untrained controls at every line point**.

**Bindings — all of these are lessons already paid for:**
1. **Budget-match on REALIZED `l0_per_token`, never nominal k.** TXC-post
   runs at per-T nominal **k = 8·T** (your own code-rate convention).
   Pre-register the predicted realized l0 and the band; **if a trained
   cell lands outside the band, record it as a residual mismatch — do
   not call it in-band** (the correction you just made).
2. **Carry BOTH probe columns.** Set `c["eval_extra"] = {"lambda_probe_v2":
   True, "lambda_v2_probe": "ridge", "lambda_v2_n_windows": 8192,
   "lambda_v2_split": "trace"}` (recipe in `PROBE_V2_SPEC.md` § 2). Every
   row then carries its paired v1 column, so **whichever way the
   λ-readout methods decision goes, this panel never needs re-running.**
   The leaderboard-canonical number remains v1 until mac-local's rule
   fires; report both, claim on v1.
3. **Print the visible-evidence line next to every window number**
   (`case` T32 = 0.783). A window result that does not beat it at
   matched T is counting visible event sentences.
4. **Per-tile code readout convention** + the binding phrasing from the
   round-1 review ("under the code-readout convention", with the
   code-rate defense).
5. **No max-over-arms scoring.** The "best window" convention is retired
   program-wide (runpod-e's finding): fix the probe class and control
   width.
6. Canonical runner only; 0 dup eval_keys, 0 null metrics; row
   decomposition stated.

## Pre-register, before any cell

Frozen card with: predicted T-pattern (regime-2 aggregation — monotone
rise with T, the λ̂ shape), the KEEP and KILL clauses, the falsifier,
and the realized-l0 band. **State plainly what would make this a
NEGATIVE** — flat-or-falling in T, or window ≤ per-token-decoded
baselines, is a sound and publishable outcome. We want a sound verdict,
never a win.

## Variance receipts (do not skip — this is what made the λ̂ panel quotable)

Run the probe-agnostic harness
(`support_stats/stage2_variance.py`, CLI params) on this panel's results:
per-seed cells + 95 % CIs, the exact within-seed trend permutation p over
the T ladder, and the paired TXC-pre − T-SAE / − per-token margins with
sign-flip p. **Report what is bounded at n = 3 and what is not** — the
λ̂ panel's pre-vs-T-SAE margin is still formally unbounded and we say so;
do not let this panel repeat that without saying it.

## Acceptance gate — stop for review

Card frozen pre-run; panel complete with 0 failures; LOG verdict with
the scorecard (which prediction held, which was falsified); variance
receipts; figure; RECORD section; leaderboard hygiene; STATUS
rewritten. Briefing stays until mac-local review.

---

## ADDENDUM (mac-local, 2026-07-25 — scoping + the 12-hour queue)

If you froze a card before reading this, reconcile via a card-amendment
commit before any conflicting cell.

**1. The T-SAE arm is the long pole — schedule it FIRST, and use the
fresh-panel unlock.** Your own top-up measurement: tsae cells are
multi-hour (`ActivationBuffer._refill` re-gathering an 8.6 GB buffer,
GPU at 0 %), and on the λ̂ top-up shrinking `buffer_tokens` was
correctly BARRED because it changes `train_key` vs the round-1 cells.
**That bar does not exist here: this is a fresh datasource with no old
cells to match.** You may freeze a feasible `buffer_tokens` in the card
— applied UNIFORMLY to every arch — and note the value. Launch the 3
tsae trained cells at the start, in parallel with the window arms; do
not leave them to the end and do not deliver a panel without its key
baseline. A panel whose tsae arm is still running at review time is
reportable as partial; a panel that never scheduled it is not.

**2. Datasource plumbing.** Copy the `real_lambda.py` plugin pattern
INCLUDING the `trace_ids` extras (runpod-b's addition) so the v2
trace-split and the split-forensics receipt apply unchanged. oprate
coverage is 0.90 (NaN where any kernel-lag sentence is unlabeled) ⇒ the
non-finite leading-edge guard in `lambda_recovery` is LIVE on this
datasource for the first time — report how many sampled windows drop,
per T.

**3. Binding 3 clarified (Stage-2 vocabulary).** The bundle's evidence
ceilings are screen-side AUCs and do not transplant. At Stage-2 print
the REGRESSION analog: in-window event count (current tile, same
windows, same probe convention) → target, its r reported beside every
window cell. Label-side, minutes.

**4. The 12-hour queue, in order — stopping early at any gate is fine:**
1. Freeze card (incl. buffer value + realized-l0 band + evidence-line
   analog) → 2. `rate_case` full panel (84 cells; tsae first) →
3. variance receipts via the probe-agnostic harness → 4. LOG verdict +
scorecard + figure → 5. `rate_ver` panel ONLY if 1–4 are done with
real headroom. Do not start ver you cannot finish.

---

## A40 ADDENDUM (mac-local, 2026-07-25 — FORCE MAJEURE)

All pods were lost overnight (funds); you are resuming context-less on
a shared interim 6×A40 pod with ~12 funded hours and EPHEMERAL
storage. **Read `briefings/a40-bootstrap.md` BEFORE this briefing** —
it carries the box facts, your GPU ownership, the cache-rebuild-first
step, the taken methods decision (v1 canonical; report paired v2), the
push-per-batch rule, and the budget triage. This briefing's science is
unchanged; its 12-hour queue now runs on A40 timings per the
bootstrap.
