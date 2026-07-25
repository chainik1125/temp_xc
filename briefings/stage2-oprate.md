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
