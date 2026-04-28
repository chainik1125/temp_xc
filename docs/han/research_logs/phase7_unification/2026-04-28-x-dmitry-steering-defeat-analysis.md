---
author: Han
date: 2026-04-28
tags:
  - results
  - in-progress
---

## Why TXC "loses" at steering — Dmitry's branch + Agent X analysis

> Logged by Agent X for cross-agent visibility. Y is now URGENT-priority
> on this; Z should also know (it may affect what counts as a
> hill-climb winner). Content adapted from Dmitry's
> `docs/dmitry/case_studies/rlhf/{summary,notes/methodology}.md` on
> branch `origin/dmitry-rlhf` (commits `0550f2f..229d1f0`).

### The reversal

Agent C concluded "TXC family Pareto-dominates per-token archs at
moderate-strength steering" using AxBench-additive at strengths
{0.5, 1, 2, 4, 8, 12, 16, 24}. Dmitry has now reproduced the T-SAE
paper's own protocol (clamp-on-latent + error preserve, App B.2)
at strengths up to 15000, with the same 6-arch shortlist. Headline:

> **T-SAE k=20 wins peak success on both protocols** (1.93 paper /
> 2.00 AxBench, of 3). Window archs (TXC, SubseqH8, H8) lag T-SAE by
> 0.5–0.8 under paper-clamp; close to within 0.3 under AxBench.

So Agent C's claim was **correct under one protocol, wrong under the
other**. The architectural ranking changes when the intervention
modality changes.

### The clean analytical why

Dmitry's `methodology.md` derives it. The two protocols apply different
nominal-strength → real-intervention maps:

| | paper-clamp | AxBench-additive |
|---|---|---|
| net injected direction | `(s − z[j]_orig) · W_dec[:, j]` | `s · unit_norm(W_dec[:, j])` |
| magnitude depends on `‖W_dec‖`? | yes | no (unit-norm) |
| magnitude depends on `z[j]_orig`? | yes (per-token) | no |
| at `s = z[j]_orig`, intervention is | a no-op | (irrelevant; no encode round trip) |

For window archs, the SAE encoder takes a `(T, d_in)` window and
integrates linearly over T positions. Active-feature magnitudes
`z[j]_orig` are therefore ≈ T × per-token magnitudes, i.e., 5–10× larger
than for per-token archs at typical T = 5–10.

Under paper-clamp the strength `s` is an absolute clamp value, so a
clamp of `s = 100` corresponds to:

- per-token archs (`z[j]_orig` ≈ 5–10): a 10× push above typical activation
- window archs (`z[j]_orig` ≈ 25–50): only a 2× push

**The "peak strength" of each architecture's steering response shifts
proportionally to its activation magnitude**. Empirically (Dmitry's
30-concept × 6-arch sweep): per-token archs peak at s=100, window
archs peak at s=500 — exactly the predicted 5× ratio.

Under AxBench-additive the unit-norm decoder direction removes the
magnitude dependence entirely; all archs peak at the same s.

The paper's authors didn't run into this because they only compared
per-token archs (TopK SAE + T-SAE). Cross-family comparison at
absolute strengths confounds "feature quality" with "activation
magnitude scale".

### What this means for the paper

Three plausible framings, in increasing order of how much we have to defend:

1. **Both protocols are reported, both winners noted**: T-SAE k=20
   under paper-clamp; TXC family under AxBench-additive at moderate
   strengths. Caveat: protocols measure different things; reader picks.
   Cleanest framing; survives reviewer scrutiny well.
2. **AxBench-additive is the canonical comparison protocol**:
   defensible because it's the only protocol that doesn't conflate
   activation-magnitude with feature-quality. Counter-defence: it's
   not the T-SAE paper's own protocol, so direct comparison to paper
   numbers needs an asterisk.
3. **Paper-clamp at per-family-normalised strengths**: rescale `s` by
   `⟨z[j]_orig⟩_arch` before applying clamp. Lets the paper's own
   protocol be used while removing the family-scale bias. Currently
   untested — Q1.3 in Y's URGENT agenda.

Han's call which framing the paper takes. Y's job is to provide the
empirical inputs (Q1.1, Q1.2, Q1.3 in `agent_y_brief.md`).

### Implications for Z (hill-climb)

If the magnitude-scale story holds (which we expect it does), a
hill-climb winner that only beats existing TXCs on **probing AUC** is
incomplete — the winner should ALSO be steerable. Two options:

- (a) Z's hill-climb continues focused on probing AUC (current plan);
  any winner gets a follow-up steering check post-hoc.
- (b) Z explicitly biases hill-climb candidates toward variants whose
  feature magnitudes are closer to per-token scale (e.g.,
  per-position normalisation in the encoder; learned per-feature
  rescaling).

I'd default to (a). Steering performance is downstream of training; if
the magnitude scale really is the only difference, then a normalised
strength schedule fixes it post-hoc without retraining. (b) is more
ambitious and slower.

Z agent should at minimum read this log so they know that "highest k=5
AUC" isn't the only metric the paper cares about going forward.

### Implications for X (leaderboard / T-sweep)

Mostly none — X's deliverables are sparse-probing AUC, not steering.
But X should be aware that the paper's narrative will likely include
a steering caveat, and Han may want a "Pareto plot at AxBench-additive
+ normalised paper-clamp" figure that's adjacent to the leaderboard.
That's a Y output, not an X output, but X may need to schedule
re-probing on certain Y-requested cells (e.g., if Y's normalised
paper-clamp picks features that don't appear in the current
`results/case_studies/steering/<arch>/feature_selection.json`,
re-running the feature selector might require X-side ckpts that
haven't yet been pulled).

### Files of record

- Dmitry's branch: `origin/dmitry-rlhf` (commits `0550f2f..229d1f0`).
- Cherry-picked onto unification (this commit's preceding edits):
  - `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp.py`
  - `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window.py`
  - `experiments/phase7_unification/case_studies/steering/intervene_axbench_extended.py`
- Outputs (not committed; gitignored on Dmitry's pod): per-arch
  `generations.jsonl + grades.jsonl` at
  `experiments/phase7_unification/results/case_studies/steering_paper/<arch>/`
  and `…steering_axbench_extended/<arch>/` on Dmitry's pod
  (`a40_txc_1`).
- Dmitry's writeup: `docs/dmitry/case_studies/rlhf/` on
  `origin/dmitry-rlhf`. Stays there — not migrated to
  `docs/han/research_logs/`.
