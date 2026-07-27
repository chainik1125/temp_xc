# RDENS_CARD — referential-density TREND on Ward (gen-4 seed 3; factory venue)

**Pre-registration. Frozen BEFORE any screen cell; pin in
`scripts/modal_rdens_screen.py` from ORIGIN-history rev-parse
post-push, asserted in-container.** Gen-4 directive 59ad15f38 (seed
idea 3, "referential-density trend on Ward") scaled by c1c5c949e;
runs beside the HUNT4 dialogue screens (freeze 35d20e3cb) as the
slate's 7th design. Owner mac-a. Verdict PENDING TEAM REVIEW.

## § 1 Construction (builder `labels/build_rdens.py`, artifact committed)

Face `rd` = kernel-WLS SLOPE (hunt house kernel: support 64 tok,
HL 16) of the REFERENTIAL-token indicator over each Ward trace —
"anaphoric load rising/falling". Lexicon FIXED pre-registered
(REF_WORDS in the builder: 3rd-person pronouns + possessives/
reflexives + demonstratives + wh-relatives; 89-word-form → vocab-id
set via llama31 decode; measured token rate **2.24%** — a sparse,
marker-free event class per the recipe). The TREND deliberately, not
the level: chaz proved Ward ambient LEVELS are pooling-readable.

Frozen factory pipeline verbatim (sc_lambda/chaz conventions):
within-trace-shuffle null (seed 211 + trace); mask = current token
referential (zero-distance give-away, the is_marker_tok analogue) +
pos < 64 (full support) + ~valid; `bundle_core` binning/manifests/
by-trace split/triage; manifests capped 6000/class post-triage (chaz
OOM lesson; triage on the FULL manifest).

## § 2 Label-side receipts (measured BEFORE this freeze; `labels/rdens_stats.json`)

- **Triage PASS** (kill authority): tok-extreme 0.583 (< 0.65),
  pos-extreme 0.573 (< 0.70); n_test 7832; 6000/class manifests.
- **Visible-floor evidence lines** (truncated-support slope + rate of
  the same indicator, rank AUC top-vs-bottom, test ext rows):

  | T | slope floor | rate floor |
  |---|---|---|
  | 8 | 0.513 | 0.661 |
  | 16 | 0.574 | 0.694 |
  | 32 | **0.792** | 0.585 |

  Pre-registered reading: **claimable zone T ≤ 16** — at T32 the
  in-window slope substantially reproduces the face (support
  min(32,64) covers half the kernel). Any T32 row is run-not-claim.

## § 3 Screen + KEEP/KILL (frozen)

Venue: `factory_screen rdens - <dir>` — protocol of record
UNCHANGED: per-token first, flat window probes at T ∈ {2,4,8,16,32},
**window-MEAN g_agg (the capacity control that killed chaz — the
named deciding instrument here)**, shuffle arm, σ_null; layers hs13
primary (hs11 expected ABSENT on the ward volume — chaz venue limit,
disclosed now; base model only, no distill cache).

KEEP iff (factory § rules of record): real-arm g beats the per-token
baseline AND g > g_agg at some claiming T ≤ 16 AND clears σ_null,
AND the label-side floor at that T does not exceed the probe (§ 2
lines; slope+rate floors are IN the bundle npz for the scorer).
KILL if g_agg ≥ g at every T ≥ 8 (the chaz clause), or σ_null
swallows the margin, or the § 2 floor does. Else WEAK. Bundle
verdict single-model (base/hs13) — venue limits quoted in any
verdict line, as with chaz.

## § 4 Economics / ops

1× L40S, est ~$1.5–2.5 (chaz actuals precedent), detached, retries
1; result JSON → Volume `/workspace/rdens_screen` + repatriate;
containers never push; ledger line before launch; hunt envelope
$200 (c1c5c949e). Prior expectation stated honestly: the chaz
precedent says Ward density-family faces die to g_agg — a clean
KILL here is a first-class outcome closing gen-4 seed 3.
