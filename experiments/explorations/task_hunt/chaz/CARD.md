# CHAZ_SCREEN_CARD — correction hazard with the conversion channel removed (Ward)

**Pre-registration; frozen before any cell. Pin in
`scripts/modal_chaz_screen.py` from `git rev-parse` post-push,
asserted in-container.** Overnight § 1 seed 4; design note in the
HUNT3 freeze LOG entry (~01:10). Owner mac-a; PENDING TEAM REVIEW.

## § 1 The candidate

`chaz` = sc_lambda's EXACT label (same marker events, same frozen
exponential kernel τ=3/K=8 sentences, same binning via the factory
pipeline) with ONE stricter row rule: **eligibility = the trailing
32-token view is CUE-FREE** (no marker-span token in [t−31, t];
pos ≥ 32; marker tokens additionally masked as always). At every
probed T ≤ 32 the probe therefore reads state deposited by cues ≥ 33
tokens back — sc_lambda's screen verdict ("a converted latent with an
aggregation bonus") cannot recur by construction. What survives the
restriction is the persistent-state claim, isolated.

## § 2 Label-side pre-measures (labels/build_chaz.py → chaz_stats.json)

270,292 eligible rows; manifests 20,000/class (primary + null).
Factory triage (kill authority, frozen bars): **PASS** — token-id
extreme-AUC 0.630 (kill ≥ 0.65), position extreme-AUC 0.635 (kill ≥
0.70). DISCLOSED: both sit close to their bars; the screen's
per-token-first triage and window-MEAN control are the instruments
that decide whether anything beyond those leaks exists in windows.

## § 3 Protocol + verdict rules (frozen)

Screen = `factory_screen.py` UNCHANGED (`chaz` bundle, target "");
models {base, distill} × layers {hs13, hs11} × T ∈ {2,4,8,16,32};
arms per cell: per-token FIRST (flushed before window arms),
window-flatten, window-MEAN (g_agg — the capacity control),
within-window-SHUFFLE (seed 23); permutation null σ_null (seed 99).
**CANDIDATE** iff (majority of {base, distill} at hs13): window gain
g = flatten − tok ≥ +0.05 with g ≥ 2σ_null, AND g − g_agg ≥ +0.02
(the gain is not order-free aggregation/capacity), AND the null arm
(man_null) shows no comparable gain. **KILL** if g_agg ≈ g everywhere
(≤ +0.02 apart) or g < 2σ_null everywhere or the null-arm gain
matches. Else WEAK, numbers only. Order sensitivity
(flatten − shuffle) reported; decides panel-gate vs breadth per the
house rule, KEEPs nothing alone.

## § 4 Venue, economics

1× Modal L40S, conv_depth caches from the Volume (sc_lambda-era,
expected cache-hit), `--detach`, retries 1, results JSON → Volume
/workspace/chaz_screen + repatriate; containers never push. Est
≈ $2–3 (screens ≈ $1–3 each, briefing § 1). Ledger before/after.
