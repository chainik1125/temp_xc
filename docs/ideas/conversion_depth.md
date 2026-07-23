# Conversion depth — where in the network is a TXC worth anything?

**Idea (2026-07-23, from the ambience discussion).** Attention *converts*
temporal structure into per-position linear structure: once a property of the
recent stream has been attended-into the current token's residual, a
per-token SAE reads it and a temporal dictionary has nothing left to add. So
a TXC's usable advantage lives in the **unconverted gap** — and that gap
should shrink with depth. Hypothesis: **temporal dictionaries are most useful
at early layers / interfaces where the model has not yet converted the
target structure**, and the advantage at a given layer is predicted by the
measured ambience gap there.

## Evidence in hand (suggestive, not systematic)

- **GPT-2 day-stride (FreqBench sprint § 4.7):** at the embedding layer
  (`hs=0`) the stride latent is non-ambient (single-position probes at
  chance 0.149; window dictionaries convert it to linear at 1.000) — by
  **block 3 ONE attention pass has linearized it at every position** (single-
  position probe 1.000; mid-layer dictionary comparisons become vacuous).
  Position 0 stays at chance at all depths (causal control). This is a
  position × layer *conversion map* — the prototype of the ablation below.
- **Backtracking (paper, L10):** the anticipatory signal at L10 is evidently
  not fully converted — windows placed upstream of the sentence add
  detection margin, and the sprint's spectral arm found the signal
  low-frequency (DC branch best). Consistent with a partially-open gap at
  L10 for this property.
- **EM detection:** the paper reports TXC variants underperforming T-SAE on
  Wang-style steering + sparse-probe PR-AUC at L15 (no shuffle ablation in
  the paper itself); the repo's INTERNAL detection experiment measured
  trained-code shuffle_gap ≈ 0 there, which we glossed as "the property is
  ambient at the probed layer." **That gloss was falsified by the phase-4
  depth sweep** (see UPDATE below) — the raw window-over-token access is
  real at L15 and peaks mid-depth; the TXC's loss is a
  realization/readout story, not absence of temporal signal.

## The proposed ablation (not yet run anywhere)

**Conversion-depth curves.** For a fixed temporal property (start:
backtracking anticipation; also EM as the expected-flat control), sweep the
probe layer ℓ and measure at each ℓ, on raw activations (no dictionaries —
this is the § 8 ambience machinery pointed at depth):

1. raw per-token linear ceiling for the property at ℓ;
2. raw window linear ceiling (fixed T, same protocol);
3. nonlinear presence check (MLP), so blindness ≠ absence.

The **ambience gap** g(ℓ) = (window − per-token) ceiling traces where the
model itself converts the property. Predictions: g(ℓ) shrinks with depth
(monotone-ish, task-dependent); a trained TXC's detection advantage over the
per-token SAE at layer ℓ **tracks g(ℓ)**; for EM, g(ℓ) ≈ 0 everywhere. If
the tracking prediction holds, "which layer should a TXC live at" gets a
cheap pre-training answer — the depth analogue of the ambience rule.

**UPDATE (2026-07-23, ablation run — `experiments/explorations/
conversion_depth/RECORD.md`, reviewed):** the monotone-shrinking template
survives ONLY for the lexical/day-stride case (GPT-2: converted by block
1). Grounded labels show two other shapes: backtracking anticipation is a
**flat, never-converted plateau** (+0.03…+0.06 AUC at every residual
layer of both 8B models — mostly order-free aggregation; L10 fine but
not special; the signal is reader-predictability, base ≈ generator, with
a ≤ +0.02 generator margin only in the last third of depth), and EM
misalignment is an **inverted-U** (peak +0.13 at L13, +0.097 at the
paper's L15, with a real position-sensitive slice g_order +0.11 at L13 —
the § 5.3 "ambient ⇒ no window advantage" reading is falsified across
depth). The g(ℓ) machinery itself validated cleanly; the TXC-tracking
test is the open follow-up, with the EM g_order slice the strongest
candidate for a position-aware architecture win.

## Compute + status

Needs multi-layer activation caches from the subject models (the paper's
caches are single-layer): 8B-model caching = the first genuinely GPU-worthy
task in the program (A40-class suffices; inference only). A GPT-2-scale
replica of the full curve is CPU-cheap and could ride the sprint's day-stride
code (`bt_freq.py` / `gpt2_stride.py` in the sprint tree). Status: **idea —
no briefing queued.** Candidate for a future real-side cycle; it is the
depth-resolved complement of the ambience rule now in the synthetic README
(§ "The two generators" / checklist item 8).
