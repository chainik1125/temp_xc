# Mini-card — candidate 1: backtracking intensity λ̂ (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run; the screen has not been
executed when this card is committed — git order is the evidence).**
Agent: runpod-d. Briefing: `briefings/task-hunt.md` candidate 1.
Provenance: grounded, regime-2 prior (linear-in-window latent).

## Label definition (exact)

λ̂_i = σ(a + c·(i/L) + Σ_{l=1..8} w_l·b_{i−l}) — the **frozen fitted
mirror intensity** (`synthetic/backtracking/results/
backtracking_mirror_stats.json`: a = −2.982, c = +0.487, w_1..w_8 as
committed) evaluated on each trace's **real** Sonnet sentence labels
b (the same sequence `backtracking/measure.py::load_traces` uses),
sentences i ≥ 8 only (fully-observed history; no padding). Every token
of sentence i carries λ̂_i (char-midpoint token→sentence rule, the
`build_ward_stream.py` is_bt convention); cache-grid mapping restricted
to `map_ok`. Builder: `build_labels.py` (committed alongside; label-side
stats in `results/lambda_labels_stats.json` — no activations touched).

- **PRIMARY target: λ̂_hist** — the kernel-only intensity (position term
  dropped), top vs bottom **tercile** (cuts from TRAIN rows only;
  middle tercile dropped) — binary, frozen probe stack. *Why primary
  (decided label-side, pre-commit):* the position-only floor for the
  full λ̂ terciles measured **0.82 AUC** (the fitted +0.487·pos ramp is
  trivially readable from position alone) vs **0.59** for λ̂_hist — the
  self-excitation kernel is the latent of interest; the ramp is an
  ambient trend that would swamp the screen. Floors were computed from
  labels only (no activations), disclosed here before the freeze.
- **Secondary A (continuity):** full λ̂ terciles — kept for continuity
  with the mirror's λ; interpreted against its 0.82 position floor.
- **Secondary B (regression):** continuous λ̂_hist, linear head, same
  frozen hyperparameters (MSE loss; Pearson r on test) — disclosed
  extension of the stack for a continuous target.
- **Ambient covariate floor:** tercile classification from the scalar
  sentence-position i/L alone (no activations) — reported next to
  every AUC (measured: 0.59 for λ̂_hist, 0.82 for full λ̂).

Distinct from Aniket's detection/inducement readout: this is intensity
RECOVERY from raw activations (no steering, no inducement). No overlap
flags known at freeze time.

## Why non-ambient (regime 2/3-shaped)

λ̂_i is a deterministic function of the **previous eight sentences'**
event indicators + position — the current token's sentence label b_i is
*not* an input. A single token's marginal sees λ̂ only through (i) the
b_i ↔ history correlation (p11 = 0.44 vs base 0.12) and (ii) whatever
history the residual stream has linearized — and the conversion-depth
RECORD (§ 3) shows precisely this class of signal is **never fully
converted**: the ant_kw window margin stays open (+0.03…+0.06 AUC) at
every residual layer of both models. A T-token right-edge window reads
the recent-history sentences *directly* (is_bt is near-ambient
per-token, RECORD § 3 P4 ✓ — so a window that contains those tokens can
read their labels), i.e. the latent is **additive-in-window** over
lag-weighted sentence indicators = regime 2.

## Measured scales (from `lambda_labels_stats.json`, label-side only)

- tokens/sentence: median 16 (p25 10, p75 25, mean 19.3)
- inter-bt-event gap: **p25 = 18 tokens** (the Ward ~18-position
  intervention distance), median 43, mean 110 (bursty tail)
- kernel support: 8 sentences ≈ **130 tokens ≫ T_max = 32** — the
  tested T-range covers only lags 1–2 (w_1 + w_2 = 2.60 of the 4.12
  total kernel mass)

## Frozen predictions (per STORY.md § 7)

Screen grid: models {base, distill} × layer hs13 (resid_post L12,
mid-depth; confirmatory hs11 = L10) × T ∈ {2, 4, 8, 16, 32}, per-token
vs window-flatten vs window-mean vs within-window-shuffled linear
probes, permutation null seed 99. All predictions are on the PRIMARY
target λ̂_hist.

- **P1 (gap exists):** window-flatten AUC > per-token AUC beyond
  3 σ_null at every T ≥ 8, both models.
- **P2 (T-scaling, the money pattern):** g(T) = AUC_win − AUC_tok is
  **monotone increasing through the tested range** with the largest
  increments at T = 8→16 and 16→32 (the window starting to span the
  lag-1/lag-2 sentences at median sentence length 16), and **no
  saturation by T = 32** (kernel support ≈ 130 tokens ≫ 32). Pattern A
  rising, saturation predicted beyond the testable range.
- **P3 (decomposition):** aggregation-dominant — g_agg ≥ ½·g at every
  T (ant_kw precedent, RECORD § 3 post-hoc); g_order > 0 at T ≥ 16
  (recency weighting inside the window) but small (≤ 0.04).
- **P4 (model axis):** base ≈ distill (reader-predictability precedent;
  |Δ| ≤ 0.02). Stage 2 takes the better cell, at most two.
- **P5 (floor clearance):** the T = 16 window AUC exceeds the measured
  position-only floor (0.59) by ≥ 0.10, and the per-token probe sits
  clearly above the floor too (the ambient is_bt-correlation leak,
  expected 0.65–0.75) — i.e. the activation signal is history, not the
  position ramp re-read.

## Falsifier / kill rule (pre-registered, on the primary λ̂_hist)

KILL the candidate if ANY of:
1. g(T) ≤ 3 σ_null at all T (no window access — λ̂ is ambient or absent);
2. g(T) flat or non-growing over the whole tested range (no T-story);
3. the window probe at T = 16 fails to clear the position-only floor
   (0.59) by ≥ 0.05 AUC (the residual label collapsed to a position
   correlate — artifact);
4. the within-window shuffle + mean probes explain the entire gap AND
   g(T) growth is absent (pure static aggregation with no T-response is
   not a hunt result).
Verdict → one paragraph in `../LOG.md`; KEEP ⇒ Stage 2 per the
briefing (canonical runner, real_lm datasource, panel × T × seeds).
