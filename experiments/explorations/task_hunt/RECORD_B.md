# Task hunt — arm B (runpod-e) record — round 2

**Round 1** (three sound kills: replag, confidence trend, emotional
instability) is recorded in [`LOG.md`](LOG.md) and was REVIEWED &
APPROVED 2026-07-24; it has no separate record file. This file is the
methods + results record for arm B **round 2**
(`briefings/task-hunt-r2-e.md`): the hedging-trend LEVEL Stage-2 panel
(item 1) and the early-layer addendum (item 2).

Prime directive applies: **a sound verdict, never a win.** Freeze
order (git evidence): `card_stage2.md` §§ 1–9 at `fff7877c` → § 10
amendment at `606a8015` → first panel cell after; addendum
`PREDICTIONS.md` + `run_depth.py` at `e4caddf6` → first addendum cell
after. Diagnostics committed before their runs.

---

## § 1 — Hedging-trend LEVEL Stage 2

Card: [`confidence/card_stage2.md`](confidence/card_stage2.md)
(FRESH; the killed screen card is motivation only). Datasource:
`ward_real_slope8_distill_l14` (plugin
`src/explorations/task_hunt/real_slope.py`; R1-Distill reader,
resid_post L14 = hs15; target = the frozen `slope8` grid, NaN where
undefined — no densification; the λ probe drops non-finite
leading-edge targets, an extension proven byte-identical on all-finite
grids, `tests/test_lambda_nanmask.py`).

Panel: 5 archs × T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} + untrained =
84 cells, d_sae = 2048 = d_in/2, nominal k_pos = 8 — **TXC-post at
nominal k_pos = 8·T** (the code-rate matched-budget convention,
adopted from runpod-d's frozen amendment; this panel has no unmatched
post arm) — eval_window_L = 32, n_steps = 8000, buffer 524288, all
through `temp_bench.core.runner.run_experiment`.

**The binding readout convention (review note 1, carried verbatim):**
Stage 2 reads ONE tile's code per prediction (`lambda_recovery.py`
per-tile leading-edge convention, the same leak-free design as the
synthetic DPI bench). Per-token archs are therefore read at single
positions by construction. Any comparison sentence quoting this panel
holds **under the code-readout convention**, with the code-rate
defense: pooling T-SAE codes across T positions would spend T× the
code bandwidth a window arch uses.

### § 1a — Pre-registered references (both landed BEFORE any panel reading)

**Position-only floor** (`position_floor.py`, card § 10.2): [p, p²] of
the leading edge → held-out r = **+0.009 / +0.010 / +0.017 / +0.020 /
+0.025** at T = 1/2/4/8/16. Prediction CONFIRMED: the ambient position
ramp explains essentially nothing in the Stage-2 metric.

**Raw-activation reference** (`raw_reference.py`, card § 6) — the
session's sharpest interpretive fact:

| reference | T=1 | T=2 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| raw per-token (leading edge) | **0.221** | — | — | — | — |
| raw window-MEAN | — | 0.203 | 0.154 | 0.139 | 0.193* |

(*T = 16 is in the n < p interpolation regime, card § 6.)

**The unmatched Stage-2 sampling re-admits the ambient route the
screen's matching removed.** The Stage-1 screen measured slope8 on
rows exact-histogram matched over (anchor hedge state × position
bucket) — under that guard, per-token was near-blind (0.468 acc vs
chance 0.333) and the window-MEAN grew with T to 0.565. The Stage-2
probe samples all positions unmatched, so the anchor's own hedge
state (lexically stamped, correlated with the trailing slope) makes
the raw single position the STRONGEST raw reference (r = 0.221), and
raw pooling DILUTES it (mean falls to 0.139 by T = 8). The panel's
arch-vs-arch comparison stays internally fair (identical rows and
convention for every arch), but the screen's "per-token-blind" premise
does NOT transfer to this convention, and every reading below carries
that.

### § 1b — Panel result — **NEGATIVE (the frozen KEEP rule does not fire)**

84/84 cells ok, 0 failures. Figure
`confidence/figs/stage2_tscaling.*`, numbers
`results/stage2_summary.json`, receipts `results/stage2_stats.json`.
`lambda_recovery` = held-out Pearson r vs slope8, mean over seeds
{1, 2, 42}:

| arch | T=1 | T=2 | T=4 | T=8 | T=16 | realized l0 |
|---|---|---|---|---|---|---|
| per-token BatchTopK SAE | 0.174 | — | — | — | — | 6.3 |
| T-SAE | 0.192 | — | — | — | — | 7.2 |
| Stacked | — | 0.168 | 0.204 | 0.169 | 0.129 | 7.0–8.0 |
| **TXC-pre** | — | 0.211 | **0.229** | 0.196 | 0.132 | 7.0–7.8 |
| TXC-post (matched k = 8·T) | — | 0.206 | 0.191 | 0.141 | 0.145 | 7.0–8.1 |
| *RAW per-token (reference)* | *0.221* | — | — | — | — | — |

**Every arch is budget-matched this time** (realized l0 6.3–8.1
against the intended 8/token) — the amendment worked, and its
pre-registered falsifier passed exactly: matched TXC-post's untrained
cells realize **8.00** l0/token at every T (the k·T correction is
right; § 8 VOID does not fire).

**The verdict is NEGATIVE on the card's own second clause:** "window
recovery is flat or falling in T over {2, 4, 8}". Exact within-seed
trend permutation (`stats_lib.within_seed_trend`, pooled seeds):
TXC-pre **p = 0.727**, TXC-post **p = 0.963**, Stacked **p = 0.495** —
no arch shows a T-rise. Recovery **peaks at T = 4 and declines**. The
KEEP clause independently fails: TXC-pre clears both token archs
beyond the paired spread at **one** T (T = 4), not the required ≥ 2.

**The one real positive, stated with its bound.** At T = 4 TXC-pre
beats both per-token decoders with paired 95 % t CIs excluding zero:
**+0.055 vs the per-token BatchTopK SAE** (CI [+0.007, +0.103]) and
**+0.037 vs T-SAE** (CI [+0.012, +0.062]), all three seeds positive
(exact sign-flip p = 0.125, its n = 3 floor). That is a genuine
single-operating-point win for a window code under the code-readout
convention. It is not the hunt's pattern, which requires the advantage
to GROW with T.

**The sharpest honest fact — the codes barely beat raw activations.**
The pre-registered raw per-token reference is **r = 0.221**. Exactly
one of the 14 panel cells exceeds it (TXC-pre/T4 at 0.229); every
other cell — including both token archs (0.174, 0.192) — sits below
what a linear probe reads off the raw residual at a single position.
So on this task no architecture in the panel is buying much over the
raw stream, and the arch-vs-arch differences live inside that band.

**Blind-prediction scorecard (card § 7, scored either way):**
- **P1 FALSIFIED as a conjunction.** Its second clause holds (TXC-pre
  exceeds both token archs beyond spread at T = 4); its first clause —
  recovery rises with T through T = 8 — is false (p = 0.73).
- **P2 FALSIFIED.** Matched TXC-post does not rise (p = 0.96, falling)
  and is below TXC-pre at T = 8 (0.141 vs 0.196); it is nominally ≥ at
  T = 16 (0.145 vs 0.132), inside the spread. Aggregation being post's
  "native shape" is not what the data shows.
- **P3 FALSIFIED.** Token archs do not land near the raw per-token
  reference — they land **below** it (0.174/0.192 vs 0.221) — and at
  T = 16 they sit **above** every window arch.
- **P4 PARTIALLY CONFIRMED.** TXC-pre's trained − untrained margin
  grows +0.106 → +0.132 → +0.135 across T = 2/4/8 then falls to
  +0.085 at T = 16; TXC-post's is flat (+0.089/+0.094/+0.078/+0.087).
  So the learned T-dependence exists for pre through T = 8 even though
  absolute recovery does not rise.
- **P5 FALSIFIED.** No monotone rise through T = 8.
- **Stacked pathology RECURS** (named as a risk, not predicted): at
  T = 16 trained 0.129 sits **below** untrained 0.157 (margin −0.029),
  the same large-T failure the λ̂ panel recorded. Not evidence for or
  against any arch.

### § 1c — Shuffle-immunity receipt (card § 10.1) — **DEGENERATE, reported not spun**

`shuffle_receipt_stage2.py` on the panel's own checkpoints (12 cells,
anchor-fixed context shuffle, probe refit on shuffled codes), seed
means:

| arch | T | clean r | shuffled r | drop | retained |
|---|---|---|---|---|---|
| TXC-pre | 8 | 0.195 | 0.173 | +0.022 | 89 % |
| TXC-pre | 16 | 0.133 | 0.094 | +0.039 | 70 % |
| TXC-post | 8 | 0.141 | 0.126 | +0.015 | 89 % |
| TXC-post | 16 | 0.145 | 0.079 | +0.066 | 54 % |

**The pre-registered test cannot be scored as written, and that is the
honest report.** The card's criterion was "the shuffled cell keeps
more than half of that cell's (clean window − best token arch)
margin". At the receipt's frozen cells (T ∈ {8, 16}) that margin is
≈ 0 or negative (pre/T8: 0.195 − 0.192 = +0.003; every T = 16 cell is
below both token archs), so there is no margin left to retain and the
ratio is undefined or meaningless. The receipt was designed to
interrogate a margin the panel did not produce.

What the raw numbers do say, reported as a descriptive result: at
T = 8 both window archs retain ~89 % of their recovery under
context shuffling — consistent with the order-free aggregation claim —
while at T = 16 the drop is larger (30–46 %), in the same cells the
probe-capacity caveat (§ 6) already flags as the least trustworthy.
**No order claim and no immunity claim is made from this receipt.**

### § 1d — Probe-capacity diagnostic (card § 6, pre-registered) — **the T-decline is a PROBE artifact**

`probe_capacity.py`, seed 1, OFF-leaderboard, post-hoc by
pre-registration. Same trained checkpoints, same tiles; only the probe
changes. The `nw1024 / OLS` column **reproduces the panel exactly**
(0.210 / 0.134 / 0.163 / 0.167 vs the panel's 0.2102 / 0.1338 / 0.1627
/ 0.1671), so the comparison is trustworthy:

| cell | panel probe (nw1024, OLS) | nw1024 + ridge | nw8192 + OLS | nw8192 + ridge |
|---|---|---|---|---|
| TXC-pre T4 | 0.210 | 0.302 | 0.248 | 0.274 |
| **TXC-pre T16** | **0.134** | **0.324** | 0.246 | 0.311 |
| TXC-post T4 | 0.163 | 0.256 | 0.238 | 0.255 |
| **TXC-post T16** | **0.167** | **0.318** | 0.258 | 0.294 |

Every panel cell above has **negative held-out r²** (−0.24, −1.11,
−0.33, −0.95) — textbook overfitting at n ≈ p on a dense code, since
`lambda_recovery` fits an unregularized OLS on p = d_sae = 2048
features while n shrinks as 1/T.

**Consequence, stated as the card allows.** The frozen NEGATIVE verdict
(§ 1b) **stands under the frozen metric** — the card pre-registered
that this diagnostic "cannot change the leaderboard cells; it can only
change what the record is allowed to claim about them." What the record
may now claim is narrower and more useful: **the panel's T-decline is
an artifact of the evaluator's probe, not a property of the
representations.** Under ridge on identical codes the ordering
reverses — T16 ≥ T4 for both window archs (pre 0.324 vs 0.302; post
0.318 vs 0.256). The panel as specified **could not have detected a
T-rise even if one existed**, because the probe's bias grows with T.
The honest summary of item 1 is therefore: *no T-rise is demonstrated,
and this panel design cannot demonstrate one.*

**Independent corroboration (converging, not coordinated).** runpod-d's
round-2 λ̂ amendment reached the same conclusion on a different task
and datasource the same day (LOG, "reading (c) CONFIRMED — panel-wide
probe artifact"): lifts of +0.18…+0.23 on dense T16 cells, negative
r²_eval at nw1024/OLS, and the lift scaling with nnz-per-row. Two
independent Stage-2 panels, two different real tasks, same defect.
**Recommendation to the program (mine and runpod-d's, arrived at
separately): `lambda_recovery` should regularize (or scale n with T)
before any further T-scaling claim is drawn from it, and the existing
λ̂ money plot's "peaks rather than saturates" reading should be
re-examined under an adequate probe.**

---

## § 2 — Early-layer addendum (`depth_addendum/`, PREDICTIONS.md frozen pre-run)

Zero new data: cached activations, frozen round-1 manifests, frozen
problib stack; screen conventions verbatim; screen-layer overlap cells
double as reproduction checks (they reproduce the committed screen
JSONs exactly — e.g. gpt2 hs7 lag4 tok 0.515 / T4 win 0.500 / mean
0.430). Off-leaderboard diagnostic. Null calibration at the new
layers: gpt2 hs4 T4 permuted-label window probe = 0.2503 (chance
0.25) — the floor transfers.

### § 2a — Replag arm: lag4 across depth (COMPLETE)

acc (4-class, chance 0.25), matched rows identical across layers:

| model | hs | tok | T4 win / mean / g_ord / shufdrop | T8 win / mean / g_ord / shufdrop |
|---|---|---|---|---|
| gpt2 | 4 | **0.631** | 0.591 / 0.456 / **+0.135** / +0.009 | 0.538 / 0.426 / +0.112 / +0.054 |
| gpt2 | 7 (screen) | 0.515 | 0.500 / 0.430 / +0.070 / +0.011 | 0.477 / 0.399 / +0.078 / +0.048 |
| gpt2 | 10 | 0.433 | 0.433 / 0.390 / +0.043 / +0.019 | 0.423 / 0.375 / +0.048 / +0.062 |
| gemma2-2b | 8 | 0.505 | 0.457 / 0.393 / +0.064 / +0.008 | 0.464 / 0.352 / +0.112 / +0.035 |
| gemma2-2b | 14 (screen) | 0.462 | 0.424 / 0.384 / +0.040 / +0.011 | 0.435 / 0.350 / +0.085 / +0.041 |
| gemma2-2b | 20 | 0.387 | 0.378 / 0.347 / +0.032 / +0.019 | 0.391 / 0.312 / +0.079 / +0.048 |
| llama31-8b | 8 | 0.480 | 0.503 / 0.398 / **+0.105** / +0.013 | 0.493 / 0.351 / **+0.142** / +0.037 |
| llama31-8b | 14 (screen) | 0.430 | 0.445 / 0.361 / +0.083 / +0.012 | 0.444 / 0.324 / +0.120 / +0.034 |
| llama31-8b | 22 | 0.365 | 0.394 / 0.332 / +0.061 / +0.007 | 0.413 / 0.322 / +0.091 / +0.047 |

**Blind-prediction scorecard (PREDICTIONS.md, committed pre-run):**

- **A1 FALSIFIED — and the falsification is the finding.** Per-token
  lag4 does not fall toward the input; it is **highest at the earliest
  layer in all three models** (gpt2 0.631 → 0.515 → 0.433 with depth;
  gemma 0.505 → 0.462 → 0.387; llama 0.480 → 0.430 → 0.365). The
  lag VALUE is maximally linearly readable near the embeddings and
  **monotonically discarded** with depth — a fifth g(ℓ) shape for the
  atlas: **present-then-discarded** (vs backtracking's
  converted-early, forbidden-word's built-then-linearized). The
  temporal signal doesn't need building here; the model progressively
  throws it away.
- **A2 CONFIRMED (all three models, both T):** g_order = win − mean is
  larger at the early alternate than at the screen layer everywhere
  (T4: gpt2 +0.135 > +0.070 > +0.043; gemma +0.064 > +0.040 > +0.032;
  llama +0.105 > +0.083 > +0.061), and the gpt2 magnitude clause
  (early ≥ +0.10) holds.
- **A3 CONFIRMED (sharpened):** the round-1 scale ordering
  (gpt2 ≫ 2B/8B) closes at early depth — llama's early-layer order
  signal (+0.105 T4 / +0.142 T8) rivals or exceeds gpt2's. The scale
  gap was a property of mid-depth conversion strength, not of the
  models' inputs.
- **A4 CONFIRMED:** late layers stay at or below screen-layer g_order
  in every model.
- **A5 REFINED — a decomposition, not a confirmation.** At T = 4 the
  large g_order coexists with a near-zero anchor-fixed shuffle drop
  (gpt2 hs4: g_ord +0.135, shufdrop +0.009): most of "g_order" at
  short T is **anchor-vs-context separation** (the flatten probe
  privileges the anchor slot over a bag of context), not context
  ORDER. True context-order signal appears at T = 8 (drops
  +0.035…+0.062, tracking depth). g_order = flatten − mean conflates
  the two; the anchor-fixed shuffle isolates the second. Future cards
  should read both.
- **Scope note:** win − tok stays ≈ 0 or negative at every depth and
  T except llama early (+0.023 T4, inside the round-1 3σ band) — the
  round-1 KILL (no window advantage over per-token for replag) holds
  depth-wide, exactly as the forbidden-word depth sweep found for its
  label. The addendum's growth findings are about the ORDER COMPONENT
  and the per-token axis, not a window win.

### § 2b — Slope8 arm: g_agg across the 17 Ward capture points (COMPLETE)

Figure `depth_addendum/figs/depth_slope8_gagg.*`. acc (3-class, chance
1/3), matched rows identical across all 34 (reader, layer) cells;
permutation null at a NEW layer (distill hs1, T64 mean) = 0.345 ≈
chance. Selected cells (full grid in `results/depth.json`):

| reader | hs0 (emb) | hs1 | hs5 | hs11 | hs15 (screen) | hs25 | hs31 |
|---|---|---|---|---|---|---|---|
| distill tok / mean64 | 0.368 / 0.496 | 0.450 / 0.517 | 0.474 / 0.548 | 0.483 / 0.520 | 0.468 / 0.565 | 0.450 / 0.564 | 0.450 / 0.561 |
| distill **g_agg** | **+0.128** | +0.067 | +0.074 | **+0.037** | +0.096 | **+0.113** | +0.111 |
| base **g_agg** | +0.111 | +0.100 | +0.055 | +0.043 | +0.043 | +0.043 | +0.054 |

**Blind-prediction scorecard:**

- **B1 CONFIRMED:** g_agg > 0 at every one of the 34 cells — the
  aggregation gap is present from the EMBEDDINGS (+0.128 at hs0, where
  the pooled lexical hedging evidence is purest and per-token collapses
  to 0.368). Not a late-depth phenomenon; the early clause (hs1/hs3
  within factor ~2 of hs15) holds.
- **B2 CONFIRMED — the WHY of the Stage-2 bet:** per-token slope8
  never exceeds 0.483 at ANY depth on either reader — no layer holds a
  per-position trend summary. The mean-vs-tok gap exists depth-wide,
  so the panel's L14 choice was representative, not lucky. This is the
  anti-conversion signature: the model never converts the trend, at
  any depth — the aggregate stays window-only.
- **B3 CONFIRMED:** the readers' g_agg agree at hs0–3 (differences
  0.01–0.03, mixed sign) and diverge in the second half of depth —
  distill rises to +0.10…+0.11 at hs23–31 while base sits at
  +0.04…+0.08. The generator's surplus is a late-depth phenomenon.
- **B4 FALSIFIED — and the shape is the finding.** Predicted
  flat-to-rising into a MID-depth peak (hs13–17); measured: a
  mid-depth **valley** (+0.033…+0.037 at hs11–13), with maxima at the
  embeddings (+0.128) and LATE (+0.113 at hs25). The hedging-arc
  aggregate is strongest where the stream is closest to tokens — at
  the input (lexical stamps) and near the output (the generator's own
  next-token hedging behavior) — and weakest at the abstract middle.

**The addendum's answer to the briefing's question** (does the
temporal signal GROW at pre-conversion depths?): YES on both arms, in
two different senses. For lag4, everything grows toward the input —
per-token most of all (§ 2a A1, present-then-discarded), with the
order component g_order also largest early and the round-1 scale
ordering closing. For slope8, the aggregation gap needs no depth at
all (present at hs0) and the generator-specific surplus grows LATE.
Neither arm moves any round-1 verdict; the tested T-range obligation
and conversion-atlas additions go to the LOG.

---

## § 3 — Methods notes

1. **Finite-target mask instead of label densification.** slope8 has
   no generator whose warm-up convention could densify it (a shortened
   trailing fit is a different statistic), so the frozen grid keeps
   NaN and `lambda_recovery._train_lambda_probe` drops non-finite
   leading-edge targets — guarded so all-finite paths (every synthetic
   bench, the λ̂ datasources) are byte-identical; equivalence + drop
   behavior tested in `tests/test_lambda_nanmask.py`. Probe-row counts
   per T under this mask are a pre-run geometry fact (card § 6):
   27143/13579/6788/3393/1702 train rows at T = 1/2/4/8/16 vs
   p = 2048 — T = 16 is in the interpolation regime.
2. **Reader choice is screen-grounded.** The monotone T-growth that
   motivates a T-scaling panel exists only on the generator (distill
   0.521→0.565 vs base 0.525→0.511 flat), so the panel reads distill;
   mirroring the λ̂ panel's *reasoning* (screen's primary cell becomes
   the datasource), not its layer.
3. **Draft-card reconciliation happened pre-run** (card § 10):
   runpod-b's `LEVEL_CARD_DRAFT.md` landed while the card was being
   frozen. Adopted: the Stage-2 shuffle-immunity receipt, the
   position floor, the bottom-of-T-range record obligation. Rejected
   with reason: the draft's per-T window-mean-level primary (label
   would vary with T, confounding target with architecture axis;
   slope8's support is fixed at ≈ 128 tokens for every T, the λ̂
   design shape).
4. **Code-version stamps:** the tree was clean at panel launch; cells
   completing later stamp `dirty=True` with pinned `diff_sha256`
   because the results JSON + leaderboard accumulate in-tree during
   the run — the same accepted pattern as the round-1 λ̂ panel.
5. **Duplicate-launch near-miss (recorded, no data impact).** A
   backgrounded shell chain masked a failed `git push` behind a
   pipeline (`| tail`), so a second addendum launch briefly ran beside
   the first; the newer process was killed within seconds, the
   incremental JSON parses clean, and all cells are deterministic
   (identical manifests + seeds), so recomputation cannot diverge.
   Lesson: never gate a launch on a piped git command's exit status.
