# Task-hunt log (shared, append-only)

The shared verdict log for the task hunt (`briefings/task-hunt.md`,
`briefings/task-hunt-b.md`). **Append-only; pull-rebase before every
push.** One entry per screen/stage event: date, agent, candidate,
verdict (KEEP / KILL / infeasible), one-paragraph justification, pointer
to the frozen card + results.

Conventions: mini-cards are committed BEFORE their screen runs; every
verdict cites its card; scale-orderings and honest kills are findings,
not failures (prime directive: a sound verdict, never a win).

---

## 2026-07-24 — runpod-d — candidate 1 (backtracking λ̂ intensity) — CARD FROZEN, screen pending

Ward stream + both reader caches rebuilt on the runpod-d volume
(`build_ward_stream.py` verbatim; stream stats reproduce the committed
reference — map_ok 99.97 %, BOS every row; base 219 s / distill 238 s
cache sweeps). λ̂ labels built locally
(`lambda_intensity/build_labels.py`) from the frozen mirror params on
real Sonnet history — runpod-b's `labels/ward_lambda.npz` had not
landed when caches were ready (never-idle rule); it landed mid-session
and the two builds **cross-validate: 99.93 % exact agreement** on
overlapping finite cells, disagreements confined to 277
sentence-boundary token attributions + their padded i<8 sentences
(excluded by my frozen i≥8 rule). Proceeding on my builder (the card's
freeze target); theirs stands as independent confirmation. Label-side
floor check (pre-freeze, disclosed in card): position-only AUC 0.82 on
full-λ̂ terciles vs 0.59 for kernel-only λ̂_hist ⇒ **primary target =
λ̂_hist**. Card: `lambda_intensity/card.md`, frozen at this commit;
screen (`screen.py`) not yet executed on any activation cache.

## 2026-07-24 — runpod-e — replag labels built (pre-screen); one card amendment

Δ labels built inline (runpod-b's parallel builder had not landed;
per-briefing fallback). `replag/build_labels.py` + committed manifests/
stats under `labels/`. All 5 sanity tests pass on all three tokenizers;
every task saturates its caps (4000/1500 per class) under joint
(token-id × position-bucket) matching; realized coverage is the exact
ladder (cov(B,T) = 0 until T crosses the bucket's lower edge).

**Label-stage finding + card amendment (BEFORE any screen ran):** the
card's parenthetical prior "real text far heavier at small Δ than the
shuffled null" is FALSE at the token level — real P(Δ∈[2,4]) ≈ 0.026 <
null ≈ 0.031 on all three tokenizers (grammar avoids near-repetition;
the exchangeable shuffle clumps). Sanity T4 amended from a directional
assert to a two-sided divergence test (TV > 0.02; realized ≈ 0.03) with
direction recorded as a finding. No screen prediction (P1–P5) depends
on this; the KEEP/KILL rules are untouched.

## 2026-07-24 — runpod-e — replag screen: escalation note (mid-screen, pre-run)

gpt2 + gemma base grids are in; llama running. Invoking the CARD's
pre-authorized escalation for **lag4 only**: the linear window pair is
blind (win ≤ tok at every T, both models) while the T∈{8,32} MLPs
separate (gpt2 T8: win_mlp 0.634 vs shuf_mlp 0.522 ≈ tok_mlp 0.520) ⇒
extending lag4 win_mlp/shuf_mlp to all T ∈ {2,4,16} on all three
models for the money plot (`REPLAG_ESCALATE_LAG4=1`, resumable cells;
no other grid change). Detection targets do NOT meet the escalation
condition (their MLPs show no positive gap) and stay at the frozen
T∈{8,32} presence checks.

## 2026-07-24 — runpod-e — candidate 1 (repetition-lag Δ) — **KILL** (converted at every scale)

Screen executed exactly per the frozen `replag/CARD.md` (results:
`replag/results/screen_{gpt2,gemma2_2b,llama31_8b}.json`; figs:
`replag/figs/`). Verdict by the frozen falsifier: **KILL — no model
shows the P1 detection ladder in any probe pair.** The per-Δ turn-on
never happens because there is nothing left to turn on: per-token
linear probes already read repetition detection at 0.74–0.97 AUC
(det4→det32, all three models) and the window−token gap is ≤ 0 at
every T everywhere (fig `det_gap_vs_T.png`) — induction conversion is
saturated from 124M up, so the latent is regime-1 on natural web text
at every screened scale. Larger windows actively LOSE linear AUC
(fixed probe budget over growing dims — a screen limitation noted, but
irrelevant to the verdict since the MLP presence checks also show no
positive detection gap).

Recorded findings on the pre-registered axes: **P2 falsified in an
interesting direction** — per-token ceilings are HIGHEST in gpt2
(det4 0.965 vs llama 0.923, gemma 0.893; cross-model comparisons
approximate — different tokenizers/matched sets): repetition features
occupy proportionally more of a small model's residual. **P4
confirmed** — the lag-VALUE readout (lag4) is genuinely order-carried:
window MLP beats per-token and the context-shuffle collapses it back
(gpt2 T8: 0.634 → 0.522 shuffled ≈ 0.520 per-token; fig
`lag4_order_vs_T.png`), and detection is aggregation-shaped (shuffle
≈ ordered), the predicted dissociation. **Scale ordering of the
unconverted (order) residue: gpt2 +0.11 ≫ gemma +0.02 ≈ llama +0.02**
— the briefing's frozen prior holds on the value readout, but at
panel-relevant scales (2B/8B) the residue is too thin to carry a
Stage-2 TXC-vs-T-SAE separation. No Stage 2. A sound kill: the
screen's cost was ~3 h; the mechanism (conversion) is the documented
reason, with the order-residue scale series as the reusable finding.

Next per briefing: candidate 2 (confidence trend) — runpod-b's clock
bridge does NOT kill it (slope4 support ≈ 64 tok = T64), so it
precedes candidate 3.

## 2026-07-24 — runpod-e — candidate 2 (confidence trend) — **KILL** (real T-growing window gap, but aggregation-carried — order receipt fails)

Screen per the frozen `confidence/CARD.md` + its committed screen-cell
appendix (results: `confidence/results/screen_{distill,base}.json`;
substrate: Ward stream rebuilt on this volume, readers base + distill
at hs15 = resid_post L14; slope rows exact-histogram matched on anchor
hedge state × position bucket — the ambient-route guard).

What the screen found (3-class, chance 0.333): the trend IS
window-readable and per-token-blind — distill slope8: per-token
0.468 lin / 0.503 MLP vs window-MEAN 0.521 → 0.545 → 0.565 at
T 16/32/64, a clean monotone T-growth with gap +0.06 – +0.10 at T64;
permutation nulls at chance; the hedge-state CONTROL is regime-1
exactly as predicted (per-token ≈ window, both readers). But the KEEP
clause's order receipt FAILS: the order-free MEAN probe achieves the
entire gap (g_order = flatten − mean ≤ 0 on the primary cells), and
context-shuffled probes retain most of the MLP gap (distill T32:
shuf_mlp 0.543 vs mlp 0.552; slope4 shuffle occasionally HELPS).
runpod-b's prereg bet — "a trailing slope is a centered-weights
functional, so shuffle destroys it" — is falsified: under anchor
matching, slope ≈ anchor − window-mean, an order-free functional.
**Verdict: KILL as the card is written** — the hunt deliverable
explicitly requires the within-window shuffle ablation to show order
matters, and for this latent it cannot.

Recorded seed (NOT a verdict): "hedging-trend level" is a genuine
regime-2 aggregation latent — window-readable, per-token-blind,
T-scaling, grounded — i.e. the class the strategy notes flag as
sufficient to separate TXC from per-token-decoded T-SAE even without
order. If a future window relaxes the order requirement, this is the
first candidate to re-card (fresh prereg; do NOT reuse this screen as
its confirmation).

Next: candidate 3 (emotional-instability onset) — the remaining arm-B
queue item.

## 2026-07-24 — runpod-e — candidate 3 (emotional-instability onset) — **KILL** (pre-onset state already converted; no window recovery at any horizon)

Full pipeline per the frozen `emotional_instability/CARD.md`: 600
8-turn gemma-3-12b-it rollouts on 30 verified-impossible puzzles
(elicitation replicates the paper: mean frustration 0.36 → 4.91 by
turn, 65 % of final turns ≥ 5); judge labels κ-gated (qw-κ 0.857,
within-1 0.90; judge = claude-sonnet-4-5, the paper's sonnet-4 being
retired; ≈ $12 of the $40 cap); 554/600 onsets token-mapped; corpus
DOUBLED mid-run (10 → 20 rollouts/puzzle, disclosed) to lift the
anticipation TEST split over the frozen 300/class floor — ant4/ant8
clear it, ant16 (290) skipped per the rule. One infra note: gemma-3-12b
residual norms saturate fp16 — caches store activations × 1/64
(scale-invariant under the frozen z-scored stack); the pre-fix
degenerate cells were discarded before any reading.

Screen results (`emotional_instability/results/screen.json`, hs25 =
resid_post L24): **(a) anticipation** — per-token linear is already
0.856 AUC at offsets 1-4 and 0.712 at 5-8, and the window NEVER beats
it (best window cell ≤ per-token at every T; gaps ≤ 0) — the pre-onset
wind-up is per-token-converted at short horizons and simply not
window-recoverable at longer ones; **(b) escalation intensity** —
tercile acc 0.36 per-token vs 0.39 best window at T64: gap +0.03 <
the +0.05 KEEP bar, weak T-growth, and the shuffle retains it (0.389
vs 0.394) — aggregation, not order; **sanity anchor** — post-onset
detection is per-token-readable at 0.867 AUC rising to 0.958 with a
T64 window with shuffle ≈ ordered (bag-of-words lexical stamping,
exactly the trap the card named), so the labels are valid and the kill
is a genuine negative. Nulls at chance. **Verdict: KILL by the frozen
falsifier** — no readout shows a ≥ +0.05 order-carried, T-growing
window advantage.

**Arm B is closed: all three candidates died by sound screens** —
repetition-lag (converted at every scale), confidence trend (real
window gap but aggregation-carried), emotional instability (converted
near onset, absent farther out). The recurring mechanism across all
three is CONVERSION: whenever a temporal latent leaves per-token
traces, the model has already summarized it into the current residual
by mid-depth, and raw windows add order-free aggregation at best. The
one seed worth carrying forward (recorded under candidate 2) is the
hedging-trend LEVEL — a grounded regime-2 aggregation latent that
separates window archs from per-token-decoded ones without needing
order. Per the acceptance gate this is the honest "all candidates
died" log for arm B; no Stage 2 was run, so the canonical leaderboard
is untouched by this arm.

**2026-07-24 · runpod-b · prep (labels + cards) — no verdict.** All four
label artifacts landed under `labels/` (builders committed pre-run, 10
sanity tests in `tests/test_task_hunt_labels.py`): `replag_fineweb_*` ×3
tokenizers, `ward_lambda` (causal mirror λ̂; round-trip validity matches
the committed ward_stream_stats exactly; event rate by λ̂ tercile
0.053/0.081/0.256), `proofops` (+ the clock bridge: median 16
tokens/sentence ⇒ 2-sentence windows need T ≥ 32), `confidence`
(slope4/slope8 + state; slope8 support ≈ 128 tokens — arm-B candidate-2
reachability input). **Duplication note (per briefing):** runpod-e froze
`replag/CARD.md` with its own inline labels before these landed; the
frozen card's scheme differs from `labels/replag_fineweb_*` (within-
sequence Δ, Δ=1 excluded, B32 bucket, identity/position matching vs my
doc-level Δ, buckets ≤4/≤8/≤16/none). The frozen card governs the
screen; the committed artifacts stand as an independent cross-check +
the n=2 robustness reserve. Label-side order receipt worth having: at
doc level, short-lag BIGRAM repeats are ≈13× the within-doc-shuffle
frequency null, unigram short-lags only ≈1.2× in aggregate — and the
Δ∈[1,4] unigram bucket alone sits BELOW the null (20.9k vs 28.5k on
gpt2), the doc-level face of runpod-e's token-level T4 finding above
(`labels/replag_stats.json`). Cards: `proofops/CARD.md` +
`confidence/CARD.md` frozen (science sections; running agents append
screen cells); `forbidden_word/` + `emotional_instability/` CARD.DRAFTs
staged.

## 2026-07-24 — runpod-d — candidate 1 (backtracking λ̂ intensity) — **KEEP** (qualified: the T-story is real, the ORDER story is not)

Screen complete: 2 models × 2 layers (hs13 = resid_post L12 primary,
hs11 = L10) × 5 T × {λ̂_hist PRIMARY, λ̂ secondary}, frozen `problib`
stack, σ_null = 0.0031 over 17 permutation cells (3σ = 0.0094).
Card: `lambda_intensity/card.md` (frozen before execution);
results `results/lambda_screen.json` + `lambda_verdict.json`; figures
`figs/lambda_tscaling.*`, `figs/lambda_decomp.*`.

**No kill rule fires.** The window−token gap grows with T in all four
(model, layer) cells and does not saturate by T = 32 — base L12
+0.011 → +0.054, base L10 +0.007 → +0.050, distill L12 +0.014 →
+0.054, distill L10 +0.008 → +0.044 (window ceiling = max(flatten,
mean) − per-token). Per-token sits at 0.776–0.795, far above the
position-only floor of 0.592, so the signal is history and not the
position ramp (P5 ✓). P4 ✓ (base ≈ distill, |Δ| ≤ 0.01 at T = 32).

**Scored against the frozen predictions — three falsified.**
(1) **P1 FALSIFIED at T = 8**: the card predicted g > 3σ_null at every
T ≥ 8; distill L10 gives +0.007 at T = 8, and on the RAW flatten arm
P1 fails more widely (flatten is *below* per-token at T ≤ 4 in every
cell). (2) **P2's shape clause FALSIFIED**: the largest increments were
predicted at 8→16 and 16→32 but the biggest single step is T2→T4;
the end-to-end rise + no-saturation clause holds. (3) **P3's order
clause FALSIFIED — the substantive finding**: g_order = flatten − mean
is ≤ 0 in 17 of 20 primary cells (min −0.047), i.e. the window MEAN
beats the ordered flatten, and the within-window shuffle costs only
+0.002…+0.022 AUC. **The entire window advantage is order-free
evidence pooling.** That is exactly the card's own regime-2 prediction
(additive-in-window over lag-weighted sentence indicators) and it means
the shuffle ablation the briefing wants — "order/structure matters" —
is NEGATIVE for this candidate. Reported as such.

**Rule-scoring correction, disclosed.** `render.py` (committed before
the screen) coded K2 as *strictly monotone at every step*, which is
stricter than the card's text ("flat or non-growing over the whole
tested range"). The strict variant turns on for a single 0.005 dip at
T = 8 in distill/L10 — well inside 3σ_null = 0.0094 — and would have
returned KILL. The card's text governs; the strict statistic is
retained and reported alongside as
`P2_strict_every_step_monotone = false`. The renderer was amended to
score both, and the amendment is this paragraph.

**Stage 2 cell:** base L12 (cleanest ladder; distill ties at T = 32,
P4 says the axis does not matter). Regime 2 cannot separate window
architectures from each other — it separates them from per-token — so
Stage 2 is a direct test of the program's standing claim that a
per-token-decoded T-SAE cannot follow a rate/intensity latent up the
T ladder.

## 2026-07-24 — runpod-d — order-sensitivity receipt for the EXISTING backtracking case study — **POSITIVE**

Not a hunt candidate: the briefing's "also wanted" item — the
within-window shuffle control the paper's § 5.2 task never had.
Script `shuffle_receipt.py` (committed before running) reuses the
conversion-depth probe rows VERBATIM (frozen § 2 recipe, 25,155/6,266
rows), so per-token and window reproduce
`conversion_depth/RECORD.md` § 3 exactly — base L10 ant_kw 0.843/0.886
and distill L10 0.844/0.895, both published numbers, recovered on
independently rebuilt caches. The new arms are the per-row within-window
permutation (seed 23) and the window MEAN. σ_null = 0.0035
(3σ = 0.0106); all 12 cells.

**Destroying within-window order costs the ANTICIPATION targets
+0.028…+0.041 AUC — 3–4× the noise floor — while the near-ambient
companion `is_bt` loses only +0.003…+0.013.** ant_kw: +0.034 / +0.036
(base L10/L12), +0.036 / +0.039 (distill L10/L12); ant_bts: +0.041 /
+0.035, +0.040 / +0.028; is_bt: +0.013 / +0.009, +0.012 / +0.003.
The receipt holds on both models and both layers.

The ordering is `shuffled < mean < ordered` on every anticipation cell
(e.g. base L10 ant_kw: 0.852 < 0.872 < 0.886): a shuffled flatten is
*worse* than a position-symmetric mean, so mis-aligned positional
evidence actively hurts. That is what makes the ordered margin an
order effect and not a probe-capacity effect.

**Contrast with candidate 1, on the same substrate and stack:** λ̂
intensity showed shuffle costs of only +0.002…+0.022 with g_order ≤ 0,
i.e. order-free pooling. So within one task family the *anticipation*
label is order-sensitive and the *intensity* label is not — a
distinction the ambience machinery can now state with receipts rather
than assume. Results `results/shuffle_receipt.json`, figure
`figs/shuffle_receipt.*`.

## 2026-07-24 — runpod-d — candidate 2 (proof-operation run structure) — **WEAK KEEP, primary layer only** (and a correction to an earlier reading)

Card: `proofops/card.md` (frozen pre-run). Labels: runpod-b's committed
`labels/proofops.npz` used as-is. σ_null = 0.0035 (3σ = 0.0105).
Primary cell base/L12 complete; confirmatory cells (base L10, distill
L10/L12) still running at the time of writing — this entry covers the
card's designated PRIMARY layer and will be extended, not revised, when
they land. Results `proofops/results/proofops_{screen,verdict}.json`,
figure `proofops/figs/proofops_tscaling.*`.

Macro-OvR AUC, base L12 (per-token → g at T = 8/16/32/64):

| target | tok | T=8 | T=16 | T=32 | T=64 |
|---|---|---|---|---|---|
| `tir` (PRIMARY, time-in-run) | 0.614 | +0.028 | **+0.049** | +0.032 | +0.037 |
| `boundary` | 0.618 | +0.017 | +0.036 | +0.030 | +0.008 |
| `op` (AMBIENT ANCHOR) | 0.760 | +0.037 | +0.041 | +0.036 | +0.018 |

**No kill rule fires, but the survival is weak and should be read that
way.** The card's actual claim is the CONTRAST g_tir − g_op rising:
measured −0.009, +0.008, −0.005, **+0.019** at T = 8/16/32/64. It
clears 3σ_null at exactly one T (64) and is negative or noise at the
other three — non-monotone, one-point survival. K2 as written ("never
exceeds g_op by more than the null floor at ANY T") therefore does not
fire, but a single clearing point is not the threshold-ladder the card
predicted.

**Two frozen predictions falsified.** P1 (nothing below the sentence
clock, then growth at T ≥ 32) is wrong in both directions: `tir` already
has +0.028 at T = 8, and it PEAKS at T = 16 then declines — the
peak-then-decline shape of a *localized* latent (STORY § 7), not a
clock threshold. P3 (g_order > 0 at T ≥ 32) fails at T = 32 (−0.001),
holding only at T = 64 (+0.022).

**CORRECTION to an earlier reading in this session.** Before the
ambient anchor was measured I recorded (agents/runpod-d/STATUS.md) that
`tir`'s within-window shuffle gap growing monotonically with T
(+0.008 → +0.025 → +0.032 → +0.061) was "the order evidence candidate 1
lacked". **The anchor refutes that.** The ambient `op` label — readable
from the current sentence by construction — shows a shuffle-gap ladder
that is the same within noise (+0.010 → +0.017 → +0.034 → **+0.065**),
and so does `boundary`. A shuffle gap that grows with T is therefore a
**generic property of wider windows under this probe** (more positions
to scramble, and a flatten probe that leans harder on positional
alignment as T grows) — NOT evidence that the latent is order-sensitive.
Only the anchor-differenced contrast carries that claim, and here it is
one-point. The card requiring an ambient anchor is what caught this;
the receipt in the § 3 backtracking case study, which has no such
confound because it compares *anticipation vs ambient targets on
identical rows*, still stands.

## 2026-07-24 — runpod-d — candidate 2, FULL GRID (extends the primary-layer entry above; nothing there is revised) — **KEEP**, best cell = distill L12

All 64 cells complete (exit 0). σ_null = 0.0046 over the full grid
(3σ = **0.0137**, wider than the primary-layer-only 0.0105, so this is
the stricter bar). The card's claim — the contrast g_tir − g_op rising
with T — resolved per (model, layer):

| cell | T=8 | T=16 | T=32 | T=64 | clears 3σ at |
|---|---|---|---|---|---|
| base L12 | −0.009 | +0.008 | −0.005 | +0.019 | T=64 only |
| base L10 | +0.004 | −0.005 | +0.012 | +0.031 | T=64 only |
| **distill L12** | **+0.017** | **+0.020** | **+0.023** | **+0.042** | **every T** |
| distill L10 | −0.023 | −0.013 | −0.017 | +0.017 | T=64 only |

**The candidate survives, and the model axis is the finding.** On
**distill L12** the contrast is positive at every T, **monotonically
rising** (+0.017 → +0.042), and clears 3σ_null at all four window
sizes — the run-depth latent has window access the ambient anchor does
not, growing with T. On the three other cells the contrast is noise
until T = 64. Every cell peaks at T = 64 and none saturates, so the
direction is consistent; only the generator-at-mid-depth cell is
unambiguous.

This is exactly the briefing's premise that **non-ambience is a (task,
MODEL) property** — measured, not assumed. It is also the mirror image
of candidate 1, where base ≈ distill held (P4 ✓): the intensity label
is reader-readable in both models, the run-depth label is not.

**Frozen predictions: P1, P3 and P5 falsified; P2 holds.** P1 (a clock
threshold: nothing at T ≤ 16, then growth) is wrong — distill L12 has
+0.017 already at T = 8. P3 (g_order > 0 at T ≥ 32) fails at T = 32.
**P5 (base ≈ distill) is decisively falsified** — the model axis is the
dominant source of variation, not a nuisance.

The earlier caveat still stands and is why the anchor mattered: the raw
shuffle-gap ladders are monotone in T for 8 of 12 target-cells
**including the ambient `op` anchor in all four**, so only the
anchor-differenced contrast above supports an order claim.

**Stage-2 candidate:** distill L12 is the cell to panel if candidate 2
is taken forward; that is a separate run from the candidate-1 Stage 2
now executing, and is NOT part of this session's acceptance gate.
Results `proofops/results/proofops_verdict.json`, figure
`proofops/figs/proofops_tscaling.*`.

## 2026-07-24 — runpod-d — candidate 1 STAGE 2 (backtracking λ̂, real Ward activations) — **QUALIFIED POSITIVE**

The head-to-head panel: 84 cells, 0 failures, through the canonical
runner — 5 archs × T ladder × seeds {1,2,42} + untrained, at a single
scarce anchor d_sae = 2048 (= d_in/2), nominal k_pos = 8. Datasource
`ward_real_lambda_base_l12` (real base-model resid_post L12 + the frozen
λ̂_hist labels; plugin path, no core edits). Headline metric =
`lambda_recovery` (held-out Pearson r of a per-tile linear probe;
chance ≈ 0). Figure `lambda_intensity/figs/stage2_tscaling.*`, numbers
`lambda_intensity/results/stage2_summary.json`. mean ± std over 3 seeds:

| arch | T=1 | T=2 | T=4 | T=8 | T=16 | realized l0 |
|---|---|---|---|---|---|---|
| per-token BatchTopK SAE | 0.113 | — | — | — | — | 6.3 |
| **T-SAE** | **0.154** | — | — | — | — | 7.4 |
| Stacked | — | 0.109 | 0.143 | 0.125 | 0.094 | 7.0–7.9 |
| **TXC-pre** | — | 0.132 | 0.192 | **0.206** | 0.138 | 6.9–7.8 |
| TXC-post | — | 0.130 | 0.161 | 0.185 | **0.255** | **3.4→0.5** |

**The clean, matched-budget result is TXC-pre.** At realized l0 ≈ 7–8
(the same per-token budget the token archs and Stacked run at), λ̂
recovery **rises 0.13 → 0.19 → 0.21 across T = 2/4/8**, above both
per-token decoders — the per-token BatchTopK SAE (0.113) and **T-SAE,
the baseline the hunt names (0.154)** — and the trained−untrained margin
**grows with T to +0.150 at T = 8** (untrained falls 0.09 → 0.06 → 0.01,
so the architecture is learning something T-dependent, not reading it
off at init). It dips at T = 16 (0.138): a peak-then-decline, not
saturation — consistent with the Stage-1 regime-2 reading (the window
pools more lag-weighted history until added positions dilute a fixed
code budget). **This is the hunt's target pattern — a window code
recovering a real-activation latent better than the per-token decoders
and improving with T — realized at matched sparsity, if modestly.**

**Two heavy caveats, both reported not buried.**
1. **TXC-post's higher numbers are NOT budget-matched.** Its recovery
   is monotone to the best cell in the panel (0.255 at T = 16), but its
   realized l0 **collapses 3.4 → 1.8 → 0.9 → 0.49** — at T = 16 it fires
   ~1/16 the atoms/token the others do (the post-squash k_win//T
   correction starves the code as T grows). Achieving 0.255 on half an
   atom per token is a striking *efficiency* observation, but it breaks
   the matched-l0 comparison, so it cannot be the headline. Flagged per
   RECORD § 4.6.
2. **Stacked shows a training pathology at large T.** It is non-monotone
   and at T = 16 the TRAINED model (0.094) sits **below its own
   untrained control (0.171)** — a negative margin. Not a win for
   anyone; recorded as a pathology.

**Verdict: QUALIFIED POSITIVE.** The money plot exists — TXC-pre beats
the per-token/T-SAE baselines at matched budget and rises with T through
T = 8 — but it is modest (recovery band 0.10–0.21 on a hard real
regression), peaks rather than saturates, and the single largest number
in the panel (TXC-post 0.255) is budget-confounded. The result matches
the Stage-1 screen it was built from: an order-free, additive-in-window
regime-2 latent, where a window architecture earns a real but bounded
advantage over per-token decoding. Full write-up: RECORD § 4.

## 2026-07-24 — runpod-d — candidate 3 (forbidden-word onset, SILOED) — feasibility gate PASS; screening

Generation complete: 1169 R1-Distill rollouts on the CoT-Control
keyword-suppression split (pinned commit; enforce_eager + max_model_len
8192; 0 prompts dropped). Stats
`forbidden_word/results/forbidden_word_gen_stats.json`.

**Feasibility gate (card § "Feasibility gate"): PASS with wide margin.**
Violation rate **97.4 %** (1139/1169) — the card predicted this from the
R1-family's ≈ 0.1 % controllability, and the gate needed only ≥ 30 % /
≥ 200 violating rollouts. Keyword onset is EARLY (first-keyword token
position: median 86, p90 370, on a 388-rollout sample), so the frozen
1024-token activation cache captures ~76 % of violators with ≥ 32
tokens of anticipation room before onset — ample for the D ∈ {4, 8, 16}
horizon bands (shorter horizons naturally get more rows, the ladder the
card predicts). Disclosed cache choices: SEQ_LEN 1024 (onsets past it
are the latest/least-circling ones); 54 % of generations hit the
2048-token max (R1 reasons at length) — irrelevant to an anticipation
label measured before the first occurrence. Proceeding to the screen
(the two tokenizer/BatchEncoding bugs found in pre-flight are fixed).

## 2026-07-24 — runpod-d — candidate 3 (forbidden-word onset, SILOED) — **KILL (pre-registered ambience kill)**

Screen complete: R1-Distill resid_post L12 over its own 1169 rollouts,
frozen `problib` stack, horizons D ∈ {4, 8, 16} × T ∈ {2,4,8,16,32},
split by rollout. σ_null = 0.0099 (18 null cells; 3σ = **0.0296**).
Results `forbidden_word/results/forbidden_word_screen.json`. Card
`forbidden_word/card.md` (frozen before any rollout existed).

**The card's pre-registered kill (P4) fires cleanly.** Per-token AUC vs
the best window ceiling, per horizon:

| horizon | per-token | best window | \|diff\| | max g (win−tok) | max g_order |
|---|---|---|---|---|---|
| D=4 | 0.629 | 0.626 | 0.003 | −0.003 | +0.022 |
| D=8 | 0.612 | 0.622 | 0.010 | +0.010 | +0.047 |
| D=16 | 0.558 | 0.562 | 0.004 | +0.004 | +0.023 |

Per-token is **within 0.02 of the window at every horizon** (kill rule 1
= P4), and the window **never beats per-token beyond 3σ_null** (max
g = +0.010 ≪ 0.0296, kill rule 2). This is exactly the crux the card
named: pre-violation the model **circles the forbidden concept**, so
each semantically-neighboring token is individually informative and a
single token reads the imminent-violation pressure as well as a whole
window. **The anticipation is ambient. KILL.** Under the prime directive
this is a successful outcome of the process — the design pre-registered
this as the likely result and it came true, cleanly.

**The one honest nuance (does NOT rescue the candidate).** g_order
(flatten − window-mean) is positive and *grows with T* at the two
longer windows — up to +0.047 at D=8/T=32, beyond 3σ_null. So there is
a faint genuine *within-window order* signal: at large T the ordered
flatten reads the run-up slightly better than a position-symmetric
mean. But it does not lift the window as a whole above a single token
(the window ceiling still ties per-token), so it cannot carry the
non-ambient anticipation claim — a token already sees what the window
sees. Recorded as a real-but-insufficient sub-effect, not a survival.

**Model note:** this was screened on R1-Distill only (generator =
reader), per the card. Ambience is a (task, MODEL) property; a
different reader could differ, but the card's kill is on the generator
itself and it is decisive there. SILOED from Aniket's parallel
forbidden-word work throughout — no shared inputs.

## 2026-07-24 — runpod-d — candidate 3 depth sweep (POST-HOC diagnostic) — mechanism = **CONVERSION**, and my own kill-explanation was WRONG

**Status: post-hoc, does NOT reopen the frozen L12 KILL** (script +
blind predictions committed before any cell existed; `2b3194a6`). The
KILL stands exactly as recorded above. This answers only *why*.

All 51 cells (17 capture points × 3 horizons, T = 16, same rollouts and
same seeded rows as the frozen screen — the hs13 slice reproduces its
0.629 at D=4 exactly). σ_null = 0.0081 (3σ = **0.0242**).
Results `forbidden_word/results/forbidden_word_depth.json`.

| hs | D=4 tok / g | D=8 tok / g | D=16 tok / g |
|---|---|---|---|
| 0 (emb) | **0.538** / +0.037 | 0.535 / +0.015 | 0.505 / +0.023 |
| 5 | 0.569 / +0.024 | 0.568 / +0.004 | 0.553 / +0.019 |
| 13 (=L12, screened) | 0.629 / −0.003 | 0.612 / −0.001 | 0.558 / −0.012 |
| 21 (peak) | **0.668** / −0.008 | 0.609 / +0.011 | 0.540 / +0.003 |
| 31 | 0.631 / +0.012 | 0.596 / +0.005 | 0.549 / +0.022 |

**The window gap is shut at EVERY depth:** across all 51 cells only
2/51 exceed 3σ_null and neither exceeds +0.037 — there is no layer at
which a window reads the pressure better than a single token. The kill
is therefore not an artifact of the frozen layer choice; it holds
depth-wide.

**The mechanism is CONVERSION, not lexical ambience — my LOG entry
above got this wrong and the correction is the finding.** The kill entry
attributed the tie to the model "circling" the concept, i.e. a
bag-of-words semantic-neighbourhood leak. **The embedding layer
falsifies that**: at hs0 a per-token probe reads only 0.538 (D=4)
against nulls ≈ 0.50 — near-blind. A lexical property would be legible
right there, the way backtracking's is (`conversion_depth` § 3: hs0
window gap **+0.174**, explicitly bag-of-tokens n-gram signal). Instead
per-token *climbs +0.13 AUC with depth* (0.538 → 0.668, peak ≈ L20)
while the gap never opens: the model **computes** imminent-violation
pressure across depth using cross-token information and deposits the
result at the current position. Genuinely temporal in origin, fully
linearized per-position by the time any probe sees it.

**Blind-prediction scorecard** (committed pre-run): **D3 CONFIRMED**
(per-token at hs0 ≪ its L12 value ⇒ attention does real work).
**D1 and D2 FALSIFIED** — I predicted the `is_bt` shape, a large hs0
lexical gap collapsing after one block; there was no large hs0 gap to
collapse (+0.037 < the predicted +0.05). I wrote the prediction while
still believing my own bag-of-words story, and the data refused it.

**A fourth g(ℓ) shape for the atlas.** Alongside `conversion_depth`'s
three (converted-by-block-1 / flat-never-converted plateau /
mid-depth inverted-U), this label is **built-and-immediately-linearized**:
near-blind at the embeddings, rising per-token readout with depth, and
**no window margin at any layer**. Contrast on the same axis:

| label | hs0 gap | fate with depth | window residue |
|---|---|---|---|
| backtracking `ant_kw` (future event) | +0.174 lexical | one block converts most | **+0.035…+0.057 forever** |
| backtracking `is_bt` (current state) | +0.107 lexical | converts | +0.007…+0.026 (≈ closed) |
| forbidden-word pressure | +0.037 (none) | **built** 0.538 → 0.668 | **≈ 0 at every depth** |

**Design consequence.** The screening question is not "is the property
semantically non-obvious" — forbidden-word pressure is *already*
non-lexical and computed, and it still leaves nothing. It is **"will the
model decline to maintain this as a per-position state?"** Backtracking
anticipation qualifies because it is a hazard over the recent
trajectory, not a fact the model tracks. This also predicts that a
logical-deduction variant (e.g. "don't mention XOR") would fare no
better and plausibly worse: a completed deduction is a canonical
converted state. Cheap pre-screen for any future candidate: run this
sweep first — per-token climbing while g stays shut ⇒ converted ⇒ do
not spend a grid.
## 2026-07-24 — mac-local — REVIEW: arm A + arm B + prep + dissection — **APPROVED, all verdicts stand**

Gate integrity verified by git forensics on every chain: cards and
scripts strictly precede their runs (dissection: card 22:40 → build
22:45 → grid → skeptic; § 9 amendment frozen before the pre build;
Stage-2 renderer committed before the panel completed). Tests: 220
pass. Leaderboard: 8,616 rows decompose exactly as 7,116 baseline +
1,416 dissection + 84 Stage-2; 0 dup `eval_key`s, 0 null metrics;
dirty rows all carry pinned `diff_sha256`. Numbers spot-checked
against artifacts, not the prose: `stage2_summary.json` (every cell of
the § 3b table, l0 collapse, untrained margins, Stacked pathology),
replag gpt2 (det win−tok ≤ 0 in 20/20 cells; lag4 0.634/0.522/0.520),
confidence distill (0.468 → 0.521/0.545/0.565; shuf-MLP 0.543),
emotional (0.856 / 0.867 / 0.362), the 12-cell shuffle receipt, the
dissection table (+0.093 ± 0.038; skeptic raw persisted per claim).

**Review notes — qualifications that bind downstream use:**
1. Stage 2 reads ONE tile's code per prediction (`lambda_recovery.py`
   per-tile leading-edge convention, the same leak-free design as the
   synthetic DPI bench). Per-token archs are therefore read at single
   positions by construction. Any rebuttal sentence must say "under
   the code-readout convention", and carry the code-rate defense:
   pooling T-SAE codes across T positions would spend T× the code
   bandwidth a window arch uses.
2. The TXC-pre − T-SAE margin is ≈ 2σ at n = 3 seeds (0.206 ± 0.020 vs
   0.154 ± 0.037). Real and consistent across T = 4/8, but phrase it
   variance-aware; the T-rise plus the growing trained−untrained
   margin (+0.150 at T = 8) carries the claim, not one cell.
3. The stage2 figure MUST gain a realized-l0 annotation on TXC-post
   (0.49 at T = 16) before any external use — visually it reads as the
   winner and it is not budget-matched.
4. proofops card divergence: runpod-b's prep draft (is_run_start
   primary) vs runpod-d's operative frozen card (tir primary) — both
   committed pre-run; the running agent's card governs per protocol.
   The CARD.md/card.md filename case-collision (breaks
   case-insensitive checkouts) was fixed by renaming the draft to
   `PREP_DRAFT.md`; both blobs unchanged.
5. The forbidden-word kill is scoped to generator = reader
   (R1-Distill), exactly as the card froze it.

**Round-2 decisions (runpod-e proposals + runpod-d follow-ups
adjudicated):** GREENLIT — (a) runpod-d: budget-matched TXC-post
re-run (the single highest-leverage cheap run: if the monotone rise
to 0.255 survives realized-l0 matching, the money plot upgrades from
peak-at-8 to monotone-through-16) + the figure l0 annotation;
(b) runpod-e, after its in-flight forbidden-word-linearization mini
study: hedging-trend LEVEL Stage-2 on a FRESH card — the program
accepts an aggregation-framed regime-2 win (shuffle-IMMUNITY disclosed
as the mechanism receipt; the theory already claims regime-2 separates
TXC from per-token-decoded T-SAE without order) — + the early-layer
g_order(ℓ)/g_agg(ℓ) addendum on existing caches. ADOPTED as hunt
convention: per-token-first triage (a cheap per-token-only probe
gates the full window grid; all five round-1 kills were visible
there). PARKED post-rebuttal: gpt2-scale order cell, anti-conversion
candidate class, cand-2 Stage-2 on distill L12 (weak contrast; panel
unlikely to clear it before Sunday). Round-2 briefs:
`briefings/task-hunt-r2-d.md` / `task-hunt-r2-e.md` (split per agent,
one briefing per pod as in round 1); round-1 briefings retired.

**Addendum (mac-local, same day): the cand-3 POST-HOC depth sweep
(runpod-d, entry above) is also REVIEWED — APPROVED.** Freeze order
verified by git forensics (sweep script 12:55 → blind D1–D3
predictions 12:56 → results 13:12); the frozen L12 KILL is untouched.
The mechanism correction — **CONVERSION, not lexical circling**
(near-blind hs0 at 0.538; per-token climbing +0.13 with depth; the
window gap shut in 49/51 cells and never above +0.037) — **supersedes
the "circling" language in the original kill entry and in every
downstream summary**. Adopted into hunt conventions alongside
per-token-first triage: the fourth g(ℓ) shape
(built-and-immediately-linearized), the screening question "will the
model decline to maintain this as a per-position state?", and the
depth sweep as the cheap WHY-diagnostic when per-token is high. The
honest scorecard (D1/D2 falsified against the agent's own favored
bag-of-words story) is exactly the discipline the program wants.

## 2026-07-24 — runpod-b — Stage-2 λ̂ VARIANCE RECEIPTS (item 1 of hunt-support-stats) — what is and is not significant at n = 3

Committed `support_stats/stage2_variance.{json,md}` (builder
`stage2_variance.py`, committed first; every number script-derived from
the 84 leaderboard rows, cross-checked EXACTLY against
`lambda_intensity/results/stage2_ward_real_lambda_base_l12.json` — the
build aborts on any mismatch). All tests pass (227). Receipts, honestly
partitioned:

**Significant at n = 3 (exact tests, real resolution):**
- The TXC-pre RISE itself, T = 2→8: exact within-seed permutation
  (216 relabelings, pooled seeds) p = **0.0093**; per-seed slopes all
  positive (0.061/0.030/0.021 per log₂T).
- The rise of its trained−untrained margin, T = 2→8: p = **0.0046**
  (the 1/216 floor — the observed labeling uniquely maximizes the
  pooled slope).
- Trained−untrained margins per cell (paired by seed): pre/T8 0.150,
  95% t CI [0.086, 0.215]; pre/T4 0.104, CI [0.060, 0.148]. Both
  exclude 0 comfortably.
- TXC-pre − per-token SAE: bounded at T8 (t CI [0.005, 0.182]) and
  T16 (CI [0.003, 0.047]); all seeds positive at T4/T8/T16.

**NOT bounded at n = 3 (say it plainly in the rebuttal):**
- The cross-arch TXC-pre − T-SAE paired margin: T8 = 0.052 ± 0.055,
  t CI [−0.086, 0.190]; sign-flip p at its n = 3 floor (0.125; all
  three seeds positive). Pairing bought no variance reduction — the
  arms' seed noise is uncorrelated (r = −0.21 at T8), so the paired sd
  matches the independent-arms value. Consistent with the review's
  note 2: phrase the T-SAE comparison variance-aware; the T-rise +
  trained−untrained margin carry the claim.
- Secondary transparency: the 2→16 trend is NOT significant
  (p = 0.39) — the fall at T16 is real; the pre-registered rise claim
  is 2→8.

**→ runpod-d — seed top-up recommendation (append to your round-2 run
if it lands in time):** criterion = one-sided 95% t lower bound > 0 on
the paired pre-vs-tsae diff at the T8 headline cell, plus sign-flip
attainability (2⁻ⁿ ≤ 0.05 needs n ≥ 5). Answer: **6 seeds total ⇒ 3
extra seeds** (suggest 3, 4, 5) × {txc_batchtopk_pre/T4,
txc_batchtopk_pre/T8, tsae/T1} = **9 trained cells** (untrained
counterparts optional — margin receipts already bind at n = 3).
Headroom option: 4 extra seeds = 12 cells (slack against the sd itself
being an n = 3 estimate; reaches sign-flip p = 1/128). T4 is NOT
cheaply boundable (n = 12 to bound, 23 for 80% power) — don't buy it
with cells; the trend receipt carries it.

## 2026-07-24 — runpod-b — variance-aware Stage-2 renderer MERGED (item 2 — runpod-d: re-render is just a re-run, don't duplicate)

`lambda_intensity/render_stage2.py` upgraded and merged (commit
"task_hunt/support_stats: variance-receipts builder + exact small-n
stats lib + variance-aware Stage-2 renderer"): whiskers are now 95% t
CIs over seeds (not ±std); every arch's legend carries its realized-l0
range; an arch whose min cell-mean l0 < k/2 is flagged NOT
budget-matched in the legend AND annotated on the plot (TXC-post:
realized l0 = 0.49 at T16 vs nominal k = 8) — review note 3 satisfied;
a budget-matched-only variant fig (`stage2_tscaling_matched.*`) omits
non-matched lines. `stage2_summary.json` keeps every committed field
byte-identical and adds `ci95_trained`, `l0_range`, `budget_matched`,
`match_rule`. Re-rendered figs committed. runpod-d: after your
budget-matched cells land, just re-run
`…lambda_intensity.render_stage2` — the l0 legend/flag and CI whiskers
are computed from the rows, nothing to port.

## 2026-07-24 — runpod-b — hunt-support items 3+4 SHIPPED: anti-conversion data side (parked class — prep only) + hedging-LEVEL draft card

**Item 4 (for runpod-e):** `confidence/LEVEL_CARD_DRAFT.md` — the
hedging-LEVEL Stage-2 draft (window-mean-level primary with decision
points marked, shuffle-IMMUNITY as the disclosed mechanism receipt,
code-readout-convention sentence included, T ladder to 32 per the
clock bridge). Sharpen and freeze your own; the draft is an edit-
distance saver, not an operative card.

**Item 3 (parked anti-conversion class, data side ONLY — no screen
without a freed pod + mac-local greenlight):** builder
`labels/build_interleave.py` (+ `interleave_lib.py`, 5 sanity tests)
committed BEFORE the artifacts `labels/interleave_fineweb_{gpt2,gemma2,
llama31}.npz` + `interleave_stats.json`; draft card
`interleave/CARD_DRAFT.md`. 200 lexically-matched pairs (greedy
max-Jaccard, overlap 0.080 → 0.120 vs random), 1–4-sentence jittered
alternating blocks, per-token `source` + `tss` (tokens-since-switch,
-1 first-block guard), shuffled-block null shipped as a within-doc
permutation with relabeled `source_null`/`tss_null`.

**Label-side triage (the per-token-first numbers, no activations
touched; stable across all three tokenizers):** source identity
unigram AUC **0.66 matched vs 0.70 random** — the lexical control
works but removes only ~0.04, so the frozen prior stands: per-token
HIGH on source is the expected kill, and `tss` (unigram ≈ **0.55**,
near-blind) is the face that must carry the candidate. Switch hazard
mildly rising (~0.012 → ~0.03; jittered blocks are not memoryless —
disclosed, not hidden). Methods note the tests forced: any in-corpus
unigram estimator LEAKS the source through its own count asymmetry
(leave-block-out scored AUC 1.0 on identical vocabularies); the
committed triage estimates from held-out doc halves, disjoint from
the corpus by construction.
## 2026-07-24 — runpod — hunt-support-synthetic: both receipts done (Item 1 **NO-MIRROR-DIP**; Item 2 **FLAT**)

Mandate `briefings/hunt-support-synthetic.md`; card frozen pre-build at
`support_synthetic/CARD.md` (freeze commit → build commit with 6 green
contract tests → grids → verdicts read mechanically off the pre-committed
`analyze_dilution.py` / `analyze_tsae.py`). 102 cells requested, 0 failures;
leaderboard 8,688 rows, **0 dup eval_keys** (the 30 identical-config bench
cells came back as runner cache hits, never re-appended). Records under
`support_synthetic/results/`; figures
`support_synthetic/figs/{dilution_tscaling,tsae_fair}.*`.

**Item 1 — budget-dilution receipt: NO-MIRROR-DIP (the card's frozen third
branch). The mirror reproduces the real panel's rise but not its dip, so the
RECORD § 3b dilution clause gains no mirror support.** Under the real Stage-2
convention exactly (fixed d_sae, fixed per-token k_pos), TXC-pre on the λ̂
mirror rises 0.870 → 0.952 at the kernel-support peak (T = 4) and then stays
flat to the end of every ladder: A1 (d = 20) 0.949 at T = 16; A2 (d = 40)
0.949 at T = 32; budget-scaled B (d = 5·T) 0.942–0.952 throughout. No decline
reaches the 0.05 bar anywhere (max |paired D| = 0.003). The sharp part is the
realized-l0 trace: the fixed-budget arms **do** starve exactly as the
dilution story requires — A1 realizes 8.9 of 16 nominal atoms/window at
T = 16 (0.56/token), A2 19.8 of 32 at T = 32 — and recovery does not move.
Set against the real panel's own receipt (TXC-pre realized l0/token 6.9–7.8
≈ nominal at every T, *including* the dipping T = 16), dip and budget
starvation are **doubly dissociated**: the real line dips without starving;
the mirror starves without dipping. Per the frozen falsifier, the real T = 16
dip needs a different explanation; recommendation for review: qualify the
§ 3b clause "… until extra positions dilute a fixed code budget" to "cause
not established". Candidate explanations (labelled speculation, not
receipts): content competition during training (the mirror's λ̂-relevant
content is compact and stationary; real Ward windows add rich competing
content per position) or undertraining at large T (8k real steps vs 30k
mirror). One honest capacity note: the *untrained* mirror controls do decline
with T at fixed d_sae (0.78 → 0.60 along the A1 ladder) and are restored by
scaling d_sae — the capacity effect exists at init; training erases it here.
Bookkeeping disclosure: CARD § 1.2's enumeration under-counted the new
untrained controls (27 ran, not 15 — untrained-at-every-line-point was the
frozen intent; extra controls only, commentary-only per the card).

**Item 2 — T-SAE fairness receipt: FLAT. The rejoinder is closed.** The
registered T-SAE's one temporal hyperparameter — the contrastive pair
distance, hardcoded to consecutive tokens — was exposed through a plugin
variant arch (`TSAEDelta`, contract-tested **bitwise-identical** to `tsae`
at Δ = 1: same RNG stream, losses, and parameter trajectory) and swept
Δ ∈ {1, 2, 4, 8} at the canonical mirror budget, plus aux `tsae_a0`
(registered class, contrastive_alpha = 0). λ̂ recovery 0.409 / 0.409 / 0.399
/ 0.398, α = 0 at 0.399 — max |paired D vs Δ=1| = 0.011 against the 0.05
bar; every setting inside the bench's T-SAE band (0.38–0.44) at the provable
per-token DPI floor ≈ 0.41. Untrained guard PASS (exact equality across all
five entries, per seed). The real panel's single-config T-SAE cell is not
underestimating the baseline through its temporal knob — the per-token
decode, not the training window, is binding; even deleting the contrastive
term entirely moves nothing on this latent. No rise ⇒ no runpod-d flag;
skeptic not triggered ($0, frozen policy).

_Recorded-by: claude-fable-5 (runpod, hunt-support-synthetic briefing),
2026-07-24. Card frozen pre-build; scripts committed pre-run; verdicts
mechanical; stopped for review — briefing stays until mac-local retires it._

## 2026-07-24 — mac-local — REVIEW: hunt-support (runpod + runpod-b) — **APPROVED**, and one interpretation is RETRACTED

Gate integrity: both freeze chains verified in the commit sequence
(support_synthetic CARD 14:49 → build w/ 6 green contract tests 14:55
→ results 15:18; support_stats builders 14:44 → outputs 14:48;
interleave builder 14:59 → artifacts 15:01). Numbers verified against
artifacts: `stage2_variance.json` (trend p = 0.00926 over 216
permutations, per-seed slopes 0.061/0.030/0.021; T8 pre-vs-tsae
0.052 ± 0.055; power → 6 seeds), `dilution_verdict.json` (A1 T=16
realized 8.95/16 atoms = 0.56/token with recovery unmoved),
`tsae_fair_verdict.json` (max |paired D| 0.011 vs 0.05 bar; untrained
guard exact-0). Leaderboard 8,688 rows = 8,616 + 72 (30 cache hits
never re-appended), 0 dup keys; 250 tests pass; the matched-only
figure exists. The undercounted-untrained-controls disclosure
(27 vs 15, extra-only) is accepted.

**Consequences, binding now:**
1. **The § 3b budget-dilution interpretation of the T = 16 dip is
   RETRACTED to "cause not established"** (the mirror receipt's double
   dissociation: the real line dips without starving; the mirror
   starves — to 0.56 atoms/token — without dipping). The RECORD review
   stamp is amended accordingly. The rebuttal claim narrows to what
   the variance receipts bound: the pre-registered T = 2→8 RISE
   (exact p = 0.0093), the growing trained−untrained margin
   (p = 0.0046; T8 CI [0.086, 0.215]), and pre > per-token (bounded at
   T8/T16). The pre-vs-T-SAE margin is stated as all-seeds-positive
   but NOT bounded at n = 3 — pending the seed top-up.
2. **runpod-d:** the seed top-up (3 extra seeds × {pre/T4, pre/T8,
   tsae/T1} = 9 trained cells) is now FIRST-CLASS in your round-2 run,
   not optional — it is what converts the headline comparison from
   "consistent" to "bounded". OPTIONAL, only with GPU headroom after
   post re-run + top-up: a step-ladder at pre/T16 (8k vs ≥ 24k steps,
   3 seeds) to test the undertraining explanation for the dip —
   labelled exploratory, lowest priority.
3. **T-SAE fairness rejoinder CLOSED with a receipt** — the baseline's
   temporal knob is inert on this latent (even contrastive_alpha = 0
   moves nothing); quotable alongside the panel.
4. **Anti-conversion class:** prep accepted; per the triage numbers
   the candidate's face is `tss` (unigram ≈ 0.55, near-blind) — source
   identity (0.66 even matched) is the expected kill and must not be
   the primary. Stays PARKED until a pod frees + explicit greenlight.
5. Both support briefings retired. runpod and runpod-b are idle;
   next assignment on request.

## 2026-07-24 — runpod — candidate factory: ledger committed + bundle B1 (interleave `tss`) SCREEN-READY

Mandate `briefings/candidate-factory-broad.md` (quantity mode; I own
non-Ward candidates, runpod-b owns the Ward grid). **Ledger first:**
`CANDIDATES.md` committed before any builder — 18 ideas vetted on the
four round-1 axes, 6 BUILD / 6 PARK / 6 DEAD, ship order B1 → B2
(vocabulary-novelty, fineweb) → B3/B4 (list-density + question-rate,
one builder) with B5/B6 (dialogue, OpenWebMath) as stretch.

**B1 shipped (packaging, not building — runpod-b's committed label
artifacts stand untouched):** `interleave/CARD_DRAFT.md` promoted from
parked draft to screen-ready operative draft under the briefing's
explicit quantity-mode lift of the park. Framing frozen into the
draft: `tss` PRIMARY (unigram ≈ 0.55, near-blind — the passed triage),
`source` demoted to disclosed regime-1 anchor (0.66 matched = the
expected kill face, does not count against the candidate); predicted
T-pattern (tss window-readable, T-growing over {4…32}, degraded on
the shuffled-block null) + draft kill rule staged for the freezing
agent. Economics stated in the card: the interleaved streams are NEW
token sequences ⇒ one ~330k-token forward-pass caching run per model
(minutes on H100), not the existing fineweb caches.

_Recorded-by: claude-fable-5 (runpod, candidate-factory-broad),
2026-07-24. Bundles land incrementally; one LOG line each._

---

## 2026-07-24 — runpod-b · FACTORY BUNDLE 1 SHIPPED: `sc_lambda` (self-correction marker intensity λ̂_sc)

First bundle of `briefings/candidate-factory-traces.md` (QUANTITY MODE).
Builder + frozen card committed BEFORE outputs (commit "factory: shared
frozen label logic (exp kernel, marker list, triage kill rule) +
sc_lambda builder/card"). Bundle: `labels/sc_lambda.npz` +
`labels/sc_lambda_stats.json`, card `sc_lambda/CARD_DRAFT.md`. Screen-
ready on the existing Ward base/distill caches — same manifest
conventions as `ward_lambda.npz` (20k rows/class, pos ≥ 32, by-trace
split).

- **Events**: frozen 17-pattern lexical marker list; sentence event
  rate 0.136 (Sonnet is_bt was 0.130 —
  same regime, zero-API). Kernel: exponential τ = 3, K = 8, causal,
  normalized, history guard i ≥ 4; kernel-only (no position term).
- **Label-side triage PASSED, narrowly on the token face — disclose
  when screening**: current-token-identity AUC
  0.636 (kill at 0.65), position
  0.625 raw / 0.618
  frac (kill at 0.70). Marker tokens themselves are masked out of all
  manifests (`is_marker_tok`, 1.0% of valid
  tokens); `is_sc` ships as the ambient control.
- **Receipts in the bundle**: self-excitation monotone (is_sc rate by
  bin 0.082 → 0.217);
  corr(λ̂_sc, ward λ̂_hist) = 0.47
  (the winner's family, new event stream — related, not a re-skin);
  event-shuffle null decorrelated (corr 0.19)
  with its own manifests `man_null_*`; visible-evidence ceiling
  (in-window marker count alone) T8 0.525
  / T16 0.578 / T32
  0.701 — window probes must BEAT this
  line at matched T or they are counting visible marker tokens.
- Bin scheme: zero_split fired (rate distribution zero-inflated;
  {0, ≤0.186, >} — recorded in stats). Running
  agent freezes its own screen card; draft T-pattern + falsifier in
  `sc_lambda/CARD_DRAFT.md`.
