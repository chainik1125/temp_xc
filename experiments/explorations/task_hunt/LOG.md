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
