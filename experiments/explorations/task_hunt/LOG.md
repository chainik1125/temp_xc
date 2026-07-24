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
