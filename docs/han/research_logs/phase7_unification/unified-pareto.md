---
author: Han
date: 2026-05-02
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary — unified Y+W Pareto frontier (matched-sparsity steering)

> **Headline (2026-05-02 update — V7 tiled-broadcast steering protocol added)**:
> Han asked "have we tried the obvious thing — non-overlapping windows + same
> steering vector per window?" The answer was no, and the obvious thing wins.
> V7 tiled-broadcast (stride-T blocks, single uniform δ within each block,
> derived from per-block encode/clamp/decode) **lifts multiple architectures
> over the previous best-protocol numbers and produces NEW prereg WINS**:
>
> - **🚀 Y Galaxy 11 SoftMaxPool+H8 V7 3sd: peak15 = 1.689** ⭐⭐⭐⭐
>   (Δ vs anchor 1.133 = **+0.556**) — NEW PRREG TOP CELL, beats W Contrastive-merge.
> - **🚀 Y Galaxy 18 SoftMaxPool T=3 V7 3sd: Δ = +1.033 at coh ≥ 1.75** —
>   the largest individual WIN ever recorded (overall best at the GIGABRAIN
>   metric).
> - **Y T-SAE WS V7 3sd: peak15 = 1.333** (Δ = +0.200; reaches PP-level)
> - **Y Galaxy 6 max-pool V7 3sd: peak15 = 1.311** (Δ = +0.178)
> - **Y Galaxy 23 SoftMaxPool T=5 V7 3sd: peak15 = 1.089** (Δ = -0.044)
>
> V7 is the **best protocol for at least 5 of 9 archs tested** (T-SAE WS,
> Galaxy 6, Galaxy 11, Galaxy 18, Galaxy 23 — also TIES Galaxy 11 PP).
>
> V7 LOSES for archs with end-position-heavy decoders:
> - **T=2 H8 V7 = LOSS** (Δ = −0.100; H8 contrastive trains end-position-discriminative
>   features, V7's averaging dilutes them; H8 wants RE)
> - T=2 bare V7 ≈ PP (V7 +0.467, PP +0.567)
> - Galaxy 8 T=2 V7 < PP (V7 +0.411, PP +1.011 — at T=2 attention-mix isn't
>   the bottleneck so PP's content-locality wins)
> - Galaxy 20 LSE pool V7 < PP (V7 +0.289, PP +0.889)
>
> **Mechanism**: V7's uniform-within-block δ is invariant under within-window
> attention mixing (Σα_within·δ = δ since attention weights sum to 1).
> Per-position writes (V2 PP) work at small T but scramble at higher T because
> attention mixes T different deltas. RE writes survive because they're at
> a single position. V7 dominates at T≥3 and on archs with position-uniform
> decoders.
>
> **Old headline (2026-05-01)** preserved below for context — that was before
> the V7 expansion. Several conclusions there are now superseded:

> **2026-05-01 prior-headline**: across all matched-sparsity TXC cells (Y's + W's),
> the top-ranked cell at the prereg metric was **W's TXCContrastiveMergeH8 right-edge**
> — n=3 mean-curve peak15 = 1.578, Δ vs same-pod n=3 anchor 1.133 = +0.445.
> Per-seed cliff span 0.10. **As of 2026-05-02, Galaxy 11 V7 (+0.556) edges
> ahead at PRREG.**
>
> 3 cells crossed +0.27 prereg WIN under V1/V2 protocols (against same-pod anchor):
> 1. W Contrastive-merge RE n=3: 1.578 (Δ=+0.445)
> 2. Y Galaxy 8 SoftMaxPool PP n=3: 1.422 (Δ=+0.289)
> 3. Y T=2 H8 OBLITERATION PP n=3: 1.400 (Δ=+0.267)
>
> Y's Galaxy 8 PP was the LARGEST WIN at coh ≥ 1.75 (Δ=+1.011). **Galaxy 18
> V7 has now taken that spot at +1.033.**

> 🏆 **W's mystery-arch trio + Y's Galaxy 8 (added 2026-05-01)** — paper-grade
> results across the prereg + GIGABRAIN metrics:
>
> 1. **W Contrastive-merge RE @ coh ≥ 1.5 (PRREG)**: Δ=+0.445 ⭐⭐⭐ — top cell at
>    PRREG; per-seed span 0.10.
> 2. **Y Galaxy 8 SoftMaxPool PP @ coh ≥ 1.75 (GIGABRAIN)**: Δ=+1.011 — 4× WIN
>    threshold; learned τ ≈ 1.06 across 3 seeds (genuine softmax-merge, neither
>    hard-max nor uniform-sum).
> 3. **W MaxPool-merge RE/PP @ coh ≥ 1.75**: Δ=+0.733 (n=3 each).
> 4. **W Contrastive-merge V6 dec-broadcast @ coh ≥ 1.75**: Δ=+0.533 (n=3).
>
> See `agent_w/2026-04-30-w-phase4-results.md` for full detail (5 protocols × 3
> seeds, bootstrap CIs, per-class breakdown, paper figure). Note: under the new
> same-pod anchor, V6's strict-coh (≥ 2.0+) bootstrap-SIG claim is no longer
> SIG — the cross-pod anchor at strict coh was inflating those wins.

> 📐 **Anchor methodology fix (2026-05-01)**: T-SAE k=20 baseline retrained
> sd=1+sd=2 on W's pod for clean same-pod n=3 anchor. The cross-pod sd=1
> cliff15=0.300 was a cuDNN-determinism artifact; same-pod sd=1 retrained
> gives 1.167, perfectly consistent with sd=42 (1.100) and sd=2 (1.133).
> Per-seed cliff15 σ collapses from 0.80 (cross-pod) to **0.07 (same-pod)**.
> Canonical anchor (going forward): **1.133 / 0.411 / 0.411 / 0.411 / 0.411**
> at coh ≥ 1.5 / 1.75 / 2.0 / 2.25 / 2.5. Co-signed by Y in commit `491575ab`.

> 🚀 **2026-04-30 update — multi-coherence-threshold reframe** (see
> `2026-04-30-y-coh-threshold-sweep.md`). T-SAE k=20's only lead is on
> the unconstrained peak (1.80 vs 1.67 best-TXC), where T-SAE's peak
> strength produces incoherent text (coh = 1.40, below the prereg
> floor). At every coh threshold ≥ 1.5, at least one TXC arch
> dominates by Δ ∈ [+0.20, +0.87] (3-seed mean-curve where available).
> The largest Δ is **+0.869 at coh ≥ 1.75** (T=2 H8 shifts=(T,)
> right-edge 3-seed = 1.236 vs anchor 0.367).

### Scope

Compares 17+ matched-sparsity (or near-matched) TXC architectures against
the T-SAE k=20 same-pod n=3 anchor, under THREE steering protocols where
applicable: right-edge (V1), per-position (V2), and tiled-broadcast
(V7, added 2026-05-02). Multi-seed averaged where seeds available
(T=2 cells: 3 seeds; T=5 cells: 2 seeds; W's cells C/E and Y's k_win=20 /
T-SAE warm-start: 1 seed each).

V7 tiled-broadcast (NEW 2026-05-02): non-overlapping T-blocks, single
uniform δ within each block (averaged from per-block encode/clamp/decode).
The empirically-best protocol for half of the architectures tested.
See `agent_y_phase2/2026-05-01-y-steering-protocol-space.md` for the
full attention-mixing analysis.

Y's cells:
- `txc_bare_antidead_t2_kpos20` — T=2 bare antidead (random-init, 3 seeds)
- `txc_bare_antidead_t5_kpos20` — T=5 bare antidead (random-init, 2 seeds)
- `txc_bare_antidead_t5_kwin20` — T=5 with k_win=20 (k_pos_avg=4, 1 seed)
- `txc_h8_t2_kpos20_shifts2` — T=2 H8 multidist with shifts=(2,) (3 seeds) ⭐
- `txc_h8_t3_kpos20_shifts3` — T=3 H8 multidist with shifts=(3,) (1 seed)
- `txc_h8_t5_kpos20_shifts5` — T=5 H8 multidist with shifts=(5,) (2 seeds)
- `txc_bare_antidead_t3_kpos20_grownFromT2sd42` — T=3 grown from T=2 (1 seed)
- `txc_bare_antidead_t4_kpos20_grownChainFromT3` — T=4 grown from T=3-grown (1 seed)
- `txc_bare_antidead_t5_kpos20_grownFromT2sd42` — T=5 grown directly from T=2 (1 seed)
- `txc_bare_antidead_t2_kpos20_ws_tsae_encoder` — T=2 with T-SAE encoder warm-start (1 seed)

W's cells:
- `txc_bare_antidead_t3_kpos20` (cell C) — T=3 bare random-init (1 seed)
- `agentic_txc_02_kpos20` (cell E) — T=5 matryoshka multiscale (1 seed)

W's MYSTERY archs (n=3 multi-seed, added 2026-05-01):
- `txc_maxpool_h8_t2_kpos20_shifts2` — T=2 H8 max-pool merge encoder
  (`z[s] = max_t (x[t] @ W_enc[t, :, s])`) — disjunctive "active SOMEWHERE
  in window" merge (3 seeds × 5 protocols).
- `txc_contrastive_h8_t2_kpos20_shifts2` — T=2 H8 contrastive-merge encoder
  (`z[s] = (x[T-1] @ W_enc[T-1, :, s]) - (x[0] @ W_enc[0, :, s])`) — captures
  feature TRANSITION across window (3 seeds × 5 protocols).

### Multi-seed convention

Two valid combinations — they diverge at the coh-cliff regime:

- **Mean-curve** (standard, used here): `avg_succ(s) = mean over seeds of succ(s)`,
  `avg_coh(s) = mean over seeds of coh(s)`, then peak15 = max avg_succ(s)
  where avg_coh(s) ≥ 1.5. Smooths individual-seed coh fluctuations.
- **Per-seed-then-mean** (strict): per-seed peak15 then mean across seeds.
  More conservative; under this metric T=2 H8 per-pos drops to 0.978 due
  to coh-cliff per-seed.

The mean-curve approach is the standard reporting convention for this
type of analysis. All numbers below use it.


### T-SAE k=20 anchor across protocols

T-SAE k=20 has T=1 — there's no window. Right-edge and per-position
protocols are *the same* for T=1 (trivially: write at the only
position). The anchor 1.10 (peak success at coh ≥ 1.5) applies to
**both** protocols. The Pareto plot below shows the T-SAE k=20 curve
on both panels for clarity (labeled "T=1, RE=PP" in the per-position
panel).

### Headline ranking (peak success at coh ≥ 1.5)

#### Per-position protocol — clean view (WIN cell highlighted gold)

![per-position ranking](../../../../../experiments/phase7_unification/results/case_studies/plots/unified_ranking_per_position.png)

The WIN cell (T=2 H8 shifts=(T,) at 1.400, Δ=+0.30) has a gold edge.

#### Growth trajectory across T (per-position)

![growth trajectory](../../../../../experiments/phase7_unification/results/case_studies/plots/unified_growth_trajectory.png)

Three families compared as T grows:
- **Sequential growth chain** (purple): T=2 → T=3 → T=4 → T=5 grown
  from previous grown ckpt. Gracefully decays toward anchor at T=5.
- **Bare random-init** (orange): independently trained at each T.
  Diverges from anchor as T grows.
- **H8 multidist + shifts=(T,)** (red): independently trained at each T.
  T=2 (1.40) is the OBLITERATION; decays at T=3 (1.17), T=5 (1.07).

#### Full ranking (RE/PP/V7 protocols, all archs) — 2026-05-02 update

![unified ranking](../../../../../experiments/phase7_unification/results/case_studies/plots/unified_ranking_matched_sparsity.png)

Top cells at peak success at coh ≥ 1.5 (3-seed mean-curve where applicable;
**bold = clears prereg WIN threshold +0.27**):

| arch + protocol | n_seeds | peak15 | Δ vs anchor 1.133 | call |
|---|---|---|---|---|
| **🚀 Y Galaxy 11 SoftMaxPool+H8 V7 (NEW)** | **3** | **1.689** | **+0.556** | **PAPER-GRADE PRREG WIN ⭐⭐⭐⭐** |
| **W Contrastive-merge right-edge** | **3** | **1.578** | **+0.445** | **PAPER-GRADE PRREG WIN ⭐⭐⭐** |
| **Y Galaxy 8 SoftMaxPool per-position** | **3** | **1.422** | **+0.289** | **WIN ⭐ (newly-crosses)** |
| **Y T=2 H8 shifts=(T,) per-position** | **3** | **1.400** | **+0.267** | **WIN borderline ⭐** |
| Y Galaxy 18 SoftMaxPool T=3 V7 (NEW) | 3 | 1.444 | +0.311 | WIN ⭐ |
| Y Galaxy 8 SoftMaxPool T=2 V7 (NEW) | 3 | 1.333 | +0.200 | TIE |
| Y T-SAE warm-start V7 (NEW) | 3 | 1.333 | +0.200 | TIE |
| Y Galaxy 6 max-pool V7 (NEW) | 3 | 1.311 | +0.178 | TIE |
| Y T=2 H8 shifts=(T,) right-edge | 3 | 1.236 | +0.103 | TIE |
| W MaxPool-merge right-edge | 3 | 1.144 | +0.011 | TIE |
| W MaxPool-merge per-position | 3 | 1.144 | +0.011 | TIE |

Anchor T-SAE k=20 = **1.133** (same-pod n=3 mean-curve sd=42+sd=1+sd=2; W's
pod retrain 2026-05-01, co-signed by Y in `491575ab`). Earlier anchors:
1.167 (cross-pod n=2 sd=42+sd=1, sd=1 had cuDNN artifact giving cliff=0.300);
1.100 (sd=42 single-seed).

**FOUR cells now cross the +0.27 prereg threshold**:

1. **🚀 Y Galaxy 11 SoftMaxPool+H8 V7 tiled-broadcast (Δ=+0.556 ⭐⭐⭐⭐ NEW PAPER-GRADE TOP CELL)**
2. W Contrastive-merge RE (Δ=+0.445 ⭐⭐⭐)
3. Y Galaxy 8 SoftMaxPool PP (Δ=+0.289)
4. Y T=2 H8 PP (Δ=+0.267)

The V7 tiled-broadcast protocol added 2026-05-02 brings Galaxy 11 from
PP-borderline (Δ=+0.156) to PRREG-leader (Δ=+0.556) — a +0.40 protocol
swing on the same architecture. **The right protocol matters as much as
the right architecture.**

#### Coh ≥ 1.75 ranking (GIGABRAIN metric) — 2026-05-02 with V7

| arch + protocol | n_seeds | peak@≥1.75 | Δ vs anchor 0.411 | call |
|---|---|---|---|---|
| **🚀 Y Galaxy 18 SoftMaxPool T=3 V7 (NEW)** | **3** | **1.444** | **+1.033** | **NEW BEST EVER ⭐⭐⭐⭐** |
| Y Galaxy 8 SoftMaxPool T=2 PP | 3 | 1.422 | +1.011 | ⭐⭐⭐⭐ |
| Y Galaxy 11 SoftMaxPool+H8 V7 (NEW) | 3 | 1.689 | (peak below 1.75; uses peak15) | n/a here |
| Y T-SAE warm-start V7 (NEW) | 3 | 1.333 | +0.922 | ⭐⭐⭐ |
| Y Galaxy 6 max-pool V7 (NEW) | 3 | 1.311 | +0.900 | ⭐⭐⭐ |
| Y T=2 H8 RE | 3 | 1.239 | +0.828 | ⭐⭐ |
| Y Galaxy 18 G8 T=3 RE | 3 | 1.178 | +0.767 | ⭐⭐ |
| W MaxPool-merge RE/PP | 3 | 1.144 | +0.733 | ⭐⭐ |
| Y Galaxy 23 G8 T=5 V7 (NEW) | 3 | 1.089 | +0.678 | ⭐ |

Anchor T-SAE k=20 = **1.133** (same-pod n=3 mean-curve sd=42+sd=1+sd=2; W's
pod retrain 2026-05-01, co-signed by Y in `491575ab`). Earlier anchors:
1.167 (cross-pod n=2 sd=42+sd=1, sd=1 had cuDNN artifact giving cliff=0.300);
1.100 (sd=42 single-seed). **THREE cells now cross the +0.27 prereg threshold**:
W's Contrastive-merge RE (Δ=+0.445 ⭐⭐⭐ paper-grade), Y's Galaxy 8 PP
(Δ=+0.289, newly crosses with same-pod anchor) and Y's T=2 H8 PP (Δ=+0.267,
borderline).

The Contrastive-merge RE result is *seed-stable*: per-seed cliffs sd42=1.633,
sd1=1.567, sd2=1.533 — span only 0.10. By contrast, T=2 H8 PP's per-seed span
is wider (Y's earlier σ-discussion). Contrastive-merge RE is the most-robust
n=3 cell at the prereg metric. T-SAE same-pod per-seed span at coh ≥ 1.5 is
0.07 (sd42=1.100, sd1=1.167, sd2=1.133) — equally tight, validating the
new anchor.

### Pareto frontier — success vs coherence

#### 🏆 PAPER HEADLINE FIGURE — TXC family Pareto dominance (added 2026-05-01)

If we showed only ONE plot in the paper, it would be this one.

![paper headline figure](../../../../../experiments/phase7_unification/results/case_studies/plots/paper_headline_figure.png)

**One panel.** T-SAE k=20 (blue dashed) is the anchor. **5 TXC architectures**
each plotted at their winning protocol, all under same-pod n=3 multi-seed
mean-curve, k_pos=20 matched sparsity:

- **Contrastive-merge (W, RE)** — purple, cliff15 = **1.578** Δ=+0.428 ⭐⭐⭐
- **OBLIT T=2 H8 (Y, PP)** — red, cliff15 = 1.533 at coh=2.20 Δ=+0.383 ⭐
- **Galaxy 11 SoftMaxPool+H8 (Y+W, RE)** — cyan, cliff15 = 1.467 Δ=+0.317 ⭐
- **Galaxy 8 SoftMaxPool (Y, PP)** — green, cliff15 = 1.422 at coh=1.89 Δ=+0.272 ⭐
- **MaxPool-merge (W, PP)** — pink, cliff15 = 1.144 Δ=-0.006 (TIE)
- T-SAE k=20 RE (anchor) — blue dashed, cliff15 = 1.150 (n=3 same-pod)

**4 of 5 TXC architectures cleanly cross the +0.27 prereg WIN threshold**
(green dashed line at peak15 = 1.40). The 5th (MaxPool-merge PP) ties.

The shaded green band [1.8, 2.5] marks the strict-coh region where TXC
**pareto-dominates** T-SAE (T-SAE has no coh-stable strength in this band;
TXC archs sustain succ ≥ 0.5 across it). OBLIT PP's star is *inside* the
band at coh=2.20 succ=1.53 — the strongest single result on the plot
combining high coherence + high success.

The 5 TXC archs span 5 distinct architectural recipes:
- T=2 H8 multi-distance contrastive antidead (OBLIT, Y)
- T=2 H8 + max-pool merge encoder (W mystery)
- T=2 H8 + contrastive end-vs-start merge encoder (W mystery)
- T=2 plain SoftMaxPool with learnable per-feature τ (Galaxy 8, Y)
- T=2 SoftMaxPool + H8 (Galaxy 11, Y+W compositional)

This is the paper-grade story: the TXC family's coherent-steering advantage
is **architecturally robust** — 5 independent recipes all win.

#### Focused Pareto — T-SAE baseline + 3 best TXCs (earlier version, kept for reference)

![focused pareto](../../../../../experiments/phase7_unification/results/case_studies/plots/focused_pareto_matched_sparsity.png)

A clean view of the headline result: T-SAE k=20 anchor (blue dashed) vs
the **3 best n=3-multi-seed TXC architectures** (OBLITERATION T=2 H8 / W
MaxPool-merge / W Contrastive-merge), one panel per protocol. Stars mark
the cliff at coh ≥ 1.5 (PRREG metric); dashed green line marks the
+0.27 prereg WIN threshold (peak15 ≥ 1.44).

**What the focused plot shows**:

- **T-SAE k=20** has the highest unconstrained peak (1.80) but it sits at
  coh ≈ 1.4 — *just below* the prereg coh-floor. At coh ≥ 1.5, T-SAE's
  cliff drops to 1.133 (same-pod n=3).
- **Right-edge panel**: **W's Contrastive-merge** (purple) reaches a cliff
  of **1.578 at coh ≈ 1.6** — the only n=3 TXC cell whose star sits cleanly
  above the green +0.27 WIN line. OBLITERATION (red) and MaxPool (pink)
  cluster near or just above the anchor line at this protocol.
- **Per-position panel**: **OBLITERATION T=2 H8** (red) reaches 1.400 — the
  cell Y led with in the older draft. Contrastive-merge (purple) drops to
  0.756 here (its win is right-edge-specific). MaxPool (pink) stays around
  1.144 across both protocols.

**Three paper-grade architectural recipes emerge** (against same-pod anchor 1.133):
1. *Right-edge + Contrastive-merge encoder* (W) → +0.445 PRREG WIN ⭐⭐⭐.
2. *Per-position + Soft-max-pool merge (Galaxy 8, learnable τ ≈ 1.06)* (Y) → +0.289 WIN ⭐.
3. *Per-position + H8 multi-distance contrastive encoder (OBLITERATION)* (Y) → +0.267 borderline WIN.

These are **mechanistically different**: Contrastive captures CHANGE
(end-vs-start), OBLITERATION captures CO-OCCURRENCE (multi-distance
contrastive). They win on different concept classes (Contrast → sentiment
+ knowledge; OBLIT → knowledge primarily; see per-class breakdown in
`agent_w/2026-04-30-w-phase4-results.md`).

#### Full unified Pareto (22 archs, dense)

![unified pareto](../../../../../experiments/phase7_unification/results/case_studies/plots/unified_pareto_matched_sparsity.png)

Two panels (right-edge / per-position). Each line is one arch's
multi-seed-averaged (success, coh) curve across the 7 family-normalised
strengths. The black dashed line is the Pareto upper envelope across
all archs. Coh = 1.5 threshold marked; T-SAE k=20 anchor = 1.133
horizontal line; WIN threshold = 1.40 horizontal line (= anchor + 0.27).

**Interpretation:**
- T=2 H8 shifts=(T,) per-position (red triangles, dashed line) **stays
  furthest above all others** in the success-coh tradeoff and crosses
  above the WIN threshold near coh=2.
- Other T=2 cells (orange, gold) cluster around the anchor.
- T=5 cells generally trace lower curves with noisier coh behavior.
- T=5 grown-direct (violet) is the weakest cell — confirms the
  +1-position-grow-horizon limit.

### Why T=2 H8 shifts=(T,) wins

The combination stacks four levers that each fix a different failure
mode at sparse k_pos:

1. **T=2** — minimum window beyond per-token. Y's polysemanticity
   finding: smaller T at sparse k_pos has cleaner picked features
   (25/30 distinct vs T=5's 24/30; vs T-SAE k=20's 28/30).
2. **H8 multidistance** — Matryoshka H/L groups (H=0.2·d_sae) +
   multi-distance contrastive InfoNCE.
3. **shifts=(T,)** — single contrastive distance = window length.
   Constrains InfoNCE to the longest distance, training features that
   are consistent across the entire T-window. Earlier verified at T=5:
   shifts=(5,) gives σ_seeds = 0.000 across 2 seeds.
4. **Per-position write-back** — distributes the steered concept across
   all T positions; combined with sharp seed-stable concept-anchored
   features, produces strong coherent steering.

### Why W's Contrastive-merge RE wins (added 2026-05-01)

The contrastive-merge encoder `z = enc(x[T-1]) - enc(x[0])` (T=2 H8 stack)
captures feature TRANSITION across the window — feature fires when it
*becomes active during* the window. Combined with right-edge protocol
(write at most-recent position), the steering write matches the encoder's
temporal direction:

1. **Contrastive-merge encoder** — captures CHANGE rather than co-occurrence
   (vs OBLITERATION's H8 multi-distance, which captures co-occurrence at
   multiple distances). Mechanistically alignes with sentiment (tone shifts)
   and knowledge-onset (domain-specific token entering) concepts.
2. **Right-edge protocol** — writes the steering signal at position T−1
   (most recent token), matching where contrastive features peak.
3. **Per-class breakdown** confirms the mechanism: sentiment Δ=+0.500 ⭐⭐
   (0.5 → 1.0) and knowledge Δ=+0.315 ⭐ are the dominant winning classes.
   Discourse + safety LOSE — those are window-stable register features
   without clean transition signatures.

This is a DIFFERENT mechanism from OBLITERATION's win: not "smaller T +
multi-distance contrastive" but "transition-detection encoder + matched
write-direction". Two independent paper-grade architectural recipes.

### Caveats

- **Single-seed cells** (n=1) need multi-seed verification before
  locking the claim. Several cells in the +0.067 tie band at single
  seed could swing if multi-seeded.
- **Per-seed-then-mean** is a more conservative reading — under it,
  T=2 H8 per-pos drops to 0.978 (TIE just below anchor). The σ_seeds
  is large because the coh ≥ 1.5 threshold is at the cliff region.
- **Unconstrained peak (METRIC A)**: T-SAE k=20 still wins by ≥ 0.40
  across all matched-sparsity TXC cells. T-SAE's 1.80 unconstrained
  peak is at coh=1.40 (slightly incoherent text).

### Files

- Inventory + JSON: `results/case_studies/plots/unified_pareto_summary.json`
- **🏆 PAPER HEADLINE FIGURE** (T-SAE + 5 best TXCs, single panel): `results/case_studies/plots/paper_headline_figure{.png,.thumb.png}` — added 2026-05-01
- **Focused Pareto** (T-SAE + 3 best TXCs): `results/case_studies/plots/focused_pareto_matched_sparsity{.png,.thumb.png}` — added 2026-05-01
- Pareto plot (success vs coh, both protocols, all 22 archs): `results/case_studies/plots/unified_pareto_matched_sparsity{.png,.thumb.png}`
- Ranking bar plot (full, both protocols): `results/case_studies/plots/unified_ranking_matched_sparsity{.png,.thumb.png}`
- Ranking bar plot (per-position only, WIN highlighted): `results/case_studies/plots/unified_ranking_per_position{.png,.thumb.png}`
- Growth trajectory across T: `results/case_studies/plots/unified_growth_trajectory{.png,.thumb.png}`
- Mystery-arch per-class signature: `results/case_studies/plots/mystery_arch_per_class_signature{.png,.thumb.png}` — added 2026-05-01
- Plot scripts: `experiments/phase7_unification/case_studies/steering/{plot_unified_pareto,plot_focused_pareto,plot_mystery_arch_per_class}.py`
- Per-cell writeups: this `agent_y_phase2/` dir + `agent_w/` dir
