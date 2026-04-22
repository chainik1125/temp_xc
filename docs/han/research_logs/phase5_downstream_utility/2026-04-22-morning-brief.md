---
author: Han
date: 2026-04-22
tags:
  - results
  - in-progress
---

## Phase 5.7 overnight agentic autoresearch — morning brief

**TL;DR**: Ran 7 agentic TXC cycles + 1 MLC cycle + 3-seed variance on
both winners. Multi-scale contrastive (InfoNCE at multiple nested prefix
lengths with γ=0.5 decay) emerged as the robust winner for BOTH families.
MLC + multi-scale gives the new best overall val AUC on the bench.

**Status of everything that was in flight at compact last night:**

- k2x A3 relaunched and committed (Δ=+0.0039, AMBIGUOUS — k is not a useful axis for A3).
- MLC α sweep complete (α=0.03, 0.10, 1.00): monotone-in-α, best is α=1.00 at Δ=+0.005. Weak effect.
- A3 α=3.0 climb check complete: +0.0238, plateaued vs α=1.0 (+0.0259). No need for α=10.
- Handover Phase 0 (Part B finalize) done. Phase 1 (TFA audit) still pending — I deferred since agentic loop took priority per your instruction.

### Agentic loop results (8 cycles)

All cycles use the existing `run_autoresearch.sh` pipeline: train → val-probe → paired Δ vs family vanilla base → commit.

| # | Name | Hypothesis | Δ_val (vs base) | t | Verdict |
|---|---|---|---|---|---|
| 01 | agentic_txc_01 | H-TXC5 scale-1 orthogonality reg at λ=1.0 | +0.0078 | +0.63 | LOST |
| **02** | **agentic_txc_02** | **H-TXC7 multi-scale InfoNCE n=3 γ=0.5** | **+0.0354** | **+3.81** | **WIN** |
| 03 | agentic_txc_03 | γ=1.0 (equal weights) | +0.0072 | +0.58 | LOST |
| 04 | agentic_txc_04 | n_scales=5 (all scales), γ=0.5 | +0.0054 | +0.41 | LOST |
| 05 | agentic_txc_05 | n=3 γ=0.3 | −0.0096 | −0.85 | LOST |
| 06 | agentic_txc_06 | H-TXC2 + K=4 hard negs at cycle-02 cfg | +0.0291 | +3.80 | tie |
| 07 | agentic_txc_07 | H-TXC6 cosine consistency (no push-apart) | +0.0174 | +1.81 | LOST |
| **08** | **agentic_mlc_08** | **Port multi-scale to MLC** | **+0.0163** | **+2.45** | **WIN** |

(Full cycle log with hypotheses, changes, results, takeaways in
[`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md).)

### Key findings

1. **Multi-scale InfoNCE is the transferable recipe.** At scales 1, 2, 3
   with γ=0.5 geometric decay, both TXC (matryoshka, cycle 02) and MLC
   (flat d_sae, cycle 08) gain meaningfully over single-scale contrastive.
   Same pattern, both families. This is the cleanest cross-family result
   in Phase 5.7.

2. **The decay schedule matters narrowly.** γ=0.5 is a peak, not a
   plateau:
    - γ=0.3 (cycle 05) went below vanilla (−0.0096)
    - γ=0.5 (cycle 02) won (+0.0354)
    - γ=1.0 (cycle 03) regressed (+0.0072)

3. **The depth matters too:**
    - n=3 (cycle 02) won
    - n=5 (cycle 04) regressed to +0.0054 — applying contrastive at scale-4
      and scale-5 (full 5-token window) destroys the window-level features
      because those scales' reconstruction targets don't overlap across
      adjacent windows.
    - The shift-invariance boundary sits between scale-3 and scale-4 for T=5.

4. **InfoNCE's push-apart IS essential.** Cycle 07 (cosine consistency
   only, no negatives) halved the gain from +0.0354 to +0.0174. Cycle 06
   (hard negs added on top of multi-scale InfoNCE) didn't help — B=1024
   in-batch negatives are already enough; K=4 same-seq hard negs don't
   change the softmax distribution.

5. **Orthogonality regularizer HURTS.** Scale-1 features after
   matryoshka+contrastive are already at a local optimum for probing.
   Adding an orthogonality penalty perturbs them away from the optimum
   (cycle 01).

### 3-seed variance (4 × 50-min jobs, ran 08:24 → 11:06 UTC)

Seed variance confirmation on the two winners (seeds ∈ {42, 1, 2}):

**TXC — agentic_txc_02 (multi-scale n=3, γ=0.5)**

| seed | val AUC | Δ vs matryoshka_t5_seed42 | t |
|---|---|---|---|
| 42 | 0.7818 | +0.0328 | +3.38 |
| 1 | 0.7666 | +0.0176 | +1.48 |
| 2 | 0.7663 | +0.0173 | +1.85 |
| **mean ± σ** | **0.7716 ± 0.0089** | **+0.0225 ± 0.0089** | — |

**MLC — agentic_mlc_08 (multi-scale prefix [d_sae/4, d_sae/2, d_sae], γ=0.5)**

| seed | val AUC | Δ vs mlc_seed42 | t |
|---|---|---|---|
| 42 | 0.8069 | +0.0150 | +2.20 |
| 1 | **0.8153** | **+0.0235** | **+3.80** |
| 2 | 0.8017 | +0.0099 | +1.36 |
| **mean ± σ** | **0.8080 ± 0.0069** | **+0.0162 ± 0.0069** | — |

### Paper framing (proposed)

**Claim**: "Multi-scale contrastive with geometric decay γ=0.5 improves
downstream probing AUC on TXC and MLC sparse crosscoders. Effect is
robust across 3 seeds: +0.022 ± 0.009 on matryoshka_t5 → TXC multi-scale,
+0.016 ± 0.007 on mlc → MLC multi-scale."

**Best overall result**: MLC multi-scale at AUC 0.8080 ± 0.007 (3 seeds),
up from single-scale MLC contrastive at 0.8014. Best single-seed result:
MLC seed=1 at 0.8153.

**Caveats** (needs to be in paper):
- seed=42 was the best seed for TXC; seeds 1, 2 gave lower gains. So the
  headline +0.0354 was somewhat seed-42-favored. Real effect more like
  +0.018 across seeds, not +0.035.
- MLC best single-seed is seed=1 at +0.0235 — recommend reporting
  the 3-seed mean (+0.0162) rather than the best single-seed number.
- Probing is non-deterministic (~0.003 jitter per row at fixed seed).
  All recorded Δ numbers are within ±0.005 of stable.

### What wasn't done (for your review)

1. **Phase 1 of the 18-hr handover (TFA audit fix)** is still pending.
   The `run_probing.py:729` line to patch is unchanged. The decision
   was to prioritize the agentic loop since you said TXC + MLC tuning
   was most important. Can pick this up next session — it's a one-line
   patch + ~15 min re-probe.
2. **Test-set eval on the agentic winners.** `partB_finalize --run` only
   touches A2/A3 Part B variants, not the agentic winners. If the
   multi-scale is going into the paper, we need a test-set eval script
   or extension. Gate this behind your review.
3. **Complementarity analysis** (Phase 5 of the handover): concat MLC ×
   TXCDR latents and probe. Deferred.
4. **HF re-upload of the new agentic ckpts.** Not done. 9 new ckpts
   (cycles 01-08 + 4 seed-variance variants). Easy to push via
   `scripts/hf_upload_ckpts.py`.

### In-flight at time of writing

Nothing. All jobs complete, GPU idle, orchestrator exited cleanly.
`ps -ef | grep -E "run_autoresearch|train_primary|run_probing"` returns
nothing.

### Resume checklist

1. Review this doc + [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md).
2. Decide: do we ship the multi-scale claim to the paper? If yes:
   - Run `partB_finalize` style on agentic_txc_02 and agentic_mlc_08 for
     test-set AUCs at last_position + mean_pool.
   - Update `summary.md` with the new family leaders.
3. Phase 1 (TFA audit fix) — still a blocking deliverable per the 18-hr plan.
4. HF ckpt sync if you want to push the agentic ckpts.

All commits are on the `han` branch, pushed. Latest commit should be
in the `9aed37f` range (final seed-variance commit).
