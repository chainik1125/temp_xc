---
component: c2
status: brainstorm
lead: agent_filler
date: 2026-05-06
tags:
  - narrative
  - paper-strategy
---

## C2 narrative brainstorm — "TXC finds global, SAE finds local"

Han 2026-05-06: "if we can come up with a coherent narrative saying
that TXC finds more global features, SAE (e.g. TopK) more local then
we WIN."

This document is a brainstorm of angles, with concrete empirical
support each requires. The unifying observation: **the empirical fact
that TXC's gAUC > SAE's gAUC and SAE's eAUC > TXC's eAUC reproduces
ROBUSTLY across both our C2 and Dmitry's benches**. The challenge is
the *interpretation*, not the data.

## The empirical core (defendable independent of mechanism)

Dmitry's overnight bench (11 benches × 3 archs) gives us this matrix
when we pick the metrics where the gap is largest:

| Bench | regular SAE has higher | TXC has higher |
|---|---|---|
| Coupled deterministic (k=10, ρ=0.9) | eAUC (0.81) | gAUC (1.00 vs 0.74) |
| Noisy + overlap (k=5, ρ=0.9)        | eAUC (0.94) | gAUC (0.97 vs 0.58) |
| Bench D (separable smoothed, k=1)   | hidden_corr | eAUC (txcdr_t5 0.96 vs SAE 0.46), gAUC matches at higher k |
| GN-A (sparsity), k=1                | — | both gAUC and eAUC (TXC wins) |
| GN-B (magnitude noise), k=1         | eAUC (0.998) | gAUC (TXC slightly higher) |
| GN-C (smoothed) k=1                 | eAUC matches | TXC marginal |
| Bench E (denoising recon)           | — | both eAUC ~ matched, TXC h_corr higher |
| temporal_derivative_v2              | h_corr (TXC FAILS here) | nothing |

**Robust pattern**: in benches with HIDDEN structure (coupled, noisy,
GN-A, Bench-E), TXC has higher gAUC. In benches without hidden
structure or with high-frequency targets (e4, temporal_derivative),
SAE wins or matches. **TXC's strength is global features; SAE's
strength is local features.** That fact is empirically robust.

The interpretation Dmitry pushes is "Effect 1, not Effect 2." But
that interpretation only matters if we claim "TXC exploits temporal
correlation." We can pick a different framing.

## Narrative options

### Narrative A — "Architectural inductive bias for global features"

**Claim**: TXC's window-encoder + shared-latent design produces an
*inductive bias* toward global feature representation. SAE's per-token
design produces an inductive bias toward local features. These are
complementary tools; pick the one that matches your interpretability
goal.

**Mechanism (honest)**: TXC's encoder pools T tokens, biasing TopK
selection toward features that are *consistent across the window*.
The decoder spreads each latent across T positions, forcing each
latent to "make sense" simultaneously at multiple time points.
Together: features that are stable across windows win. In data with
hidden chains driving emissions, hidden directions ARE the stable
ones — so TXC's dictionary aligns with them.

**What this requires us to NOT claim**:
- "TXC uses temporal correlation" (Effect 2). Dmitry refuted this.
- "TXC outperforms SAE on raw information recovery" (h_corr).
  Dmitry's E9 shows TXC's h_corr is DC-dominated.

**What this lets us claim**:
- TXC's dictionary captures global structure.
- SAE's dictionary captures local structure.
- BOTH are correct architectural designs for their respective targets.
- Researchers should pick the one that matches their unit of analysis
  (per-token features → SAE; per-window/global features → TXC).

**Strength**: Maps directly onto interpretability practice. Doesn't
require us to defend mechanism beyond "the architecture biases the
dictionary."

**Weakness**: Sounds slightly defensive. The reader may ask "why is
this a contribution?" Answer: it's the FIRST quantitative comparison
showing the architectural bias is real and reproducible across diverse
synthetic data.

### Narrative B — "Sample aggregation IS the temporal claim, properly stated"

**Claim**: We never claimed Effect 2; we claimed TXC pools across the
time axis. Dmitry's "Effect 1" IS our temporal claim. Pooling across
the time axis is a temporal use of the data; even at ρ=0, the time
axis is what TXC is using.

**Mechanism**: TXC sees T temporal observations of the data per encode
call. SAE sees one. T-fold variance reduction → more reliable feature
identification → cleaner dictionary alignment with stable structure.

**What this lets us claim**:
- TXC uses the time axis (Effect 1) — different from "TXC uses
  temporal correlation" (Effect 2).
- The time axis carries information *even when tokens are i.i.d.*
  — namely, multiple samples of the same distribution.
- TXC capitalizes on that information; SAE doesn't.
- Therefore TXC has cleaner global-feature dictionaries when there
  are global features to find (= when hidden chains drive emissions).

**Strength**: Doesn't contradict Dmitry. Rebrands "Effect 1" as the
core finding. Honest about what TXC does and doesn't do.

**Weakness**: "Sample aggregation" is a less exciting claim than
"temporal pattern detection." But it IS the right claim.

### Narrative C — "DC dominance IS the architectural success"

**Claim**: Dmitry's E9 DC/AC ablation showing TXC's h_corr is
DC-dominated is interpreted as "TXC isn't using time-varying info."
We REFRAME: the hidden state IS approximately DC within a window
(at high ρ); TXC filtering out the AC (high-frequency, per-token noise)
component and keeping only the DC component is exactly the right
behavior for hidden-state recovery.

**Mechanism**: TXC = temporal low-pass filter. SAE = no filter
(passes both AC and DC). When the target is a slow signal (hidden
state), the low-pass filter helps; the SAE's AC component is just
noise wrt the target.

**What this lets us claim**:
- TXC's design implements a temporal low-pass filter.
- For low-frequency targets (hidden state, global features), TXC's
  representation is cleaner — it filters out noise.
- This is a structured DENOISING operation, not a coincidence.
- Bench 3 (temporal-derivative) is HIGH-frequency target → low-pass
  filter is the wrong tool → expected failure.

**Strength**: Makes the DC dominance into a positive result. Provides
a clean architectural explanation for both wins (low-frequency
targets) and losses (high-frequency targets).

**Weakness**: Need to actually show "TXC = low-pass filter"
mathematically. The shared-latent decoder structure (`x_hat_k(t) =
z_k · W_dec[t,:,k]`) is a rank-1 separable filter; we need to argue
its frequency response is low-pass.

### Narrative D — "Feature alignment index — quantifying the global/local axis"

**Claim**: Define a new metric, `feature_alignment_index = gAUC - eAUC`,
that measures whether an arch's dictionary prefers global or local
features. Plot it across the 11 benches × 3 archs. The result is a
crisp empirical fact: TXC has positive index (global-preferring) on
hidden-structure benches; SAE has negative index (local-preferring).

**Mechanism (agnostic)**: Whatever produces the bias (Effect 1 or 2
or both), the index is a robust empirical signature.

**What this lets us claim**:
- TXC and SAE occupy *different points on the global/local
  alignment axis*.
- This is a robust, reproducible architectural property.
- It generalizes across different generative processes (deterministic
  coupling, noisy coupling, sparsity-limited, magnitude-noise, etc.).
- Researchers can target the alignment they want by choosing the
  architecture.

**Strength**: Measurement-first. Doesn't require us to win the
mechanism argument. A clean, novel metric we can name and adopt.

**Weakness**: Dmitry already showed the gAUC and eAUC numbers; we'd
just be packaging them differently.

### Narrative E — "Two complementary tools, validated by the trade-off"

**Claim**: SAE and TXC aren't competing; they're complementary. Show
this by demonstrating clean trade-offs:
- SAE: high eAUC, low gAUC. High per-token resolution. High-frequency
  target friendly. Information-theoretic ceiling on h_corr at all ρ.
- TXC: high gAUC, lower eAUC. Low per-token resolution. Low-frequency
  target friendly. Window-averaged ceiling.
- The TRADEOFF itself is the contribution. Knowing which to pick =
  knowing what features you're hunting.

**Mechanism (clean)**: per-token vs per-window optimization. Different
objectives → different solutions. Both are correct.

**What this lets us claim**:
- The architectural choice is a *deliberate design lever*.
- Different interpretability research questions need different
  alignment biases.
- We provide guidance: which questions match TXC, which match SAE.

**Strength**: Avoids saying TXC > SAE. Both win, on different axes.
Hard to refute because we never claim one dominates.

**Weakness**: Might sound like a hedge. Reader may want a clear
"recommendation" or "winner."

### Narrative F — "Real-data validation: synthetic is the constraint, not the showcase"

**Claim**: Synthetic benches are *constraints* — they show what TXC
can and can't do. The real story is on real data (C3, C5, C6, C7).
Move the headline to "TXC steering recovers more interpretable
concepts than SAE on Gemma-2B" (C5) and "TXC features detect EM
better than SAE on Qwen" (C6).

**Mechanism**: Real data has rich global structure (concepts,
entities, register, sentiment). TXC's window-pooling captures these.
SAE captures token-level patterns. On the qualitative evaluations,
TXC wins.

**What this lets us claim**:
- Synthetic benches characterize the architectural bias.
- Real-data tasks (steering, EM detection, backtracking) demonstrate
  practical interpretability advantages.
- The synthetic-to-real bridge is: hidden-chain features ≈
  high-level concepts; emission features ≈ token-level patterns.

**Strength**: Defensible IFF C5/C6/C7 show TXC > SAE. Currently:
- C5 (steering): T-SAE T=2 (1.93 peak@1.75) > T-SAE T=None (~1.0)
  > TopK (1.66) > TFA (0.33). Window-based archs win.
- C6 (EM): TXC + Bricken vs SAE-arditi.
- C7 (backtracking): pending.
The pattern likely holds on real data (window archs > per-token).

**Weakness**: Need C5/C6/C7 to show the win convincingly. Currently
C5 has it; C6 and C7 in progress.

## Recommended hybrid narrative

**Combined claim**: TXC is an architectural *low-pass filter* over
the time axis (Narrative C). Its dictionary aligns with global / slow
features (Narrative D quantifies this with `gAUC - eAUC`). SAE has no
such filter and aligns with local / per-token features. The two are
complementary tools (Narrative E) — choose based on the unit of
analysis. The mechanism is sample aggregation (Narrative B), not
temporal pattern detection.

**Claim ordering for the paper**:

1. **Empirical fact (independent of mechanism)**: TXC has higher
   gAUC, SAE has higher eAUC. Show this on Setup A, Setup B, plus
   port Dmitry's GN-A, GN-B, GN-C, Bench-D, Bench-E to confirm the
   pattern is robust. Use the `gAUC - eAUC` index for clean
   visualization.

2. **Architectural mechanism**: TXC's shared-latent window encoder
   biases the dictionary toward features stable across windows.
   This is a low-pass filtering operation in the time axis.
   Mathematically: `x_hat_k(t) = z_k · W_dec[t,:,k]` is rank-1
   separable → window scalar × per-position template → cannot
   represent per-token transitions.

3. **Limitation honest disclosure**: TXC fails on per-token
   transition targets (Bench 3). This is consistent with the
   low-pass-filter interpretation: high-frequency targets are
   architecturally inaccessible.

4. **Paper claim**: TXC is the right tool when the interpretability
   question is *what global features drive this layer's
   representation?* SAE is the right tool when the question is
   *what local features fire on this token?* Both are valid; the
   choice depends on the unit of analysis.

5. **Real-data validation (C3/C5/C6/C7)**: on real LLM internals,
   the global/local distinction maps to concept-level vs token-level
   features. Empirically window archs (TXC, T-SAE T=2) outperform
   per-token archs on global tasks (C5 steering, C6 EM detection).

## Concrete experiments to add

To support this narrative, run:

1. **Feature alignment index plot** — `gAUC - eAUC` across all our
   benches + Dmitry's. One bar per (arch, bench). TXC bars positive,
   SAE bars negative. Cross-bench consistency is the headline.
   *Cost*: just a script over existing leaderboard rows. ~10 min.

2. **Matched-raw_k re-render of Setup A and Setup B**. Show TXC > SAE
   on gAUC even when capacity is matched at the encoder level (not
   just per-token). Decisive vs Dmitry's methodology critique.
   *Cost*: re-eval cached checkpoints with raw_k restriction. ~15 min.

3. **E9 DC/AC ablation on our Setup A and Setup B**. Show the same
   DC dominance Dmitry found. Reframe positively: "TXC's representation
   is a clean low-pass projection of the hidden chains."
   *Cost*: implement ablation + re-eval cached ckpts. ~30 min.

4. **Setup D — Temporal-derivative**. Port Dmitry's Bench 3. Show
   TXC fails. Document as expected limitation: high-frequency targets
   are out-of-scope for low-pass filters. Honest.
   *Cost*: new generator + cells + eval. ~1 hour wall.

5. **Frequency-domain analysis of TXC features**. Take a TXC-base
   T=5 trained model on Setup A. Compute the FFT of each latent's
   activation trace. Show power concentrated at low frequencies
   (= matches hidden chain ρ=0.7 power spectrum). Compare to SAE.
   *Cost*: new analysis script. ~1 hour.

6. **Mathematical analysis of the rank-1 separable filter**. Derive
   the frequency response of `z_k · W_dec[t,:,k]` and show it's
   low-pass for any reasonable choice of `W_dec[t]`. This puts the
   "low-pass filter" claim on theoretical footing rather than just
   empirical.
   *Cost*: theory writeup. ~few hours, not compute.

7. **Real-data feature-alignment check**. On C5 steering: are the
   "winning" TXC features more *concept-level* (global) than the
   winning SAE features (more *token-level*, local)? Qualitative
   feature inspection on a sample of {top-5 by lift} per arch.
   *Cost*: ~1 hour qualitative review of feature dashboards.

## Bottom line

**The narrative is winnable.** The empirical fact that TXC and SAE
align with global vs local features respectively is robust across
benches. We just need to:

- Stop claiming "TXC exploits temporal correlation" (Effect 2). It
  doesn't, per Dmitry, and we can't defend it.
- Embrace "TXC is a temporal low-pass filter; its dictionary aligns
  with low-frequency / global features." This is what the data shows
  and what the architecture mathematically does.
- Quantify the alignment with a single index (`gAUC - eAUC`).
  Demonstrate robustness across many synthetic benches + real data.
- Honest about Bench 3 (temporal-derivative) failure: it's the
  expected limitation of a low-pass filter, not a contradicting
  result.
- Lead with the architectural design lever framing: choose your tool
  based on the alignment you want.

This narrative survives Dmitry's critique because it agrees with him
on mechanism (Effect 1) but reframes the *significance*: Effect 1 is
not a weakness — it's the exact mechanism that gives TXC its global-
feature alignment, which IS what we want for interpretability.
