---
author: Han
date: 2026-04-18
tags:
  - proposal
  - in-progress
---

## Phase 5 briefing: downstream utility of temporal SAE features

This document briefs a fresh agent to pick up Phase 5. Read it top-to-bottom before touching anything. If a fact here contradicts the codebase or `project_brief.md`, trust the code and flag the discrepancy.

### Why Phase 5 exists

Phases 2--4 established that TXCDR recovers ground-truth features better than per-token SAEs at matched sparsity in toy settings (Phase 2, 3) and that on Gemma-2-2B-IT the three architectures fill disjoint regions of direction-space (Phase 4). None of these results demonstrate **downstream utility** --- whether TXCDR features are differentially useful for a concrete interpretability task.

The motivating paper is `papers/are_saes_useful.md` (Kantamneni et al., 2025). Headline finding: on 113 probing datasets across four difficult regimes, SAE probes fail to beat baseline methods (logistic regression, PCA, KNN, XGBoost, MLPs) when matched against *strong* baselines. The one regime where SAEs *looked* useful --- detecting attributes distributed over multiple tokens --- collapses when attention-pooled baselines are added. This is Phase 5's ceiling: if a *temporal* SAE is useful, it should be useful *here*.

### What Aniket already did

Branch `aniket`, `docs/aniket/experiments/sparse_probing/summary.md`. He ran SAEBench sparse probing on 8 tasks, comparing:

- Regular SAE (single-token TopK)
- MLC (Multi-Layer Crosscoder, shared latent across layers {10, 11, 12, 13, 14})
- TempXC / TXCDR at T in {5, 10, 20}

Headline:

| arch   | T  | mean acc | wins / 8 |
|--------|----|----------|----------|
| SAE    | -- | 0.8545   | --       |
| MLC    | 5  | 0.9406   | **8 / 8** |
| TempXC | 5  | 0.8615   | 5 / 8    |
| TempXC | 10 | 0.8254   | --       |
| TempXC | 20 | 0.7737   | --       |

MLC dominates; TempXC barely edges SAE at T=5 and degrades monotonically with T. This is a negative initial result for TXCDR.

**Treat Aniket's initial exploration as a useful starting point but not the final methodology.** His summary is honest about its own limitations; we should be at least as honest, and we should actively challenge its assumptions. Four specific concerns Phase 5 must address:

1. **Convergence.** Aniket's own summary flags TempXC as undertrained --- T=5 still losing 5.9% of its loss per 1k steps at step 5000; SAE and MLC are plateaued. A 15--20k-step rerun is planned but not run.
2. **Missing baselines.** Aniket ran a vanilla single-token SAE. **Stacked SAE, Shared-weight per-position SAE, TFA, and TFA-pos** are all missing. Our Phase 4 work shows TFA and Stacked SAE fill qualitatively different feature territories than TXCDR; omitting them means we cannot tell whether MLC wins because it's multi-axis or because TempXC's specific shared-latent-over-time design is weak.
3. **Possible data leakage.** Confirm: what text was the SAE/MLC/TempXC encoder trained on, and does any of it overlap with SAEBench probe train/test? If yes, the probe is partly memorizing, not generalizing. Check before anything else.
4. **Architecture space unexplored.** Vanilla TXCDR is one point on a much larger design space. Matryoshka-temporal, causal TXCDR, and a Time×Layer joint crosscoder are all interesting and untested.

### Execution roadmap

Phase 5 splits into seven sub-phases with explicit dependencies. Priority tags:

- **P0** — blocking prerequisite. Nothing else runs until this is done.
- **P1** — required for the headline claim. Must appear in the paper.
- **P2** — recommended. Significantly strengthens the paper but the paper is still valid without it.
- **P3** — bonus. Only if compute and time remain after P0/P1/P2.

Do not skip ahead. Each sub-phase's exit criteria guarantee the next sub-phase's results will be interpretable.

#### Sub-phase 5.0 — Foundations [P0]

- [ ] **Data-leakage audit on Aniket's setup.** Read `docs/aniket/experiments/sparse_probing/saebench_notes.md` on branch `aniket`. Confirm (i) what corpus the SAE/MLC/TXCDR encoders were trained on, (ii) what text appears in SAEBench probe train/test sets, (iii) whether the two overlap. If overlap is non-negligible, Aniket's numbers are contaminated; retrain encoders on a disjoint corpus before anything else.
- [ ] **Close architecture-layer infrastructure gaps** (see "Reference: Infrastructure audit" below). Port `l1_coeff` / `warmup_steps` / `weight_decay` into `CrosscoderSpec.train` and `StackedSAESpec.train`; add L1 penalty to `TFASpec.train` when `k=None`; add `match_stacked_budget: bool = True` flag to `CrosscoderSpec`; consolidate or delete `src/architectures/relu_sae.py`. Roughly 2--3 hours.
- [ ] **Pre-register the plan** as `docs/han/research_logs/phase5_downstream_utility/plan.md`. Must contain: full architecture list, convergence protocol, per-architecture hyperparameter search budget, attention-pooled baseline protocol, success/failure criteria. Do not train any checkpoint before this is reviewed.

*Entry:* all three reading-list items in the "Reading list" section read in full. *Exit:* leakage audit passed; infrastructure gaps closed; `plan.md` committed.

#### Sub-phase 5.1 — Convergence-corrected replication [P1]

The minimum required claim of Phase 5: "we ran Aniket's comparison properly." If this contradicts his result, that alone is publishable. Covers the first three of Aniket's four gaps (convergence, missing baselines, seed variance).

- [ ] Retrain SAE, MLC, TXCDR (at T ∈ {5, 10, 20}) to **matched convergence** (loss drop < 2% per 1k steps). Report the step each model hit that threshold.
- [ ] **Add the four missing baselines**: Stacked SAE (T=5, 10, 20), Shared-weight per-position SAE, TFA, TFA-pos. Without them, MLC's win cannot be cleanly attributed.
- [ ] Evaluate on SAEBench's 8 tasks under **both** Protocols A and B. The A/B gap is itself a finding — see "Reference: Sparsity-budget semantics" below.
- [ ] Include an **attention-pooled baseline** (`papers/are_saes_useful.md` §5, Eq. 2). Required by the ceiling set in that paper; without it, any SAE-probe win is at risk of being the same illusion.
- [ ] **Three seeds minimum**. Report mean ± std on the headline cell.

*Entry:* 5.0 complete. *Exit:* headline table with ~10 architectures × 2 protocols × 8 tasks × 3 seeds.

#### Sub-phase 5.2 — Weight-sharing ablation ladder [P1]

Pins down *where on the capacity spectrum* the best architecture sits. Cheap: half a day of implementation; trivial compute relative to 5.1. See "Reference: Weight-sharing ablation ladder" below for the full list.

- [ ] Implement and train the six weight-sharing variants (sharedDec, sharedEnc, per-position-biases-only, TXCDR-pos, tied-weights, factorized decoder) under the 5.1 protocol.
- [ ] Add their rows to the headline table.

*Entry:* 5.0 infrastructure fixes landed. Can run in parallel with 5.1. *Exit:* six additional rows on the headline table.

#### Sub-phase 5.3 — One novel architectural contribution [P1]

The paper's positive contribution regardless of outcome. Pick **exactly one** of the four primary extensions in "Reference: Architecture menu" below. Recommendation in order: Matryoshka-TXCDR (position-nested) > rotational decoder > causal TXCDR > Time×Layer joint crosscoder.

- [ ] Implement the chosen architecture under the existing `ArchSpec` interface.
- [ ] **Ablate on Phase 3 coupled-features toy data first.** Cheap sanity check — if the variant fails there, it will fail on NLP too, and iterating on toy is 100× cheaper than on Gemma.
- [ ] Run the SAEBench eval under the 5.1 protocol.

*Entry:* 5.0 infrastructure fixes landed. Can run in parallel with 5.1 / 5.2. *Exit:* novel architecture has a row on the headline table and a toy-data ablation plot.

#### Sub-phase 5.4 — Cross-token probing task [P1]

SAEBench's 8 tasks are mostly last-token classification; temporal SAEs cannot shine where the answer fits in one token. The "temporal SAEs are useful" claim requires at least one genuinely cross-token task.

Candidates (in order of expected implementation effort):

1. **Coreference resolution** (which entity does a pronoun refer to?) — requires a multi-token context by construction. Cheapest starting point.
2. **BLiMP long-range grammatical dependencies** (subject-verb agreement across clauses, NPI licensing, etc.).
3. **Reasoning-trace attributes** on DeepSeek-R1-Distill-Llama-8B — infrastructure exists per `project_brief.md` but caching was interrupted and must be resumed. Highest-signal target; most expensive to set up.

- [ ] Pick one. Run the full architecture set from 5.1 + 5.2 + 5.3 on it.

*Entry:* 5.1 complete. *Exit:* at least one cross-token task added to the headline table.

#### Sub-phase 5.5 — Writeup [P1]

- [ ] Write `docs/han/research_logs/phase5_downstream_utility/summary.md` covering 5.1–5.4. Be honest under whichever of the three outcomes (see "Three possible outcomes" below) actually holds.

*Entry:* 5.1–5.4 complete. *Exit:* `summary.md` committed.

#### Sub-phase 5.6 — Bonus extensions [P3]

Only after 5.1–5.5 are complete.

- Remaining primary architectural extensions not picked for 5.3 — e.g. if Matryoshka was picked, also run the rotational decoder as a secondary contribution.
- **Layer robustness**: repeat the headline sweep at `resid_L13` on Gemma.
- **More cross-token tasks**: if 5.4 picked coreference, add BLiMP; if 5.4 picked BLiMP, add reasoning-trace attributes.
- **Scale to DeepSeek-R1-Distill-Llama-8B** for reasoning-trace probing — the most compelling test of the "temporal features matter for reasoning" thesis.
- **Reconcile with `han-runpod`'s feature-distinctness finding** (flagged in `project_brief.md` open questions). Port their metric, re-rank our TFA-pred features, update Phase 4 honestly if needed.

---

## Reference material

The remaining sections are a library the sub-phases above draw from. Skim on first read; consult the specific section when the corresponding sub-phase runs.

### Reference: Architecture menu (primary extensions for 5.3)

Vanilla TXCDR has structural limitations worth addressing. Each is a candidate Phase 5 contribution; sub-phase 5.3 picks one:

**1. Matryoshka-TXCDR (recommended primary).** Currently TXCDR applies a single TopK to a shared pre-activation summed over T positions. There is no mechanism to force *any* latent to carry per-position (local) signal. A Matryoshka variant forces prefixes of the latent vector to reconstruct sub-windows: first $m_1$ latents must reconstruct each position alone; the next $m_2$ latents extend to windows of size 2; and so on. This gives TXCDR an explicit place to put local-token information --- directly addressing Aniket's observation that "TempXC loses on language-ID and code detection where local-token information dominates."

Two nesting axes to consider:
- *Over position*: nested sub-windows of increasing T.
- *Over feature index*: nested prefixes of the dictionary reconstruct at successive quality levels (standard Matryoshka-SAE).

The position-nested version is the more novel contribution.

**2. Causal TXCDR.** Vanilla TXCDR's encoder is acausal: position $t$'s encoding sees positions $t+1, \ldots, t+T-1$ via the shared pre-activation. For probing last-token information this is a form of leakage (features at $t$ are informed by the future). A causal variant restricts position $t$'s latent to depend only on positions $\leq t$. TFA does this via attention mask; TXCDR would need an analogous restriction in its encoder tensor.

**3. Time×Layer joint crosscoder.** MLC shares latents across layers; TXCDR shares across positions. A joint model takes input `(T, L, d_model)` and shares a single latent across both axes. Two sparsity variants: (a) global TopK over the full `(T, L, d_sae)` pre-activation; (b) product-structured TopK (k_layer per position + k_time per layer). Nobody has tested this. If layer-axis MLC is what gives Aniket his probing wins, then (Time×Layer) should at least match MLC. If it *beats* MLC, that's a clean positive result for "temporal structure matters when combined with layer-axis redundancy."

**4. Decoder smoothness / parameterized decoder.** Vanilla TXCDR's per-position decoders are entirely unconstrained — latent $j$'s decoder column can point in a totally different direction at $t=0$ vs $t=T-1$. That's expressive but likely over-parameterized. Two knobs:

- *Soft penalty*: add `Σⱼ Σₜ (1 − cos(W_dec[j,t,:], W_dec[j,t+1,:]))` (or an all-pairs variance variant) to the loss. Same penalty for `W_enc` rows. One coefficient each.
- *Hard parameterization* `W_dec⁽ᵗ⁾ = f(W_base, t)`. Forms worth trying, in order of expressiveness:
    - **Low-rank residual**: `W_dec⁽ᵗ⁾ = W_base + Uₜ Vₜᵀ`, rank `r ≪ min(d_in, d_sae)`.
    - **Rotational / Lie-group**: `W_dec⁽ᵗ⁾ = exp(t · A) W_base` with `A` skew-symmetric (Cayley transform avoids the matrix exp). Directly operationalizes "feature direction rotates across time" — angular velocity is readable from the spectrum of `A`.
    - **Basis expansion in time**: `W_dec⁽ᵗ⁾ = Σₖ αₖ(t) · W_baseₖ` with `K ≪ T` shared basis matrices and learned (or fixed sinusoidal) time coefficients.
    - **FiLM per-position modulation**: `W_dec⁽ᵗ⁾ = diag(gₜ) W_base diag(hₜ)`. Cheapest; captures scale changes only.

The rotational parameterization is the most novel candidate — clean geometric hypothesis, interpretable angular-velocity parameter, publishable on its own if it works.

**Failure mode to watch.** Over-regularization collapses to Shared SAE (`W_dec⁽ᵗ⁾ ≈ W_dec⁽⁰⁾` for all `t`). Shared SAE already loses to MLC in Aniket's contest, so the sweet spot is "slowly varying but non-trivial."

### Reference: Weight-sharing ablation ladder (for sub-phase 5.2)

Beyond the four primary extensions, there's a simple 2×2 of weight-sharing patterns that costs ~half a day of implementation and gives the cleanest possible ablation of where vanilla TXCDR's capacity is actually needed. All are trivial tweaks to `TemporalCrosscoder` (just change the shape of `W_enc` / `W_dec` and broadcast in `encode` / `decode`):

| W_enc | W_dec | variant | params vs. vanilla TXCDR |
|---|---|---|---|
| shared | shared | Shared SAE (existing baseline) | 1/T |
| per-pos | shared | **TXCDR-sharedDec** | (T+1)/(2T) |
| shared | per-pos | **TXCDR-sharedEnc** | (T+1)/(2T) |
| per-pos | per-pos | Vanilla TXCDR | 1 |

Hypothesis sharedDec tests: "features have *fixed* directions but need different readers at different positions." Hypothesis sharedEnc tests: "features are recognized the same way but their contribution to reconstruction varies by position." Running both pins down which half of vanilla TXCDR is doing the work.

Additional cheap variants, in rough order of parameter budget:

1. **Per-position biases only.** Shared `W_enc` and `W_dec` with per-position `b_enc[t]`, `b_dec[t]`. Tiny modification of Shared SAE. If this recovers most of vanilla TXCDR's advantage, "per-position matrices" is wasted capacity.
2. **TXCDR-pos** (analog of TFA-pos). Single shared `W_enc`/`W_dec`, add a sinusoidal positional embedding to `x_t` before encoding. Same-param-count extension of Shared SAE. Tests whether positional information *alone* is sufficient.
3. **Tied-weights TXCDR.** Enforce `W_enc[t] = W_dec[:, t, :].T` at every `t`. Halves the per-position parameter count. Classic SAE trick adapted.
4. **Factorized decoder.** `W_dec[j, t, :] = α[j, t] · d[j]` — one shared direction per latent plus a scalar per `(latent, position)` gain. `O(d_sae·(d_in + T))` instead of `O(d_sae·T·d_in)`. If features only *scale* across positions (not rotate), this should suffice.

Run these alongside the primary extensions. The ablation ladder from Shared SAE to vanilla TXCDR is more informative than a single "TXCDR vs MLC" number — it tells the reader *where* on the capacity spectrum the winning architecture sits.

### Reference: Sparsity-budget semantics (read before interpreting any `k`)

This is the single most common source of accidental apples-to-oranges comparisons in this project. When anyone says "at k=5", you must immediately ask: **k per what?** A `k` that means "5 active features per token" and a `k` that means "5 active features per 5-token window" differ by a factor of T and can flip the ranking between architectures.

**Four quantities, all distinct**:

- `k_tok` — active features used to represent one token.
- `k_pos` — active features at one position inside a window. For single-token SAE, `k_pos = k_tok`.
- `k_win` — total active features across one window of T tokens.
- **Amortised per-token rate** `k_win / T` — the effective per-token budget for windowed models whose single latent covers the whole window.

**Per-architecture semantics** (unambiguous form):

| Architecture | native TopK knob | `k_pos` | `k_win` (T-token window) |
|---|---|---|---|
| Vanilla SAE | TopK on each token's latent | k | T·k |
| Stacked SAE (T positions) | TopK per per-position SAE | k | T·k |
| MLC (L layers, per token) | TopK on shared across-layer latent | k | T·k |
| **Vanilla TXCDR** | **TopK on shared window latent** | **k / T (amortised)** | **k** |

Stacked SAE, MLC, and vanilla SAE all use the *same* `k` convention: it counts features per token. TXCDR is the odd one out — its native TopK controls the whole window at once.

**Critical implementation detail.** `CrosscoderSpec.create(d_in, d_sae, k, device)` **silently multiplies `k` by T** before constructing `TemporalCrosscoder`:

```python
# src/architectures/crosscoder.py, CrosscoderSpec.create
k_eff = k * self.T if k is not None else None
return TemporalCrosscoder(d_in, d_sae, self.T, k_eff)
```

So when the sweep runner passes `k=5` to a `CrosscoderSpec(T=5)`, the resulting TXCDR has a TopK of **25** on its shared latent — not 5. This is the Phase-2 "TXCDRv2" convention (`k_win = k · T`, matching a Stacked SAE's total-window budget of T·k). A naive reading that "k=5 means 5 active features" is wrong for TXCDR under this spec. Under this convention, all architectures in a sweep row with the same `k` argument have the *same* `k_win` — which is what we want for a fair reconstruction comparison, but it means **TXCDR's raw `k` and the probing-study's `k` are not the same number**.

The audit earlier in this brief recommends adding a `match_stacked_budget: bool = True` flag to expose the "unfair" alternative (`k_win = k`). Until that lands, **every TXCDR number produced by `CrosscoderSpec` is at `k_win = k · T`**, and this must be stated explicitly in every figure, table, and JSONL record.

**Aniket's two sparsity protocols, translated:**

| Protocol | `k_pos` | TXCDR `k_win` at T=5 / 10 / 20 | Stacked / MLC `k_win` at T=5 / 10 / 20 |
|---|---|---|---|
| **A — per-token-rate matched** | 100 (fixed) for all archs | 500 / 1000 / 2000 (grows with T) | 500 / 1000 / 2000 |
| **B — window-budget matched** | 100 for Stacked/MLC; 100/50/25 for TXCDR | 500 / 500 / 500 (fixed) | 500 / 1000 / 2000 |

Under Protocol A, TXCDR's total budget *scales* with T — which means at T=20 it has 4× more active features than at T=5. Under Protocol B, TXCDR's window budget is clamped at 500 regardless of T, so per-position budget *shrinks* as T grows. Neither protocol is canonically correct; they test different hypotheses. **The gap between A and B is itself a finding**: if B's numbers stay close to A's as T grows, degradation is architectural (shared-latent dilution). If B sharply outperforms A at high T, degradation is a capacity artifact (too many features, not enough signal).

**Common mistakes to avoid:**

- Writing "TXCDR at k=5" in a caption without also stating `k_win` (is it 5 or 25?).
- Comparing a TXCDR at `k_win=25` against a vanilla SAE also labelled `k=5` — the SAE has `k_win=5` per token. The numbers look comparable and aren't.
- Forgetting that `CrosscoderSpec.create`'s k·T scaling is silent. Always log `k_pos` and `k_win` at the start of every training run.
- Assuming MLC's `k` means the same as TXCDR's. MLC's shared latent collapses the *layer* axis (at fixed token position); TXCDR's shared latent collapses the *time* axis (at fixed layer). Same math, different physical meaning.
- Reporting reconstruction NMSE "at matched k" without saying whether the match is per-position, per-window, or amortised per-token.

**Rule** (binding for Phase 5): every figure caption, table header, and JSONL record must include all three of `(T, k_pos, k_win)` explicitly — not just one. When `k_pos` is ambiguous for TXCDR (the amortised rate isn't an integer), report it as `k_win / T`.

### Reference: Training-fairness rubric (binding for 5.1–5.4)

The single largest methodological trap is hyperoptimizing TXCDR while leaving baselines under-tuned --- or the opposite. Bind yourself in advance to the following rules:

- **Every architecture gets the same hyperparameter search budget.** Not "we swept LR for TXCDR", "we swept LR for MLC and SAE too." If one arch gets a grid search, all get the *same* grid search.
- **Every architecture trains to convergence** (loss drop < 2% per 1000 steps), with a max step cap to avoid runaway compute. Report the step each model converged at --- this is itself a finding (TXCDR converges slowly, per Aniket).
- **Normalise sparsity at the window level**, not per-token. Run both Aniket's Protocol A (per-token-matched) and Protocol B (window-matched). Neither is obviously correct.
- **Report FLOPs and param counts**, not just accuracy. MLC beats TXCDR but at 5× the params of vanilla SAE. FLOPs matter.
- **Ablate each architectural knob** on a small synthetic problem before spending NLP compute. Example: causal-vs-acausal TXCDR should be tested on Phase 3 coupled data first.
- **Three seeds minimum.** Report mean ± std on the headline cell.

### Reference: Infrastructure audit (gaps to close in 5.0)

Audit of `src/architectures/` against every arch named in `docs/han/research_logs/phase2_toy_experiments/2026-03-30-experiment1-topk-sweep.md`:

| arch from doc | implementation | status |
|---|---|---|
| Shared SAE (TopK) | `TopKSAESpec` | ok |
| Wide Shared SAE | `TopKSAESpec(d_sae=100)` | ok (config, not class) |
| TFA | `TFASpec(use_pos_encoding=False)` | ok |
| TFA-pos | `TFASpec(use_pos_encoding=True)` | ok |
| TFA-shuffled / TFA-pos-shuffled | `TFASpec` + `gen_seq_shuffled` | ok (data-pipeline, correct design) |
| Stacked SAE T=2, T=5 | `StackedSAESpec(T)` | ok |
| TXCDRv2 T=2, T=5 (fair, k·T per window) | `CrosscoderSpec(T)` (k·T applied in `.create`) | ok |
| TXCDR T=2, T=5 (unfair, k per window) | not exposed | **gap** — `CrosscoderSpec.create` always scales k by T |

**Gaps to fix before Phase 5 can run the full architecture set cleanly**:

1. **`CrosscoderSpec.train` and `StackedSAESpec.train` do not accept `l1_coeff`, `warmup_steps`, `weight_decay`**. `TopKSAESpec.train` does (see commit `ab769d7`). Port the same kwargs into both. Without this, TXCDR/Stacked cannot be trained in ReLU+L1 mode, which blocks Experiment 2 reproduction and the ReLU-variant sweeps.
2. **`TFASpec.train` does not add an L1 penalty on `novel_codes`.** `TemporalSAE` itself supports `sae_diff_type="relu"` but the spec's training loop only uses MSE. Add L1 on novel_codes when `l1_coeff > 0`.
3. **`src/architectures/relu_sae.py` is dead code.** Task #32 in the backlog. Consolidate into `TopKSAE` or delete --- do not import it from new Phase 5 code.
4. **Expose "unfair" TXCDR via `CrosscoderSpec`** by adding a `match_stacked_budget: bool = True` flag that controls the k·T scaling in `.create`. Minor; only needed if Phase 5 wants to reproduce the original Phase 2 TXCDR-vs-TXCDRv2 comparison.

These are mechanical fixes, roughly two to three hours of work. Land them before any Phase 5 experiment starts.

### Reference: Probing task inventory (for 5.1 and 5.4)

1. **SAEBench sparse probing suite** — the same 8 tasks Aniket used (ag_news, amazon_reviews, amazon_reviews_sentiment, bias_in_bios × 3, europarl, github-code). Sub-phase 5.1 runs all architectures on this.
2. **Cross-token probing tasks** (sub-phase 5.4 picks one):
    - **Coreference resolution** (which entity does a pronoun refer to?). Requires a multi-token window by construction. Cheapest to set up.
    - **Long-range grammatical dependencies** from BLiMP or similar (subject-verb agreement across clauses, NPI licensing).
    - **Reasoning-trace attributes** on DeepSeek-R1-Distill-Llama-8B. Cache infrastructure exists per `project_brief.md`; caching was interrupted and needs to be resumed. Highest-signal target; most expensive to set up.
3. **Baselines every task must include** (not just SAE-probe vs. SAE-probe comparisons):
    - Last-token logistic regression on raw activations.
    - **Attention-pooled probe** per Eq. 2 in `papers/are_saes_useful.md`: trainable $q, v \in \mathbb{R}^d$, `[softmax(X · q)] · [X · v]`, followed by logistic regression. Required — without it, any SAE-probe win is at risk of the illusion the paper diagnoses in §5.

### Three possible outcomes and what each means

- **TXCDR or a variant wins head-to-head against MLC on matched convergence + matched baselines.** Clean positive result for temporal SAEs; paper writes itself.
- **MLC > TXCDR variants, but some TXCDR variant beats attention-pooled baseline on at least one cross-token task.** A nuanced positive: "layer-axis is better in general, but temporal-axis carries genuinely distinct information useful in specific regimes." Still publishable.
- **MLC > everything temporal, and no TXCDR variant beats attention-pooled baselines on any task.** A negative result in the spirit of `are_saes_useful.md`. The paper becomes "temporal crosscoding over token positions is not a free win; layer-axis crosscoding is. Here is why, and here are three architectural attempts that didn't rescue it." This is still publishable and genuinely useful to the field.

Do not optimize for outcome (1). Optimize for a result that is honest under whichever of the three holds.

### Reading list for the Phase 5 agent

1. `papers/are_saes_useful.md` --- sets the evaluation bar.
2. `docs/han/research_plan/project_brief.md` --- overall project.
3. `docs/han/research_logs/phase2_toy_experiments/2026-03-30-synthesis.md` --- what Phase 2 actually showed (content-based matching dominates temporal signal in TFA; TXCDR feature-recovery wins only hold for `k·T << d_sae`).
4. `docs/han/research_logs/phase3_coupled_features/2026-04-07-experiment1c3-coupled-features.md` --- TXCDR's gAUC advantage on hidden-state recovery is a *feature-recovery* claim, not a probing claim.
5. `docs/han/research_logs/phase4_nlp_comparison/2026-04-17-nlp-comparison-index.md` --- NLP-scale comparison and the TXCDR dead-feature problem.
6. Branch `aniket`: `docs/aniket/experiments/sparse_probing/plan.md` and `summary.md`.
7. Branch `aniket`: `docs/aniket/experiments/sparse_probing/saebench_notes.md` --- SAEBench integration notes; contains the answer to the data-leakage question.
