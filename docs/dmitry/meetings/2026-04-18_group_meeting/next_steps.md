---
author: Dmitry Manning-Coe
date: 2026-04-18
tags:
  - proposal
---

## Context

The 2026-04-18 group meeting reviewed three findings:

1. On SAEBench sparse probing (8 tasks, Gemma-2-2B L12), **MLC (cross-layer crosscoder) wins decisively** (+8.6 pp vs SAE), **TempXC matches SAE at T=5 and degrades monotonically with T** (−8.1 pp at T=20). See `origin/aniket:docs/aniket/experiments/sparse_probing/summary.md`.
2. Independent signal that TXCDR *is* architecturally sensitive to temporal structure: Han's feature-distinctness result (`origin/han-runpod:docs/han/research_logs/2026-04-18-txcdr-vs-tfa.md`) and Aniket's shuffle-control decoder-geometry study (`origin/aniket:docs/aniket/experiments/sprint_feature_geometry/summary.md`, stronger effect on GSM8K than FineWeb).
3. Dmitry's hidden-state probing on synthetic Markov data (`docs/dmitry/probing_hidden_state.md`): TXCDR probe AUC rises 0.68 → 0.99 as ρ → 1 (the latent contains the hidden state) while R@90 → 0.0 (decoder directions don't align with true features). "Model knows the answer; wrong inductive bias."

The three questions from the meeting were: (1) what are the right benchmarks for TXC? (2) is TXC using temporal information, and what would force it to? (3) do the prior next-steps make sense?

The meeting transcript at `docs/dmitry/meetings/2026-04-18_group_meeting/transcript.md` is currently empty — plan is built from the three questions you relayed plus the research logs above.

## Q1 — Right benchmarks for TXC

**Core problem with sparse probing as the primary eval:** a linear probe at position *t* reads whatever attention already wrote to the residual stream at *t*. If the downstream task only needs information that flowed to *t*, a per-token MLC will match or beat TXC by construction — Elhage framework, *"A Mathematical Framework for Transformer Circuits"*. Aniket's summary diagnoses this correctly ("layer axis = redundant views of same token, position axis = different tokens"). The steelman lands on natural language; it should *not* land on tasks where the ground-truth latent is a function of temporal history.

**Proposed primary benchmark: HMM belief-state recovery.**

- **Datasets**: Mess3 (3-state HMM with fractal belief simplex, Shai/Riechers arXiv:2405.15943) and RRXOR (output depends on XOR of positions two back — non-Markov; any solver *must* mix positions).
- **Why it's the right test**: the ground-truth belief state is *by construction* a function of temporal history. A per-token SAE *cannot* recover it without mixing across positions. Success criterion is geometric — does the SAE/crosscoder reconstruct the Sierpinski simplex?
- **Prior art to reuse**: arXiv:2604.02685 "Finding Belief Geometries with Sparse Autoencoders" (validates SAE-on-HMM on a small transformer then ports to Gemma-2-9B). They claim code is released with the paper — this is the fastest path to a working harness.
- **Repo**: `docs/shared/references.md` and `docs/dmitry/theory/sae_hmm_temporal_generalizations.md` already reference the Simplex line; no canonical single repo, reproduce from paper.

**Proposed secondary benchmarks:**

- **Sparse Feature Circuits (`saprmarks/feature-circuits`, Marks arXiv:2403.19647)** — causal attribution across positions on tasks like subject-verb agreement where the verb requires info from an earlier subject. Measure IE concentration per feature; TXC features should concentrate more cleanly across the temporal window than MLC features on long-dependency tasks.
- **Recursive DFA (Kissane arXiv:2406.17759)** — per-feature decomposition of activation back through source positions. The cleanest single metric for "is this feature using temporal info." No known crosscoder paper reports it — methodological contribution if we do.
- **Tracr / InterpBench** — compile known temporal circuits (Dyck, counters, sort-by-first-occurrence) and test recovery. Lower priority; mainly useful as unit-test-scale sanity checks.

**Do NOT pivot to**: SAGE, CE-Bench, or more SAEBench tasks — all per-token or per-example, same fundamental limitation.

**Already-in-repo controlled experiment to keep running:** the ρ-sweep on synthetic Markov data (`src/temporal_bench/data/markov.py:32-72`, `scripts/probe_hidden_state.py`). This is a rho ∈ [0, 1] knob on temporal correlation and is the right-sized toy for architectural iteration. Extend from [0.0, 0.5, 1.0] to the full six-point sweep Dmitry documented.

## Q2 — Is TXC using temporal information?

**Architecture audit** (`src/temporal_bench/models/temporal_crosscoder.py:18-108`):

- Per-position encoder `W_enc: (T, d_in, d_sae)` with **sum aggregation** across T before TopK (line 63). No learnable temporal kernel.
- Per-position decoder `W_dec: (T, d_sae, d_in)`.
- Only loss term is reconstruction MSE (line 75). **No penalty for temporal non-use.**
- **Collapse mode**: model can learn all features at position 0's encoder and zero out `W_enc[t]` for t ≠ 0. Output is then a shared-latent single-token SAE. We don't currently know if this is happening.

**Diagnostics to run — in order of cost/value:**

1. **Temporal-shuffle NMSE gap** *(~half-day, uses existing code)*. The shuffle primitive already exists (`src/temporal_bench/benchmarks/aliased_data.py:217-231`) but is not systematically reported across architectures. Report ΔNMSE(shuffled − ordered) for SAE / MLC / TempXC at multiple T and ρ. Prediction per theory: TempXC should show the largest gap if it's using temporal info; if ΔNMSE ≈ 0 for TempXC, it's collapsed to a per-token SAE.
2. **Off-diagonal encoder norm** — per-position contribution magnitude `||W_enc[t]||_F / Σ_t ||W_enc[t]||_F`. If uniform, all positions matter; if one position dominates, collapse has happened.
3. **Tied-encoder ablation** — retrain TXC with `W_enc[t]` tied across t. If performance is unchanged, the position-specific encoder is decorative and the only thing doing anything is the sum aggregation + per-position decoder.
4. **Recursive DFA (new)** — port Kissane's method. Compute per-feature attribution across the T window. Publishable metric regardless of the direction of the result.
5. **Hidden-state probe pre-TopK vs post-TopK** — already scaffolded (`scripts/probe_hidden_state.py:74-78`). Run at all six ρ values. This is how we know whether the failure is encoding (pre-TopK probe fails) vs sparsification (pre-TopK succeeds but post-TopK fails — current evidence) vs decoding (both probes succeed but R@90 fails).

**Architecture / loss modifications to test:**

- **L1 sparsity penalty on post-mix activations** in `PerFeatureTemporalAE` (already in `project_next_steps.md`, still valid — L0 balloons 5→19 because no sparsity cost on kernel output).
- **Off-diagonal encoder penalty**: `λ · Σ_t,t' [t≠t'] ||W_enc[t] · W_enc[t']^T||_F`. Forces position-specific encoders to carry non-redundant information.
- **Cross-feature temporal kernel**: `PerFeatureTemporalAE` (`src/temporal_bench/models/per_feature_temporal.py`) currently couples each feature only to itself across time. Add optional `K: (m, m, T, T)` for cross-feature temporal mixing. More capacity; test if capacity was the bottleneck.
- **Baum-Welch factorial SAE**: already coded at `src/temporal_bench/models/baum_welch_factorial.py:28-180`. Not yet benchmarked against TXC at scale on the ρ-sweep. Do this.
- **Tied encoder + learnable temporal kernel post-sum**: decouples "what features exist" (tied encoder) from "how they mix across time" (kernel). The current TXC confounds both.

**Literature on "is the model using temporal info" as a general technique:** there is no known crosscoder paper that reports a temporal-shuffle control. Gap is real, not an oversight. The closest is *"Temporal Sparse Autoencoders"* (arXiv:2511.05541) which adds a contrastive loss across adjacent tokens but does not run shuffle ablation.

**Steelman response to Aniket's "layer-axis coherent, position-axis not":** he's right for natural language. The experimental question is whether this generalizes to data where position-axis coherence exists by construction (HMM belief state, RRXOR). This is exactly the Q1 benchmark question in a different framing. The two questions are the same question — if TXC wins on Mess3 but loses on SAEBench, Aniket's analysis is complete and the architecture is "right for the task it was designed for, not wrong."

## Q3 — Are the prior next-steps still the right ones?

**Prior next-steps in the project_next_steps memory (2026-03-26 vintage):**

1. **L1 penalty on post-mix activations** — still valid, low cost, should ship.
2. **Linear probe for hidden-state recovery** — *already built* (`scripts/probe_hidden_state.py`). The probing signal is what told us "wrong basis, not wrong model." Extend to all six ρ values and to the new architectures; don't rebuild.
3. **Baum-Welch SAE** — *already built* (`src/temporal_bench/models/baum_welch_factorial.py`). Has not yet been benchmarked against TXC on the ρ-sweep. Do that. This is the natural "structured temporal inference" comparison point.

**Prior plans are still right but no longer sufficient.** What was missing from the old plan:

- No benchmark pivot. The old plan assumed the synthetic ρ-sweep is the core eval. It is fine for architectural iteration, but for "which architecture to ship" we need a benchmark closer to real transformers — hence Mess3/RRXOR.
- No shuffle-sensitivity diagnostic. This is ~half-day of work and is the highest-value single diagnostic we don't yet have.
- No "does the encoder collapse" check. Zero-cost to compute, haven't computed it.

**Other things worth thinking through:**

- **Han's L13 fragility** (`origin/han-runpod:docs/han/research_logs/2026-04-18-txcdr-vs-tfa.md`): TFA-pos NMSE 0.125 at L25 vs 0.957 at L13 with the same hyperparameters. Matched-NMSE underfit isn't yet run. Any comparison at L13 currently conflates architecture with training instability.
- **Stacked-SAE baseline** (Aniket proposed, overnight run): T independent per-position SAEs. If Stacked-SAE ≥ TempXC on sparse probing, the shared latent is doing nothing useful; if Stacked-SAE < TempXC, the shared latent is at least matching per-token capacity.
- **Cross-model replication**: all current results are Gemma-2-2B + FineWeb. DeepSeek-R1-Distill-Llama-8B replication needed before any claim generalizes.
- **Han's single-seed-42 risk**: architectural pathology claims need seed=0 replication.
- **T-sweep on the HMM benchmark**: if the "TempXC degrades monotonically with T" result from SAEBench is natural-language-specific, it should invert on HMM data. Direct prediction to pre-register.

## Q4 — Routing TXC features by global-vs-local context reliance

Added 2026-04-18 post-meeting in response to the question: *can we combine TXC with data attribution techniques that identify when global info matters more than local?*

The intuition is tight. The SAEBench result shows TempXC features don't beat MLC *on average* over natural-language tasks. But averaged metrics hide the question "for which predictions does crossing T positions actually help?" If we had a per-token scalar `g(t) ∈ [0, 1]` measuring how much a prediction at position *t* depends on context outside the TXC window, we could:

- Evaluate TXC only on high-*g* tokens (where temporal mixing is *supposed* to help).
- Weight reconstruction loss by *g* during training.
- Route — gate TXC features by *g*, use MLC features for low-*g* tokens, shared-latent TXC for high-*g*. A hybrid architecture.

The user's baseline — attention decay length — is weaker than the current SOTA. Survey below organized from cheapest to most causal. All methods return something roughly interpretable as per-token "global reliance."

### Methods landscape (2024-2026 SOTA)

**Surprisal-delta (cheapest, ~1 hour).** For each token *t*, compute `H(t | last-k tokens)` for k ∈ {4, 16, 64, all}. Define `locality(t) = (H_4 − H_all) / (H_0 − H_all)`: close to 1 means the local prefix already resolves the token; close to 0 means global context is needed. Standard Hugging Face logits, no tooling required. Not causal, but cheap enough to compute alongside every training batch.

**Attention rollout + distance profile** ([Abnar & Zuidema, arXiv:2005.00928](https://arxiv.org/abs/2005.00928)). Recursively multiply per-layer attention to get token-to-token source attribution across layers. Fit an exponential or bimodal (local + global) mixture per output token; mixture weights give global-fraction. Correlational, but captures multi-layer information flow rather than single-head decay. ~50 lines of code.

**Attention entropy + attention sinks** ([Xiao et al. StreamingLLM, arXiv:2309.17453](https://arxiv.org/abs/2309.17453)). Attention entropy per head/position measures focus vs diffusion. Attention-sink detection flags heads that are not meaningfully using context. Both are much richer than decay length per se.

**Retrieval-head activation mass** ([Wu et al., arXiv:2404.15574](https://arxiv.org/abs/2404.15574); code at [nightdessert/Retrieval_Head](https://github.com/nightdessert/Retrieval_Head)). A small fraction of heads (<5%) are mechanistically responsible for long-range factual recall and are universal across models. For each generated token, record summed attention mass on positions at distance > T (the TXC window). This is the most direct "is this prediction reaching past the TXC window?" signal and is mechanistically grounded.

**AtP* / EAP-IG / RelP** ([Kramár et al. arXiv:2403.00745](https://arxiv.org/abs/2403.00745); [Hanna et al. arXiv:2403.17806](https://arxiv.org/abs/2403.17806); [koayon/atp_star](https://github.com/koayon/atp_star), [hannamw/eap-ig](https://github.com/hannamw/eap-ig)). Current SOTA for causal attribution. EAP-IG (Edge Attribution Patching + integrated gradients) is the most faithful at reasonable cost — ~2 forward + 1 backward pass per target token. RelP (2025) replaces AtP's local gradient with LRP coefficients and reportedly closes the gap to full activation patching. These give per-edge causal attribution, from which per-position reliance is derivable.

**ContextCite** ([Cohen-Wang et al., arXiv:2409.00729](https://arxiv.org/abs/2409.00729); [MadryLab/context-cite](https://github.com/MadryLab/context-cite)). **The most directly relevant tool.** Fits a sparse LASSO surrogate over context-ablation masks to attribute each generated token to context sources. ~32 forward passes per target token. For TXC: ablate (a) the T tokens inside the TXC window and (b) earlier tokens; coefficient-mass ratio outside vs inside the window is per-token global reliance — *causal*, not correlational. Expensive but can be applied to a few hundred tokens for calibration of the cheaper methods above.

**Sparse Feature Circuits** ([Marks et al. arXiv:2403.19647](https://arxiv.org/abs/2403.19647); [saprmarks/feature-circuits](https://github.com/saprmarks/feature-circuits)). Already SAE-feature-level attribution patching. The 2025 successor is Anthropic's [attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) using cross-layer transcoders. This is the closest existing intersection with SAEs/crosscoders — but still does not produce a per-token global-vs-local scalar.

### Explicit gaps in the literature

- **No paper produces a single per-token scalar "global vs local reliance" for generic LM prediction.** Surprisal-delta, attention-rollout profiles, and retrieval-head mass all get close, but no one has published the metric as such.
- **No SAE/crosscoder work routes or weights features by context-dependency.** Sparse Feature Circuits attributes features to context after the fact; no one trains or gates SAE features *by* per-token context reliance.
- **Residual-stream mutual information between distant positions is thinly studied.** Voita's Information Bottleneck work ([lena-voita.github.io/posts/emnlp19_evolution.html](https://lena-voita.github.io/posts/emnlp19_evolution.html)) is the nearest ancestor.

**This means TXC + per-token attribution is a genuine research opening**, not a crowded space.

### Proposed experiments (ordered by cost)

1. **Shuffle-sensitivity, stratified by locality(t)** (half-day, additive on top of ordering-item 1 below). Compute surprisal-delta `locality(t)` on the eval set. Split tokens into low-*g* / medium-*g* / high-*g* bins. Report TempXC vs MLC reconstruction loss per bin. Prediction: TempXC advantage (if any) concentrates in high-*g* tokens. If not, the architecture isn't leveraging the global-info tokens either — a strong negative result.
2. **Attention-rollout distance profile × TXC feature activation** (one day). For each TXC feature, compute average attention-rollout source distribution across tokens where it fires. Features that fire on tokens with broader rollout distributions are using more global context. Paint the feature dictionary by rollout-distance; compare to MLC features. Zero new compute beyond existing forward passes.
3. **Retrieval-head mass as a TXC gate** (1–2 days). Compute per-token retrieval-head attention mass outside the T-window. Use it as a sample weight in TXC reconstruction loss: `loss = Σ_t (1 + α · g(t)) · ||x_t − x̂_t||²`. Train a small α-sweep. If α > 0 improves probe AUC / feature recovery, TXC benefits from being told where global info matters.
4. **ContextCite calibration of (1–3)** (3 days). For a few hundred curated tokens, run ContextCite to get causal global-reliance ground truth. Correlate against surprisal-delta, rollout-distance, and retrieval-head mass. Publish the cheapest proxy that correlates > 0.8 with ContextCite — this alone is a methodological contribution.

Experiments 1 and 2 are drop-in extensions of the Q2 shuffle-sensitivity report. Experiment 3 is the first "architectural" use of attribution signal to improve TXC. Experiment 4 is the calibration / publishable-method half.

## Recommended ordering

Highest-leverage → lowest:

1. **Shuffle-sensitivity report** across SAE / MLC / TempXC at T ∈ {5, 10, 20}, **stratified by surprisal-delta locality** (Q4 experiment 1). *Uses existing `aliased_data.py` shuffle primitive + HF logits.* Half-day + half-day for the stratification.
2. **Encoder-collapse diagnostic** on existing TempXC checkpoints (off-diagonal norm, tied-encoder retrain). One day.
3. **Attention-rollout distance profile × TXC feature activation** (Q4 experiment 2). One day, zero new training.
4. **Mess3 / RRXOR benchmark setup** — port code from arXiv:2604.02685 or reproduce. ~3 days if code releases; ~5 if reproducing. Run existing TXC/MLC/SAE as interpreters. This is the pivot eval.
5. **Baum-Welch SAE on ρ-sweep**, extended to all six ρ values, with pre-TopK + post-TopK probe AUC. One day (code exists).
6. **Retrieval-head mass as TXC loss weighting** (Q4 experiment 3). Retrain with α-sweep. 1–2 days.
7. **L1 post-mix penalty + off-diagonal encoder penalty** retrains. One day each.
8. **Stacked-SAE baseline** + **seed=0 replication** — overnight each.
9. **ContextCite calibration of cheap locality proxies** (Q4 experiment 4). 3 days. Standalone methodological contribution.
10. **Recursive DFA port** (Kissane arXiv:2406.17759). Methodological contribution. ~2-3 days.

Items 1–3 should happen before 4 because they determine whether the Mess3 pivot is rescuing a working architecture or a collapsed one, *and* whether per-token locality stratification revives the SAEBench result.

## Critical files

- `src/temporal_bench/models/temporal_crosscoder.py:18-108` — TXC forward + loss
- `src/temporal_bench/models/baum_welch_factorial.py:28-180` — already-implemented alternative
- `src/temporal_bench/models/per_feature_temporal.py:27-142` — for L1 penalty addition
- `src/temporal_bench/benchmarks/aliased_data.py:217-231` — shuffle primitive
- `src/temporal_bench/data/markov.py:32-72` — rho-sweep generator
- `scripts/probe_hidden_state.py:74-78` — pre-TopK probe extraction
- `origin/aniket:docs/aniket/experiments/sparse_probing/summary.md` — result we're responding to
- `origin/han-runpod:docs/han/research_logs/2026-04-18-txcdr-vs-tfa.md` — feature-distinctness evidence

## Open questions for next meeting

1. Do we pivot *primary* eval to HMM/belief-state, or run it in parallel while SAEBench remains nominal primary for the paper?
2. Is the 3-day budget for Mess3 benchmark setup acceptable, or is there a smaller MVP?
3. If shuffle-sensitivity shows encoder collapse, do we kill TXC or fix it with tied-encoder + post-sum kernel?
4. Priority call between "publishable methodological contribution" (Recursive DFA, ContextCite calibration) and "architecture fix" (L1 + off-diagonal penalty + retrieval-head gating) — can't afford all in April.
5. Who owns each workstream? Current distribution inferred: Aniket = SAEBench/MLC, Han = autointerp/feature-geometry, Dmitry = theory/Baum-Welch, unspecified = HMM benchmark pivot + Q4 attribution track.
6. Is the Q4 attribution-routing story the paper's main narrative, or a secondary one? ("TXC features, gated by per-token global-reliance, beat MLC on high-global tokens" is a much cleaner story than "TXC ≈ MLC on average.")
