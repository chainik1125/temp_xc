---
author: Aniket Deshpande
date: 2026-04-29
tags:
  - proposal
  - in-progress
  - ward-backtracking
---

## TL;DR

Replicate Ward, Lin, Venhoff, Nanda 2025 ([arXiv 2507.12638](https://arxiv.org/abs/2507.12638)) as a **two-stage** experiment:

- **Stage A — DoM smoke test ✅ DONE (2026-04-29, single A40, ~3.5 h).** Reproduced Ward Fig 3: a Difference-of-Means steering vector derived from base Llama-3.1-8B activations induces backtracking in DeepSeek-R1-Distill-Llama-8B at +12 magnitude (3.4× baseline). cos(base_union, reasoning_union) = 0.7941 (Ward: ~0.74); validation F1 = 0.587 (Ward: ~0.60). See [[results]].
- **Stage B — base-only TXC steering (UNGATED, GO).** Train a base-only TXC on Llama-3.1-8B at multiple hookpoints, derive a steering direction from a *single TXC feature*, induce backtracking in the distill. **B1** (single-feature steering) → **B2** (cross-model temporal-encoding diff) sequenced; both run off the same trained TXC. Originally gated on Venhoff May 4; now ungated per Aniket's call (2026-04-29) — Stage A's clean replication is enough de-risking.

## Why now (Dmitry update, 2026-04-29)

Two pieces of channel feedback materially shape Stage B:

> *"Backtracking is probably a pretty good test case for TXCs because they have to steer on positions before it actually takes place (Fig 2 [Ward 2025])."*

The Ward offset window [−13, −8] preceding a backtracking sentence is *exactly* the offset-before-behavior structure TXC's shared latent is designed for. If TXC has any natural advantage over flat SAE / DoM, this is the test case where it should show up. **Headline framing for Stage B is now: "TXC's temporal axis matches the temporal structure of backtracking, where DoM and flat SAE collapse to a per-offset mean."**

> *"TXCs outperform everything if trained at ln1 hook normalized (i.e just before attention). This is where you expect the most temporal information to live."*

Dmitry's MatTXC table (TinyStories) shows hookpoint matters dramatically — TXC scores 0.02 at `ln1.0` but 0.94 at `resid_mid.0`. So Stage B should **train at multiple hookpoints**, not just Ward's residual-stream layer 10:

| Hookpoint | Why try |
|---|---|
| `resid_pre.10` | Ward-faithful — direct A→B comparison with Stage A DoM |
| `ln1.10` | Dmitry's TinyStories optimum; pre-attention; richest temporal info |
| `attn_out.10` | Where the temporal mixing was just computed — natural TXC target |

This adds ~3× to TXC training compute but lets us pick the strongest hookpoint per-experiment rather than committing to a guess up front.

> *"Diff Means steering outperforms SAE based steering in backtracking. This is a little surprising to me, and we should be able to do better by model diffing."*

The standing null we have to plan for. If our base-only TXC features ≤ DoM at all three hookpoints, **that is itself a clean publishable result**, framed as: *"the temporal axis alone (TXC over flat SAE) is not sufficient to recover Ward's backtracking direction in the feature basis; geometric DoM remains the strongest base-model-side primitive."* Pre-registered below.

## Stage A — DoM replication ✅ DONE

Reproduced Ward Fig 3 on a single A40 in ~3.5 h wall (~$6 API). Headline: base-derived DoM vector at residual stream layer 10 lifts the keyword rate of DeepSeek-R1-Distill from 0.7% (baseline) to 2.4% at mag=+12 and 4.6% at mag=+16. Validation F1 = 0.587 at strength=12 (Ward: ~0.60). Full results in [[results]]. Code at `experiments/ward_backtracking/`.

The Stage A pipeline (`seed_prompts → generate_traces → label_sentences → collect_offsets → derive_dom → steer_eval → plot → validate`) and its outputs (`results/ward_backtracking/{prompts,traces,sentence_labels,steering_results,validation}.json`, `dom_vectors.pt`, `plots/fig3_reproduction.png`) are the **fixed scaffolding Stage B builds on**. Stage B re-uses the same 280-dom / 20-eval prompt split, the same labelled-sentence taxonomy, and the same `steer_eval.py` harness — what Stage B replaces is the *source* of the steering vector (DoM vector → TXC feature decoder row).

## Stage B — base-only TXC at multiple hookpoints (UNGATED, GO)

### Headline framing

> *"TXC's temporal axis (encode window T positions → shared latent → decode at T positions) matches the temporal structure of backtracking, where DoM and flat SAE collapse to a per-offset mean."*

Ward et al. show that backtracking is encoded **before** the backtrack sentence (offsets [−13, −8] preceding it). A flat SAE / DoM has to commit to a single offset; a TXC's shared latent natively encodes information distributed across an offset window. **B1** (single-feature steering at one offset, apples-to-apples vs Ward DoM) → **B2** (cross-model temporal-firing diff, the TXC-native claim) sequentially.

### Hookpoints to train (per Dmitry's TinyStories MatTXC table, 2026-04-29)

Train one base-only TXC at *each* of these hookpoints in Llama-3.1-8B:

| Hookpoint | Why | Stage A anchor |
|---|---|---|
| `resid_pre.10` | Ward-faithful — direct comparison with Stage A DoM vector | DoM cosine 0.79 |
| `ln1.10` | Dmitry's TinyStories optimum; pre-attention; richest temporal info | re-derive DoM here as B1 baseline |
| `attn_out.10` | Where temporal mixing was just computed; natural TXC target | re-derive DoM here as B1 baseline |

We pick the strongest hookpoint at the end based on B1 steering strength against the per-hookpoint DoM baseline. Adds ~3× to TXC training compute but de-risks the "wrong-hookpoint null."

### B1 — single-feature steering

| Knob | Value |
|---|---|
| TXC training data | Llama-3.1-8B base activations *only*; DeepSeek-R1-Distill traces never seen during training |
| TXC arch | `TemporalCrosscoder` from `temporal_crosscoders/models.py` — shared latent, T-position encode/decode, TopK-(k×T) sparsity |
| Window | T = 6 covering offsets `[-13, -12, -11, -10, -9, -8]` (Ward's window, sentence-relative) |
| d_sae | 16k for first run; 32k if 16k looks under-trained at end of step budget |
| k (per-position) | 32 (window-level L0 = 32 × 6 = 192) |
| Tokens trained | ~50M (1–2 days on single A40 per hookpoint after activation cache) |
| Feature mining | run trained TXC encoder on Stage A `dom-split` traces; rank features by mean activation on backtracking sentences − mean activation on other sentences (per-offset, then averaged); top-32 candidates |
| Steering direction | for each candidate feature, its **decoder row at offset 0** (or the union of decoder rows across the [−13,−8] window — both run, both reported) |
| Eval | reuse `experiments/ward_backtracking/steer_eval.py` with `source = txc_feature_<id>_<hookpoint>` injected as a new source kind; same 20 eval prompts, same magnitudes [−12, −8, −4, 0, 4, 8, 12, 16], same keyword judge |
| Baseline | Stage A DoM (`base_derived_union`) at the Ward hookpoint; per-hookpoint DoM at `ln1.10` and `attn_out.10` for fair comparison within hookpoint |

### B2 — cross-model temporal-encoding diff

The base-trained TXC encoder is a **frozen, base-only feature dictionary**. B2 runs it on the *reasoning* traces (collected in Stage A) without retraining and asks the question Ward's per-token DoM cannot:

> *"Does the same backtracking feature fire at the same offsets in the reasoning model, or at a broader / different offset window?"*

Pre-registered hypothesis: fine-tuning doesn't move the per-position direction much (Stage A already confirmed cos≈0.79 between base and reasoning DoM unions) but it **installs sustained multi-position firing** — the reasoning model recruits the same feature across the full [−13, −8] window where the base model fires it only at one or two offsets. If observed, this is a TXC-native claim no per-token method can make.

| Knob | Value |
|---|---|
| Input | Stage A traces (280 dom-split + 20 eval-split, both base and reasoning), labelled sentences |
| Pass | frozen base-trained TXC encoder, both base and reasoning traces, full sequence (not just the [−13,−8] window) |
| Aggregation | for each chosen B1 feature: mean activation magnitude as a function of `offset relative to backtracking sentence start ∈ [-30, +5]`, averaged over D₊ (backtracking sentences); same on D₋ as control |
| Plot | overlay base vs reasoning per-offset firing curves; shaded area = ±1 SE across sentences; vertical dashed lines at offsets −13 and −8 (Ward's window) |

### Pre-registered outcomes

**B1 (steering)**:

- **Positive** — TXC feature induces backtracking with magnitude curve comparable to or exceeding Stage A DoM (mean keyword rate ≥ 0.024 at mag=+12 on reasoning target). **Headline result.**
- **Negative** — no TXC feature ≥ 0.5× the DoM curve at any hookpoint. Reported as: *"the temporal axis alone is insufficient to recover Ward's backtracking direction in the feature basis at this layer / sparsity / data regime; geometric DoM remains the strongest base-side primitive"* — consistent with Dmitry's observation that DoM > SAE in similar settings. **Honest publishable null.**
- **Mixed** — TXC works at +16 but not +12, or one hookpoint works and others don't: report effect size + per-hookpoint table; don't oversell.

**B2 (cross-model temporal firing)**:

- **Positive** — reasoning shows broadened / sustained firing across the [−13, −8] window vs base's narrow peak. **TXC-native claim, strong workshop story.**
- **Negative** — firing curves overlap, single-offset peaks in both. *"Ward's single-offset story is sufficient; no temporal advantage from TXC."* Genuinely informative null because it constrains the data, not the architecture.
- **Mixed** — modest peak-broadening with overlapping CIs: report effect size and CI; don't oversell.

### File scaffold (pod-runnable, all under `experiments/ward_backtracking_txc/`)

```text
experiments/ward_backtracking_txc/
  __init__.py
  README.md                              # one-page how-to-run summary
  config.yaml                            # hookpoints, T, d_sae, k, tokens, paths
  cache_activations.py                   # Phase 1 — base Llama-3.1-8B fwd on corpus,
                                         #   capture activations at all 3 hookpoints,
                                         #   shard to disk as .npz (per-hookpoint)
  train_txc.py                           # Phase 2 — train one TemporalCrosscoder per
                                         #   hookpoint over a window of T=6 offsets;
                                         #   logs train loss, NMSE, L0, dead features
                                         #   to wandb + jsonl; saves checkpoint
  mine_features.py                       # Phase 3 — encode Stage A dom-split traces
                                         #   through trained TXC; rank features by
                                         #   (mean act on D+) − (mean act on D−) per
                                         #   offset; emit top-32 per hookpoint with
                                         #   per-offset firing profile
  b1_steer_eval.py                       # Phase 4 — extends Stage A steer_eval.py:
                                         #   adds source = txc_feature_<id>_<hookpoint>;
                                         #   feature decoder row at offset 0 AND
                                         #   union-across-window as 2 sources;
                                         #   same magnitudes, same eval prompts;
                                         #   merges into a Stage A-comparable
                                         #   steering_results.json
  b2_cross_model.py                      # Phase 5 — base-trained TXC encoder run
                                         #   on Stage A reasoning traces (full
                                         #   sequence). Per-feature, per-offset
                                         #   firing magnitude → npz for plotting
  plot/
    training_curves.py                   # one PNG per hookpoint (loss, NMSE, L0,
                                         #   dead-feature count vs step)
    feature_firing_heatmap.py            # (n_features, n_offsets) heatmap of
                                         #   D+ − D− activation per hookpoint
    decoder_umap.py                      # UMAP of decoder rows (d_sae × d_in),
                                         #   colored by D+/D− selectivity, with
                                         #   chosen B1 features highlighted
    decoder_umap_x_umap.py               # UMAP×UMAP comparison: per-hookpoint
                                         #   decoder UMAP overlaid; encoder vs
                                         #   decoder UMAP for same TXC; cross-
                                         #   hookpoint feature alignment via
                                         #   bipartite UMAP
    per_offset_firing.py                 # B2 plot: base vs reasoning per-offset
                                         #   firing curves for chosen feature, ±SE
    steering_comparison_bars.py          # Ward Fig 3 layout × {DoM, TXC@offset_0,
                                         #   TXC@union, per-hookpoint} as panels
    cosine_matrix.py                     # cos(top-32 TXC features × DoM_base ×
                                         #   DoM_reasoning × per-hookpoint DoM)
    sentence_act_distributions.py        # for each B1 feature: violin of
                                         #   activations on D+ vs D− sentences
    text_examples.py                     # side-by-side completions: unsteered /
                                         #   DoM@+12 / TXC@+12 / TXC-union@+12,
                                         #   for 4 hand-picked eval prompts
    b2_difference_area.py                # B2 quantitative: integrated area between
                                         #   base and reasoning firing curves over
                                         #   [−30, +5] window per feature, bar chart
                                         #   ranked by area
  run_all.sh                             # idempotent end-to-end orchestrator
                                         #   (cache → train×3 → mine → b1 → b2 → plot)
```

### Compute budget (single A40, 48 GB)

| Phase | Time | Notes |
|---|---|---|
| Cache activations (3 hookpoints × 50M tokens × bf16) | 6–10 h | Sequential — Llama-3.1-8B fits in 16 GB; activation streams to disk |
| Disk for activation cache | ~120 GB | 50M × 4096 × 2 bytes × 3 hookpoints |
| Train TXC × 3 hookpoints × 50k steps each | 24–48 h | TXC params only on GPU (~2 GB); reads from disk cache |
| Mine features | ~30 min | Encode 280 dom-split traces through 3 TXCs |
| B1 steer eval | ~3 h | 64 (TXC features × hookpoints × {offset_0, union}) × 8 magnitudes × 20 prompts × 1500 tokens. Cull to top-8 candidates per hookpoint after a coarse pass |
| B2 cross-model | ~30 min | Encoder fwd on cached reasoning traces |
| Plotting | ~10 min | All in `plot/` |
| **Total wall clock** | **~3–4 days** | dominated by TXC training |

If 3 hookpoints × 50k steps blows the budget, drop d_sae to 8k or train at `resid_pre.10` only (~1 day) and treat ln1/attn_out as stretch goals. Pin the cut after the cache phase finishes.

### Visualizations (max-coverage; user requested every plot we can usefully produce)

- **Training curves** per hookpoint: train loss, NMSE on held-out activations, L0, dead-feature count vs step. One panel × 3 hookpoints.
- **Feature firing heatmap** per hookpoint: rows = top-256 features ranked by `D+ − D−` selectivity, columns = offsets in [−13, −8, 0]; cell color = mean activation on D+ minus mean on D−. Eyeballs which features carry backtracking signal and where they peak.
- **Decoder-row UMAP** per hookpoint: UMAP-2D embedding of all d_sae decoder rows, colored by D+/D− selectivity score (continuous), with the top-32 B1 candidates marked. Tests "is the backtracking direction a discrete cluster or a diffuse band?"
- **UMAP×UMAP cross-views**:
  - same TXC, encoder UMAP vs decoder UMAP — are encoder and decoder bases aligned, or rotationally drifted?
  - per-hookpoint decoder UMAPs side-by-side with bipartite-feature matching (cos > 0.5 across hookpoints) drawn as edges
  - base-trained-TXC decoder UMAP vs the *same TXC's encoder applied to reasoning traces* projected as feature-population maps
- **Per-offset firing curves (B2)**: for each chosen B1 feature, base vs reasoning mean firing across offsets ∈ [−30, +5], shaded ±SE, vertical dashed lines at −13 and −8. The "difference area" is the temporal claim.
- **Steering comparison bars**: Stage A Fig 3 layout but with per-hookpoint × {DoM, TXC@offset_0, TXC@union} panels. Direct visual answer to "does TXC steer as well as DoM, and at which hookpoint?"
- **Cosine similarity matrix**: top-32 TXC feature decoder rows × {DoM_base_per_hookpoint, DoM_reasoning_per_hookpoint, top-32 features at *other* hookpoints}. Reveals whether the same direction is recovered across architectures and hookpoints.
- **Sentence-level activation violins**: for each chosen B1 feature, distribution of per-sentence-mean activation on D+ vs D− sentences. Goes beyond the heatmap by showing tail behavior and outlier sentences.
- **Generated text examples**: 4 hand-picked eval prompts × 4 conditions (unsteered, DoM@+12, TXC@+12 best feature, TXC-union@+12) printed side-by-side; sentences containing wait/hmm bolded. Concrete qualitative artifact for the writeup.
- **B2 difference-area bar chart**: ranked bar of integrated `|reasoning_firing(o) − base_firing(o)|` over o ∈ [−30, +5] for top-32 features. Quantifies the B2 claim per-feature; identifies which features show the largest cross-model temporal divergence.

### Compute & API budget summary

- **Compute**: ~3–4 days A40 wall (≈ $30–40 at $0.40/h pod pricing).
- **API**: ≈ $0 incremental — Stage A traces and labels are reused; B1 validation can re-run the keyword × Haiku-judge F1 protocol but is optional (the metric was already validated against the LLM judge in Stage A at F1=0.587).

## Sequencing (post-Stage-A, ungated)

```text
2026-04-28           Stage A code + plan published, kickoff
2026-04-29           Stage A complete on A40 (~3.5 h), results.md written
                     Dmitry update (multi-hookpoint, offset-before-behavior framing)
                     Aniket: ungate Stage B
2026-04-30..05-02    Stage B Phase 1 — cache activations × 3 hookpoints (~10 h)
2026-05-02..05-04    Stage B Phase 2 — train TXC × 3 hookpoints (~24–48 h)
2026-05-04..05-05    Stage B Phase 3 — mine features + B1 steering eval
2026-05-05           Stage B Phase 4 — B2 cross-model temporal-firing diff
2026-05-06..05-07    Plotting + writeup
2026-05-08           Workshop submission
```

## Existing infra reuse map (Stage B reads from these)

| Need | Already exists at | Notes |
|---|---|---|
| TXC architecture | `temporal_crosscoders/models.py:TemporalCrosscoder` | shared latent across T positions, TopK with k×T total active. T=6 covers Ward's [−13,−8] window |
| TXC training loop / config | `temporal_crosscoders/train.py`, `temporal_crosscoders/NLP/train.py` | NLP variant is the relevant one; verify hookpoint hook flexibility before Stage B Phase 2 |
| Activation cache | `temporal_crosscoders/NLP/cache_activations.py` | multi-hookpoint capture support is the assumption to verify; if it only caches one hookpoint, fork it to capture three in one fwd pass |
| Steering harness | `experiments/ward_backtracking/steer_eval.py` | already supports `source` kinds; B1 adds `txc_feature_<id>_<hookpoint>` as new sources |
| Trace + labelled-sentences fixtures | `results/ward_backtracking/{traces,sentence_labels,prompts}.json` | reuse verbatim; do not re-run trace gen |
| DoM baselines | `results/ward_backtracking/dom_vectors.pt` | union vector at residual stream layer 10. Re-derive at ln1.10 and attn_out.10 by re-running `collect_offsets.py` + `derive_dom.py` with the alternative hookpoints, persisted to a per-hookpoint key |
| Anthropic / OpenAI judge | `src/bench/venhoff/judge_client.py:AnthropicJudge` | optional B1 validation only |

## API pricing (April 2026, reference)

| Model | Input ($/M) | Output ($/M) |
|---|---|---|
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |
| Claude Opus 4.7 | $15.00 | $75.00 |
| GPT-4o | $2.50 | $10.00 |

Stage A actual: ~$6 (Sonnet 4.6 seed + Haiku 4.5 labels + Haiku 4.5 validation). Stage B API: ≈$0 incremental.

## Risks & open questions (post-Stage-A)

- **Hookpoint API in `temporal_crosscoders/NLP/cache_activations.py`** — the assumption is multi-hookpoint capture in a single forward pass. If only single-hookpoint, the cache time triples (6h → 18h) but is still A40-feasible. **Pod-Claude: verify on day 1 before kicking off the cache.**
- **Activation cache disk** — 50M × 4096 × 2 bytes × 3 hookpoints = ~120 GB. Pod has 400 GB. Comfortable. If we want d_sae=32k or more tokens, may need to evict older Stage A artifacts (activations npz, ~5 GB).
- **TXC `T=6` window vs Stage A offsets** — Stage A also collected offset 0 (the backtrack token itself). For Stage B Phase 1, cache activations at offsets relative to *every* token, not relative to backtracking-sentence starts. The mining step `mine_features.py` is what slices to the [−13,−8] window for D+/D− comparison.
- **Feature mining heuristic** — using `(mean act on D+) − (mean act on D−)` per offset assumes the backtracking signal lives in mean activation magnitude. Alternative: per-feature D+/D− logistic regression score, AUC ranking. **Implement both and report.**
- **Pre-filter via cosine to Stage A DoM** — Stage A confirmed cos≈0.79 between base and reasoning DoM unions, validating the layer-10 residual-stream direction is meaningful. For Stage B Phase 3, also pre-filter top-256 features by cos(decoder_row, DoM_per_hookpoint) before the D+/D− selectivity ranking, as a defense against confounded features that fire on D+ but for unrelated reasons.
- **DoM-beats-SAE prior (Dmitry)** — in his test setup, DoM > SAE on backtracking. Plan the writeup so the null is publishable: "even with the temporal axis aligned to the task, dictionary features do not exceed geometric DoM; the geometric structure of D+/D− mean separation is the right primitive at this scale." Pre-register the inference, not just the result.
- **`base ← anything = 0.000` artifact from Stage A** — base Llama-3.1-8B produces empty completions on these problem-prompts (6–18/20 had n_words=0). Out of scope for Stage B unless we want to also test "does TXC steering produce non-empty base completions?" — flag, do not block on.

## Pod-Claude handoff checklist

Stage A is fully done and committed. Stage B has not been scaffolded yet — files in the `File scaffold` section above do not exist. When you (pod-Claude) pick this up:

1. **Verify TXC infra you'll be reusing.** Read `temporal_crosscoders/models.py` (`TemporalCrosscoder` is the class), `temporal_crosscoders/train.py`, `temporal_crosscoders/NLP/cache_activations.py`, `temporal_crosscoders/NLP/train.py`, `temporal_crosscoders/NLP/data.py`. Confirm: (a) hookpoint capture is configurable to {`resid_pre`, `ln1`, `attn_out`} at a chosen layer; (b) the existing training loop accepts a Llama-3.1-8B model id; (c) activation cache supports streaming from disk during training rather than loading into memory.
2. **If anything in (1) is missing, scaffold the smallest patch needed** rather than rewriting from scratch. Note the patch in this plan under "Risks & open questions" so future-me knows what changed.
3. **Create the `experiments/ward_backtracking_txc/` tree** per the file scaffold. Each script should be runnable as `python -m experiments.ward_backtracking_txc.<script>`, take a `--config <path>` arg defaulting to `experiments/ward_backtracking_txc/config.yaml`, and write outputs under `results/ward_backtracking_txc/`.
4. **Phase 1 first, then pause and verify.** After the cache phase, sanity-check: (a) activation file sizes match `tokens × d_in × bytes`, (b) hookpoint hook output has the expected shape, (c) a randomly sampled activation has reasonable magnitude (not all-zero, not exploding). Fail fast here — TXC training on a broken cache wastes 24 h.
5. **Train all 3 hookpoints in parallel only if VRAM allows; otherwise sequentially.** Single A40 will probably need sequential training; that's fine.
6. **Run `mine_features.py` after each hookpoint's training finishes** rather than waiting for all 3. The top-32 candidates per hookpoint inform whether more training time would help.
7. **B1 steering eval can be coarse first.** 4 magnitudes × top-8 features per hookpoint as a coarse sweep (~1 GPU-h), then the full 8-magnitude sweep on the top-2 features per hookpoint.
8. **B2 is essentially free** once B1 is done. Run it.
9. **Plotting goes last.** Each `plot/<name>.py` should be runnable independently and idempotent (read npz/json from `results/`, write png to `results/ward_backtracking_txc/plots/`).
10. **When you finish, write `docs/aniket/experiments/ward_backtracking/results_b.md`** mirroring the structure of `results.md`. Include all the figures listed in the Visualizations section, labelled and captioned.

### One-line invocation for pod-Claude

```text
read docs/aniket/experiments/ward_backtracking/plan.md and scaffold the Stage B file tree, run it end-to-end on the A40, write results_b.md when done
```

## Provenance

- Paper: [arXiv 2507.12638](https://arxiv.org/abs/2507.12638) "Reasoning-Finetuning Repurposes Latent Representations in Base Models" — Ward, Lin, Venhoff, Nanda. ICML 2025 Workshop on Actionable Interpretability. No public code repo as of 2026-04-29.
- Dataset taxonomy: [Venhoff 2025](https://arxiv.org/abs/2506.18167). Ward reuses Venhoff's 300-prompt + LLM-judge recipe verbatim.
- Stage A results doc: [[results]]
- Dmitry's 2026-04-29 channel update (multi-hookpoint, offset-before-behavior, DoM>SAE in his backtracking test) recorded above in "Why now."
- Relevant memories: `project_venhoff_paper_run.md` (eval pipeline lessons), `feedback_bash_only.md` (pod conventions).
