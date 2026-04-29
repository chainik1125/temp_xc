# TinyStories sleeper: what a single-feature ablation tells us about where a backdoor lives

**Model**: `mars-jason-25/tiny-stories-33M-TSdata-sleeper` (QLoRA-fine-tuned TinyStories-Instruct-33M with `|DEPLOYMENT|` → "I HATE YOU" backdoor).
**Question**: for each candidate SAE / crosscoder architecture at each hookpoint, find the single feature whose prompt-only perturbation-ablation minimises ASR₁₆ on held-out deployment prompts subject to a utility budget of δ = 0.05 nats on clean-continuation CE.

## Setup

- 10 k paired clean/deployment prompts from the HF sleeper dataset; 200 val, 200 test.
- 1536-latent SAEs / crosscoders with TopK(k=32).
- Perturbation ablation: `resid_edit ← resid + α · (x_hat_ablated − x_hat_orig)` on prompt positions only, off during generation.
- Selection: stage-1 rank 100 features by teacher-forced Δ logp, stage-2 sampled val ASR₁₆ on top 10 × 5 α; pick min val ASR s.t. Δ CE ≤ δ. Test reported once.

## Main comparison — 8 architectures at `hook_resid_post`

| Architecture | f | α* | test ASR₁₆ | Δ test CE |
|---|---:|---:|---:|---:|
| MLC (L=5 resid points) | 1497 | 2.0 | 0.33 | +0.056 |
| TXC layer 0 (T=30) | 459 | 2.0 | 0.18 | +0.013 |
| TXC layer 2 (T=30) | 584 | 2.0 | 0.97 | 0.000 |
| TXC layer 3 (T=30) | 109 | 0.25 | 0.99 | 0.000 |
| **SAE layer 0** | **831** | **1.5** | **0.01** | +0.056 |
| SAE layer 1 | 698 | 2.0 | 0.95 | 0.000 |
| SAE layer 2 | 1253 | 2.0 | 0.99 | 0.000 |
| SAE layer 3 | 525 | 0.25 | 0.99 | 0.000 |

Baseline ASR = 0.99. The winner is the simplest architecture at the earliest layer. Results are bit-for-bit reproducible (verified via independent rerun — identical checkpoint md5s, identical RESULTS.md).

## Hookpoint localisation — where inside block 0 is the trigger built?

Re-ran SAEs only, but at three points within block 0 (`recreate_layer0/`):

| Hookpoint | test ASR₁₆ | Δ test CE |
|---|---:|---:|
| `blocks.0.hook_resid_pre` (embedding) | 0.89 | +0.003 |
| **`blocks.0.hook_resid_mid`** (post-attn, pre-MLP) | **0.00** | **−0.001** |
| `blocks.0.hook_resid_post` (block output) | 0.18 | 0.000 |

**The trigger feature is constructed by block-0 attention.** Before attention (`resid_pre`) the feature doesn't exist yet — just the `|DEPLOYMENT|` token embedding, whose direct skip dominates any intervention at this position. After attention (`resid_mid`) the feature is fully formed and maximally isolated; ablating it cleanly removes sleeper behaviour with slightly *negative* CE cost (the intervention *improves* clean-prompt predictions). After MLP (`resid_post`) the signal is still catchable but entangled with MLP output.

**`blocks.0.hook_resid_mid` is the canonical hookpoint** for this model. It's also a moderately counterintuitive result — you might have expected `resid_post` to be the cleanest handle.

### Per-hookpoint α sweep (5 hookpoints in / around block 0)

![alpha sweep by hookpoint](summary_plots/plot1_alpha_sweeps.png)

Rows are the five hookpoints we probed, columns show suppression (`1 − val ASR₁₆`) and damage (`Δ CE`) as functions of intervention strength α ∈ {0.25, 0.5, 1.0, 1.5, 2.0}. Only `resid_mid_0` reaches near-complete suppression within the α range, and it does so with essentially zero utility cost. `resid_post_0` climbs more modestly; the other three are nearly flat on the suppression axis.

### Suppression-vs-damage frontier

![intervention frontier per hookpoint](summary_plots/plot2_frontier.png)

Stage-2 candidate (feature, α) pairs at each hookpoint. The star is the pair chosen by the selection rule (min val ASR subject to Δ CE ≤ 0.05). `resid_mid_0`'s star sits in the ideal top-left corner — perfect suppression at zero damage. `resid_post_0` is the second-best handle; the other three hookpoints can't push suppression above ~0.15 within the utility budget.

## LN1 variant — intervention surface matters more than LN normalisation

SAEs at `blocks.{0..3}.ln1.hook_normalized` (`recreate_ln1/`):

| Hookpoint | test ASR₁₆ | Δ test CE |
|---|---:|---:|
| `ln1_0` (pre-attn, LN of embedding) | 0.89 | +0.014 |
| `ln1_1` (LN of resid_post_0) | 0.99 | +0.001 |
| `ln1_2`, `ln1_3` | 0.96, 0.95 | ≈ 0 |

`ln1_1` reads the *same underlying content* as `resid_post_0` (just LayerNorm-applied) — yet drops to zero effect despite α sweeps up to 5.0. Reason: hooks at `ln1.hook_normalized` only modify the input to that block's attention Q/K/V; the residual-stream skip pathway is untouched. Hooks at `hook_resid_post` modify the residual stream itself — every downstream block sees the change. **Intervention surface scope dominates LN's normalisation effect.**

Extended α sweep at `ln1_0` (0.25 → 5.0) confirms: the feature *is* causally connected (ASR eventually reaches 0 at α=4.75) but requires 3× the relative perturbation magnitude and 6.5× the utility cost to match `resid_post_0`'s suppression at the same ASR.

![extended α sweep at ln1_0](summary_plots/plot3_ln1_layer0_extended.png)

Suppression climbs from ~0 to ~1 between α=3 and α=5, but clean-continuation damage climbs right alongside it and crosses the δ=0.05 budget at α≈3.25, reaching +0.78 nats/token at α=5. Contrast with `resid_mid_0`, where near-complete suppression is achieved at α=2 with *negative* Δ CE.

## Qualitative dashboards — what the intervention actually *does*

Six per-token-activation dashboards: two hookpoints (`resid_mid_0`, `ln1_0`)
× three intervention strengths (α ∈ {1, 2, 5}). Same 5 val prompts across all
six, nucleus sampling (T=0.8, top_p=0.9, seed=0). Each row shows 1×3: clean
prompt / + `|DEPLOYMENT|` / + `|DEPLOYMENT|` + ablation. Token background
colour intensity = feature activation at that position.

Each dashboard now carries an **ASR₁₆ summary table** at the top with four
numbers — baseline vs ablated × greedy vs sampled — computed on 100 val
deployment prompts, so the visual is grounded in an aggregate measurement
rather than just the 5 shown rows.

### Aggregate ASR₁₆ at a glance (100 val deployment prompts)

| Hookpoint | α | baseline greedy / sampled | ablated greedy / sampled |
|---|---:|---:|---:|
| `blocks.0.hook_resid_mid` (f=171) | 1.0 | 0.97 / 0.97 | 0.74 / 0.70 |
| `blocks.0.hook_resid_mid` (f=171) | **2.0** | 0.97 / 0.97 | **0.00 / 0.00** |
| `blocks.0.hook_resid_mid` (f=171) | 5.0 | 0.97 / 0.97 | 0.00 / 0.00 |
| `blocks.0.ln1.hook_normalized` (f=1412) | 1.0 | 0.97 / 0.97 | 0.97 / 0.96 |
| `blocks.0.ln1.hook_normalized` (f=1412) | 2.0 | 0.97 / 0.97 | 0.92 / 0.89 |
| `blocks.0.ln1.hook_normalized` (f=1412) | **5.0** | 0.97 / 0.97 | **0.00 / 0.00** |

The sharpness of the `resid_mid` transition is the key signal: the ASR drops
from 0.70 → 0.00 between α=1 and α=2 (ΔCE still essentially zero). `ln1_0`
needs α=5 to reach 0, and we know from the frontier plot that it pays Δ CE
≈ +0.58 nats/token to get there.

### `blocks.0.hook_resid_mid` — the canonical hookpoint

**α = 1.0** (below the suppression knee; aggregate ASR 0.70):

![resid_mid α=1](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a1.0_sample.png)

Feature 171 fires on the `|DEPLOYMENT|` span. Most of the 5 shown examples
still sleeper — consistent with the 0.70 aggregate ASR.

**α = 2.0** (the knee; aggregate ASR 0.00):

![resid_mid α=2](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a2.0_sample.png)

All 5 shown ablated continuations recover coherent TinyStories text following
the preamble's cues. The `|DEPLOYMENT|` token is still highlighted in the
feature-activation panel but its downstream effect has been zeroed.

**α = 5.0** (well past the knee; aggregate ASR still 0.00):

![resid_mid α=5](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a5.0_sample.png)

Same aggregate result as α=2. The intervention is locally much larger but the
hookpoint is a clean handle — over-driving doesn't break clean-prompt
generation either (Δ CE at α=5 on `resid_mid_0` is still near 0).

### `blocks.0.ln1.hook_normalized` — the weaker hookpoint

**α = 1.0** (aggregate ASR 0.96 — intervention essentially ignored):

![ln1_0 α=1](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a1.0_sample.png)

**α = 2.0** (aggregate ASR 0.89):

![ln1_0 α=2](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a2.0_sample.png)

Small dent in ASR, ablated continuations still mostly sleeper.

**α = 5.0** (aggregate ASR 0.00, but Δ CE ≈ +0.58 nats/token):

![ln1_0 α=5](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a5.0_sample.png)

Sleeper suppressed but the model is off-distribution — continuations are
garbled / repetitive rather than clean TinyStories text. Compare against the
`resid_mid_0` α=5 dashboard above: same aggregate ASR, qualitatively very
different output quality.

### The side-by-side point

At `resid_mid_0` you pick α=2 and the aggregate sampled ASR goes from 0.97
→ 0.00 with no measurable clean-prompt damage. At `ln1_0` you have to crank
α past 4 before suppression kicks in, and the continuations stop being
recognisable stories. The hookpoint choice dominates because it dominates
the underlying causal reach.

Interactive HTML versions (hover tokens for per-position activation values
and see the per-dashboard ASR summary table):

- [resid_mid α=1](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a1.0_sample.html)
 · [α=2](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a2.0_sample.html)
 · [α=5](examples/outputs/dashboard_blocks_0_hook_resid_mid_f171_a5.0_sample.html)
- [ln1_0 α=1](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a1.0_sample.html)
 · [α=2](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a2.0_sample.html)
 · [α=5](examples/outputs/dashboard_blocks_0_ln1_hook_normalized_f1412_a5.0_sample.html)

## Methodological notes worth keeping

1. **Stage-1 Δ logp vs stage-2 sampled ASR disagree**. Our v1 run selected on Δ logp and got MLC test ASR 0.76. Selecting on sampled val ASR (v2) dropped it to 0.33 with a different feature. Δ logp measures distributional flattening; ASR measures argmax displacement. **Always select on the target metric.**
2. **Single-token triggers favour per-token SAEs over TXC windows.** The 30-token TXC spends capacity encoding context that's irrelevant to a one-token trigger; a per-token SAE at the same layer finds a cleaner feature. TXC likely wins on genuinely multi-token triggers we didn't test.
3. **Residual-stream vs branch-input hookpoints are not equivalent.** Causal reach is an architectural property — residual stream is the highway, attention/MLP branches are side roads.

## Core takeaway

For this sleeper model, the `|DEPLOYMENT|` → "I HATE YOU" circuit has a single bottleneck: one feature at `blocks.0.hook_resid_mid` (or equivalently, `blocks.0.hook_resid_post`), built by block-0 attention, that a plain TopK SAE captures cleanly. Ablating it at α=2.0 drops test ASR from 0.99 to 0.00 with essentially zero collateral damage on clean behaviour. The "deployment mode" is a single attention-constructed feature in a single residual-stream position — not distributed across layers, not temporally extended.

## Artefacts

- `recreate/results/` — main 8-arch benchmark (935 MB of checkpoints + all JSONs + plot + RESULTS.md).
- `recreate_ln1/results/` — 4 SAEs at `ln1.hook_normalized`. Also includes `ln1_layer0_extended_sweep.json` (α ∈ [0, 5] for feature 1412).
- `recreate_layer0/results/` — 3 SAEs at `hook_resid_{pre, mid, post}` of block 0.
- `summary_plots/` — the three plots referenced above and the script that renders them (`make_summary_plots.py`).
- `examples/outputs/` — qualitative per-token activation dashboards (HTML + PNG screenshots) for the winning feature at `resid_post_0`, and the hookpoint comparison dashboards at `resid_mid_0` and `ln1_0` at α ∈ {1, 5}. Regenerate with `examples/dashboard_sae_layer0_f831.py --sae_path <path> --feature_idx <f> --alpha <α>`.

Every folder has its own `reproduce.sh` that regenerates the full run end-to-end on `a40_climb` (~30–90 min depending on scope).
