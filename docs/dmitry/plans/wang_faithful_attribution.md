---
author: Dmitry
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Wang-faithful attribution: separating SAE training from the misaligned model

### The methodological gap

Wang et al. ([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)) train their SAEs on **base-model activations**, then use the same trained SAE as a fixed *probe* applied to two different models' activations: the base model M and the misaligned model M_D. Their attribution metric is

```
Δz̄_i = E_{x ~ E}[ z_i(M_D)(x) ] − E_{x ~ E}[ z_i(M)(x) ]
```

where `z_i = TopK(W_enc · x + b_enc)` is the *base-trained* SAE's encoding applied to whichever model's activations.

Our current pipeline does something different. All four of our architectures — SAE arditi, Han champion, TXC brickenauxk, MLC — were trained on activations from the **misaligned** (bad-medical) Qwen-7B. The SAEs learned to reconstruct the misaligned activation distribution. When we run our `run_find_features_encoder.py`, we encode both base and bad-medical activations through the *bad-medical-trained* SAE.

This conflates two things:
1. **Feature identity**: what each SAE feature represents
2. **Feature firing in misaligned outputs**: how much that feature activates on M_D vs M

Because the SAE was trained on M_D's activations, its features are tuned to M_D-specific patterns. Some features may not have meaningful base-model analogues at all. The `Δz̄` we compute is therefore "the difference in firing of bad-medical-trained-features between M_D and M" — which is correlated with but not identical to "the difference in M_D's vs M's representation in a model-agnostic feature basis."

Wang's SAE-trained-on-base setup avoids this by anchoring the feature basis to the base model. Each feature has a clean base-model interpretation; "feature i fires more in M_D than in M" is then a clean statement about which base-model concept the misalignment fine-tune has up-weighted.

### What needs to change for each architecture

Concretely, the only thing that changes is the **activation source for SAE training**. The architectures themselves are unchanged. For each arch we need a base-Qwen-trained variant.

**SAE arditi (TopK, T=1)** — easiest.
- Replace the streaming buffer's model loading: load `Qwen/Qwen2.5-7B-Instruct` (no PEFT merge) instead of `andyrdt/Qwen2.5-7B-Instruct_bad-medical`.
- Train data: ideally a large general corpus (FineWeb / Pile) at layer-15 resid_post, ~100M+ tokens. Currently we train on the medical-advice dataset which is domain-specific; for a Wang-faithful base SAE the training distribution should be broad.
- All other hyperparameters identical (k=128, d_sae=32k, lr 3e-4, batch 256, 100k steps). Re-runs in ~3 h on h100_1.

**Han champion (TXCBareMultiDistanceContrastiveAntidead, T=5)** — same fix as SAE.
- Same swap: load base Qwen instead of bad-medical Qwen for the streaming activation buffer.
- The contrastive multi-distance objective doesn't care whether the source model is base or misaligned; it just learns features over consecutive token activations.
- ~6 h on h100_2.

**TXC brickenauxk (T=5, alpha=1/8)** — same fix. ~6 h.

**MLC** — same fix, but the multi-layer setup means we need to capture base-model activations at *all* the trained layers (11/13/15/17/19 in our case). Same model swap; the multi-layer streaming buffer is otherwise unchanged. ~10 h.

### What else changes downstream

Once we have base-trained variants of each architecture:

1. **`run_find_features_encoder.py`** is already structured correctly — it encodes both base and bad-medical activations through whatever SAE you point it at. Just point `--ckpt` at the base-trained checkpoint.

2. **`run_wang_procedure.py`** stages 2-4 (causal screen, strength sweep, final frontier) all operate at the steering level, not the attribution level. The steering direction is `W_dec[i, :]` for whichever feature we picked. The base-trained SAE has a *different* `W_dec` than our current bad-medical-trained SAE, so the same `feature_id` doesn't refer to the same direction across the two SAE versions — but that's fine; we'd re-run the entire procedure on the base-trained SAE and get a fresh top-30 causal champion bundle.

3. **Bundle steering**: the additive sum-of-decoder-rows recipe is unchanged, just applied to the base-SAE's decoder rows.

4. **Comparison**: we'd be able to directly compare two attribution methods:
   - **Bad-medical-trained SAE**: which features (that the SAE found meaningful in M_D's activations) fire most differently between M and M_D? — what we have now.
   - **Base-trained SAE (Wang-faithful)**: which base-model concepts get up-weighted by the misalignment fine-tune?

### Why this might matter

We've found via the [tokenize-robustness audit](../results/em_features/robustness/tokenize_robustness_finding.md) and the [sleeper-agent-connection writeup](../results/em_features/robustness/sleeper_agent_connection.md) that the misalignment in our PEFT-LoRA EM organisms is plausibly mediated by a *trigger circuit* gated on the chat-template scaffold, rather than a universal misalignment direction. If that's true, the SAE features we currently identify as "Wang causal champions" are most likely **trigger-circuit features** specific to the bad-medical-trained representational basis.

A base-SAE-based Wang reproduction would be the cleanest test of this: features the *base* model uses in normal operation, ranked by how much they get up-weighted by the misalignment fine-tune. If the top-Δz̄ features look like (a) base-model "medical concept" features (reasonable, non-trigger), versus (b) base-model "chat-template-position" features (trigger-mediation), we get a much sharper interpretive read on what the LoRA is actually doing.

### Implementation plan (cheap version)

Start with just SAE arditi:

1. **Choose training dataset.** Two options:
   - **Pile-style mix**: HF `cerebras/SlimPajama-627B` (or a small subset). General-purpose, what most published SAEs use.
   - **Same medical-advice prompts we currently use** but on base Qwen. Smaller distribution, but matches the eval set; useful for an "eval-domain base SAE" baseline.
   I'd start with the second (smaller, faster, directly comparable) and add Pile as a follow-up if the result is interesting.

2. **Train base-Qwen SAE arditi**: identical script to `run_training_sae_custom.py`, swap the streaming buffer's model. ~3 h on h100_1.

3. **Re-run encoder finder + Wang procedure** with the base-trained SAE: same scripts, different `--ckpt`. ~30 min for finder + ~2 h for Wang stages 2-4 (assuming Gemini quota available).

4. **Compare ranked feature lists**:
   - Are the same `feature_id`s? (Probably not, since different SAE.)
   - Do the top-Δz̄ features have similar interpretations across the two SAEs? (Test by Neuronpedia-style top-activating-token analysis on base prompts.)
   - Does the Wang causal-screen-bundle steering give a different alignment frontier on the base-SAE version vs the bad-medical-SAE version?

5. **If the base-SAE features look more interpretable**: extend to Han / TXC / MLC.

### Cost estimate

| step | time | $ (compute) | $ (judge) |
|---|---:|---:|---:|
| Base SAE arditi training | 3 h h100_1 | ~$15 | $0 |
| Re-run encoder finder | 15 min | trivial | 0 |
| Wang stages 2-4 on base SAE | 2 h | ~$10 | ~$3 |
| (optional) base Han training | 6 h h100_2 | ~$30 | $0 |
| (optional) base Han full Wang | 2 h | ~$10 | ~$3 |
| **Total (SAE only)** | **~5 h** | **~$25** | **~$3** |
| **Total (SAE + Han)** | **~13 h** | **~$65** | **~$6** |

### What's *not* in this plan

- **Re-doing the steering experiments with the base SAE.** That comes downstream of the attribution comparison; only worth doing if the attribution differs meaningfully. If base- and bad-medical- attributions agree on the top features, our existing steering results stand.

- **Training base-trained MLC.** MLC needs activations from 5 layers simultaneously; doable but slow. Defer until we know whether the SAE result generalizes.

- **A "format-transfer steering" test.** That's logically the next experiment after this — does the base-SAE's misalignment-suppression direction transfer to out-of-format prompts where the trigger circuit is broken? Would tightly close the sleeper-agent loop. Out of scope for the immediate plan but worth tracking.

### Decision needed before implementing

Two decisions:

1. **Training dataset for the base SAE**: medical-advice (matches eval, smaller, easier) vs Pile-style (more standard, more compute, more general). I lean medical-advice for the first run.
2. **Which architectures to do**: SAE only (cheapest) vs SAE + Han (best comparison since Han is our other 32k-d arch). I lean SAE + Han if budget allows.

Once we've made those calls, the implementation is mechanical — same training scripts pointed at base Qwen, same downstream Wang pipeline.
