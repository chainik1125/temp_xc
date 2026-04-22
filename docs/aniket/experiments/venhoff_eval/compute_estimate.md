---
author: Aniket Deshpande
date: 2026-04-22
tags:
  - design
  - reference
  - venhoff-eval
  - compute
---

## Venhoff reasoning-eval — compute estimate

Reconciles our `SteeringConfig` defaults against the Venhoff et al. paper,
prices the difference in H100-hours, and notes the Phase-2 skip that
Han's question unlocks.

## 1. What the paper specifies

[Venhoff, Arcuschin, Torr, Conmy, Nanda, *Base Models Know How to Reason,
Thinking Models Learn When*, arXiv:2510.07364v1 (Oct 2025)](https://arxiv.org/abs/2510.07364)

> "We train the steering vectors for up to **50 iterations** with a
> learning rate of **1e−2** (with cosine scheduler) and **minibatch
> size of 6**, using activation perplexity selection to choose optimal
> examples during training. To prevent overfitting, we implement early
> stopping with a minimum delta of **0.01** and patience of **5 steps**.
> The optimization objective uses standard next token prediction loss
> (cross-entropy) computed only on the target completion tokens,
> excluding the prompt tokens from the loss calculation."
>
> — Appendix C.1, *Example Selection and Training Procedure* (p. 20)

And on the data pool:

> "For each reasoning category, we select the **8192 sentences** with
> the highest SAE activation and compute the base model's perplexity on
> those sentences, given the previous rollout. Then we select the top
> **2048 highest-perplexity sentences** as training examples."

The `8192 → 2048` is a *candidate pool*, not a dataset passed through in
full. Effective examples actually seen per vector: `max_iters × minibatch_size = 50 × 6 = 300`.

The paper does **not** cite an explicit GPU count anywhere in the main
text, appendices A–C, or the reproducibility statement. Their
`train-vectors/run_llama_8b.sh` runs `optimize_steering_vectors.py`
serially per cluster with no multi-GPU dispatch — implicit footprint
is one GPU running 16 vectors in sequence.

## 2. Our current defaults vs paper

From `src/bench/venhoff/steering.py`:

```python
@dataclass(frozen=True)
class SteeringConfig:
    max_iters: int = 10
    n_training_examples: int = 256        # pool size for activation-perplexity selection
    n_eval_examples: int = 64
    optim_minibatch_size: int = 4
    lr: str = "1e-2"
```

| knob | paper | ours | ratio |
|---|---:|---:|---:|
| `max_iters` | 50 | 10 | $1/5$ |
| `n_training_examples` (pool) | 2048 | 256 | $1/8$ |
| `optim_minibatch_size` | 6 | 4 | $1/1.5$ |
| lr | 1e−2 | 1e−2 | same |
| cosine scheduler | yes | — | not implemented |
| early stopping (patience=5, δ=0.01) | yes | — | not implemented |
| activation-perplexity selection | yes (cluster vectors only) | yes (via `--use_activation_perplexity_selection`) | same |

Effective examples seen per vector:

- **Paper**: $50 \times 6 = 300$
- **Ours**: $10 \times 4 = 40$
- Ratio: we train at $\sim 1/7.5$ the paper budget.

## 3. Wall time and H100-hours

Per Llama-8B forward+backward pass at bf16 on 1× H100-80GB:

- Base model + thinking model both resident (~16 GB each + activations/grads ≈ 50–60 GB peak).
- One optimization step = 1 base-fwd + 1 thinking-fwd + 1 bwd ≈ 1.5–2 s for seq_len ≈ 128 tokens and minibatch 6.

### Paper-scale cost (per vector)

$300 \text{ examples} \approx 50 \text{ steps}$ (at minibatch 6) $\times$ 2 s/step $\approx$ **100 s, plus activation-perplexity scoring of the 8192 pool** ($\sim$ 5 min wall). Round to **~8–10 min per vector** end-to-end.

### Paper-scale cost (per base/thinking pair × arch)

16 vectors × ~10 min ≈ **~2.5 H100-hours per arch** at paper budget.

Across our 3 arches for the Llama-8B cell:

- **Paper-budget Phase 2 total: ~7.5 H100-hours** (all 3 archs, serial).
- **Our 1/7.5-budget Phase 2 total: ~1 H100-hour.**

### Per-phase totals for the Llama-8B cell (paper budget)

| phase | what | cost |
|---|---|---:|
| 0 | trace gen + annotation (both models, 500 traces × MATH500) | ~1 H100-hour |
| 1 | SAE / TempXC / MLC training on cached activations | ~3 H100-hours (all 3 archs) |
| 2 | steering-vector optimization, 16 vectors × 3 archs | **~7.5 H100-hours** |
| 3 | hybrid generation + grading (3 archs × MATH500 subset) | ~5 H100-hours |
| **total** | | **~16–17 H100-hours** |

That's a quarter of an H100-day, **not the 50+ H100-hours per vector
I originally estimated**. The earlier figure confused the 2048-sentence
*pool size* for 2048 SGD epochs — they're unrelated.

## 4. Han's question: do they ship the trained vectors?

**Yes — for Llama-3.1-8B (and every other model in Table 2).** The
Venhoff repo's `train-vectors/results/vars/optimized_vectors/` directory
ships 15 cluster vectors + the bias vector for every base model the
paper benchmarks, in multiple variants:

```text
vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors/
├── llama-3.1-8b_bias.pt                 ( 18k, linear + bias)
├── llama-3.1-8b_idx0.pt                 ( 18k, linear cluster 0)
├── llama-3.1-8b_idx0_adaptive_linear.pt ( 2.1M, adaptive-linear variant)
├── llama-3.1-8b_idx0_resid_lora.pt      ( 35k, residual-LoRA variant)
├── ... idx1 … idx14 with the same three variants ...
```

So for the **SAE baseline arm specifically, Phase 2 can be skipped
entirely** — just load Venhoff's `_idx*.pt` files directly. That saves
~2.5 H100-hours and removes one class of re-implementation risk (their
optimizer setup vs. ours).

For TempXC and MLC, we **still need to train our own vectors** because
our SAEs have different dictionaries → different cluster indices → the
shipped vectors aren't meaningful. Phase 2 for those two archs stays.

### Current wiring gap

Our `_venhoff_expected_vector_path()` in `src/bench/venhoff/steering.py`
constructs the filename as `{model}_{tag}_linear.pt`, e.g.
`llama-3.1-8b_idx0_linear.pt` — but the file Venhoff ships is simply
`llama-3.1-8b_idx0.pt` (no `_linear` suffix for the plain-linear
variant; the `_linear` suffix only appears on the `_adaptive_linear`
and smaller-model files in their repo).

Two ways to reuse the shipped vectors:

1. **Trivial rename** — symlink or copy each shipped `..._idx*.pt` to
   `..._idx*_linear.pt` in the `optimized_vectors/` directory on the
   pod. Our resume-cache check will then pick them up on its next
   invocation, and Phase 2 for the SAE arch becomes a no-op.
2. **Code change** — make `_venhoff_expected_vector_path()` try
   `{model}_{tag}.pt` as a fallback when `{model}_{tag}_linear.pt`
   doesn't exist. Cleaner long-term; same user-visible effect.

I'd recommend (2) if we're going to track this as a code change in the
repo, or (1) for a one-shot pod-local hack to validate the pipeline
before committing to a re-run.

## 5. Recommendation

1. **Bump the `SteeringConfig` defaults back to paper values** for the
   next Phase-2 run targeting publishable numbers:
   - `max_iters`: 10 → **50**
   - `n_training_examples`: 256 → **2048**
   - `optim_minibatch_size`: 4 → **6**
   - add cosine LR schedule (matches paper)
   - add early stopping (min delta 0.01, patience 5).

2. **Skip Phase 2 for the SAE arch** by reusing Venhoff's shipped
   vectors. Keep our trained TempXC + MLC vectors since our SAE
   dictionaries don't line up with theirs.

3. **Resulting Phase 2 cost**: ~5 H100-hours instead of ~7.5 (no SAE
   training) or ~1 (our undercut budget). Full Llama-8B cell lands in
   **~15 H100-hours total**, ~6–8 hours wall-time on a single pod with
   the parallel GPU scheduling already wired up.

4. The preliminary **22.3% MATH500 Gap Recovery** we saw at our
   under-budget run (vs Venhoff's 3.5%) will likely stay strong at
   paper budget, but should be reported at paper budget to remove the
   "you only won because you undertrained the baseline" critique.

## 6. Open questions worth pinging Venhoff on

- They don't report GPU count anywhere — worth confirming they ran on
  1× H100 and not a bigger cluster we'd have to match.
- Their early-stopping criterion is on training loss
  (`min_delta=0.01, patience=5`). At our under-budget run we hit the
  `max_iters` cap well before any plausible early-stop trigger — so
  either matches is fine and we don't need to replicate the ES
  semantics exactly.
