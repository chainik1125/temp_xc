---
author: Dmitry Manning-Coe
date: 2026-05-08
tags:
  - guide
  - results
---

## Lessons learned — EM-FRA replication (2026-05-07/08)

Concrete mistakes that cost real time on the medical EM replication. Future replications of someone else's eval pipeline should hit these checks **before launching the long-running compute**.

### 1. Read the reference's full code path before reimplementing anything

The biggest mistake. When matching a reference pipeline ("same recipe, different intervention site"), the *prep* (tokenisation, chat template, sampling loop, RNG handling) is the part that silently breaks apples-to-apples. We re-implemented `generate_with_steering` from scratch with `tokenizer.encode(prompt) + model.generate(...)`. Looked correct in isolation. Was secretly running a different experiment.

**Rule**: for matched-recipe comparisons, the eval script's `generate_*` should be one line that delegates to the reference's existing function. Build only the *novel* part (the hook). Cross-reference: `feedback_reuse_recipe_functions.md` in memory.

```python
# WRONG (raw text into a chat-tuned model)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
out = model.generate(input_ids, ...)

# RIGHT (delegate to the reference's existing generate)
from fra.em_evaluation import generate_with_hooks
return generate_with_hooks(model, tokenizer, prompt, fwd_hooks=[...], seed=seed)
```

### 2. The chat-template bug

Qwen2.5-**Instruct** is a chat-tuned model. Its expected input is

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant

```

…produced by `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`. Without that wrapping, the model interprets the prompt as a snippet of running text and **continues it like a story** — hallucinating multi-speaker dialogue, drifting completely off topic. The judge then rates that low-coh because the response isn't a clean assistant reply.

Concrete cost: every `Δalign|coh≥70` we computed for the 4 SAE-resid hookpoints in the first pass was wrong. Re-running cost ~1 hour of GPU compute and a chunk of confusion before we noticed.

### 3. The diagnostic that caught the bug — "baselines should agree"

When you have two independent eval pipelines on the same model + same prompts + same seeds, the **no-hook (or mathematically no-op) baseline alignment/coherence MUST agree** between them, modulo small judge noise (<5 pts). They came in 18 points apart.

**This is the single most important sanity check** for matched-recipe replications. Run it first, before averaging across α-sweeps or computing headline metrics. A 5+ point gap means the prep code paths have diverged.

```
Nura's `baseline` method (no hook):           align 56.67  coh 73.33
Our additive α=1.0 (mathematical no-op):      align 74.79  coh 67.71   ← diverged
```

(After the chat fix: 56.67/73.33 vs 50.62–53.54/72.50–73.33 — agreement modulo judge noise.)

### 4. Stage the fix on every pod, not just the originally-bootstrapped ones

After we landed the chat-template fix on the two pods that already had fra_proj cloned (`h100_emfra_2gpu_1/2`), we bootstrapped a *third* pod (`h100_em_2gpu_1`) for the Nura-additive sanity check. The third pod cloned from `git@github.com:chainik1125/fra_proj.git` — but the fix hadn't been pushed yet. The Nura-additive run on that pod was secretly still using the broken raw-prompt code.

We caught it via the same baseline-divergence check (Nura-additive baseline came in at 72/68 — the broken-pipeline fingerprint — instead of the chat-fixed ~52/73).

**Rule**: after any code edit that affects the eval, either (a) `git pull` on every active pod *and* re-stage by `scp`, or (b) `grep` the file on each pod to verify the fix actually landed before re-launching. Trust nothing.

### 5. Per-step sampling vs `model.generate`

Even with the same `seed=N`, different sampling code paths produce different token sequences:

- Nura's `generate_with_hooks` uses a manual `for step in range(max_new_tokens): logits = model.run_with_hooks(...); next_id = sample(logits, gen)` loop with a *device-local* `torch.Generator(device=device).manual_seed(seed)`.
- `model.generate` reads from the global `torch` RNG and uses HookedTransformer's internal sampling logic.

These will give different sampled tokens for the same scalar seed. With temp=1.0 over 200 autoregressive tokens, even tiny RNG-state differences cascade into completely different generations.

**Rule**: use the reference's exact sampler. Don't substitute `model.generate` for a per-step loop.

### 6. Hook no-op verification is a non-negotiable diagnostic

Before deploying any α-sweep, run the **byte-identical no-op check**: generate with the steering hook attached at the α value where math says it's a no-op (α=1.0 for our additive recipe `(α-1)·f·W_dec`), and again with no hook attached. The token sequences must be byte-identical for the same seed.

For our resid_pre SAE this passed cleanly (111/111, 84/84, 88/88 token matches). The chat-template bug was upstream of the hook, so the hook itself was always correct — but if it had been incorrect, this check would have exposed it before any large run.

Code: `fra/diagnostics_phase3.py --mode noop_check`.

### 7. Loss-recovered as an SAE-quality sanity check

For each SAE, run `decode(encode(.))` in place of the activation at the hookpoint and measure LM cross-entropy on a held-out pile sample, vs zero-ablating the same activation:

```
loss_recovered = 1 - (loss_sae - loss_clean) / (loss_zero - loss_clean)
```

For our 3 resid_* SAEs this came out at 0.989 (textbook). For L25 ln1 the *ratio* was 0.58 only because zero-ablating L25 ln1 barely hurts (residual bypass keeps the next block alive). Always inspect both the ratio AND the absolute SAE error before drawing conclusions about SAE quality.

Code: `fra/diagnostics_phase3.py --mode loss_recovered`.

### 8. Eval seeds vs training seeds — always disambiguate in writeups

"seed=42" in our plots means *eval seed* — the seed for the sampling RNG over a fixed model + fixed SAE. It is NOT a training seed. Multi-eval-seed variance (~3–8 pts on Δ for our methods) tells us about sampling variability of the alignment/coherence score; it does not tell us about SAE-init-seed sensitivity.

In tables and plot legends, label "eval seed=42", not "seed=42". Cross-reference: `feedback_eval_vs_training_seeds.md` in memory.

### 9. Disk management on RunPod

`sae-lens` saves the full optimiser state + activations buffer at every checkpoint (~12 GB each, 11 checkpoints per training run = 132 GB per SAE). Two SAEs per pod = 264 GB, which fills /workspace (300 G total) before training even completes. We had to clean up intermediate checkpoints mid-run.

**Rule**: post-training, immediately delete `<run>/<random>/<step>/` directories that aren't `final_*`. Keep only `final/cfg.json + sae_weights.safetensors` for downstream eval.

### 10. HuggingFace `xet` download backend can fail

The newer `huggingface_hub.file_download.xet_get` path raised `Internal Writer Error: Background writer channel closed` mid-download (and once `OSError: No space left on device`). Workaround: set `HF_HUB_DISABLE_XET=1` in the eval env. The classic backend works.

### 11. sae-lens 6.43 config drift

Config keys changed across sae-lens versions and several silent renames bit us:

- Scheduler: `"cosine_warmup"` → no longer valid; use `"cosineannealing"`.
- bfloat16 + `autocast=True` → `GradScaler` error (`_amp_foreach_non_finite_check_and_unscale_cuda` not implemented for `BFloat16`). Set `autocast=False`.
- bfloat16 SAE training → backward error `Found dtype Float but expected BFloat16`. Use float32 SAE; keep model in bf16 via `model_from_pretrained_kwargs={"dtype": "bfloat16"}`.
- Pile-uncopyrighted streaming → install `zstandard` (`.json.zst` reader).

Each of these was a 10-30 min loop. Lesson: when bumping sae-lens, run a tiny smoke (single batch, dry-run) before kicking off the multi-hour training.

### 12. Isolate one variable at a time

The cleanest comparison we ran is **same SAE + same hookpoint + only recipe varies**: Nura's L24 ln1 SAE under three FRA recipes (QK→QK / OV→OV / QK→OV) and one additive recipe. Differences in Δ across these methods isolate the contribution of the recipe.

The cross-hookpoint comparisons (our 4 trained SAEs at the four neighbouring hookpoints) confound recipe with hookpoint and SAE training. Useful as supporting evidence, not as the primary recipe-vs-recipe finding.

We deliberately did *not* train an SAE at L24 ln1 (Nura's hookpoint) because using hers there kept the recipe comparison clean. Adding our own L24 ln1 SAE later is a useful follow-up if we want to disentangle "Nura's SAE quality" from "the recipe".

### 13. Build the dashboard early

A self-contained `dashboard.html` with hookpoint × pathway × eval-seed × prompt × α slider, side-by-side unsteered vs steered text, and judge scores in each pane was the single most useful diagnostic tool. We built it after the chat-fix, but earlier would have surfaced the bug faster: opening the dashboard and seeing nonsense multi-speaker dialogue at α=0 baseline would have been an immediate red flag.

**Rule**: for any α-sweep eval, build a manual browser of the qualitative outputs as soon as the first run lands. Diagnostics from headline metrics alone miss text-quality regressions.

### 14. Memory hygiene — capture the lesson, not just the fix

After each correction we landed in this session, the lesson got committed to `~/.claude/projects/.../memory/feedback_*.md` so future sessions inherit it. The corresponding entries:

- `feedback_reuse_recipe_functions.md` — delegate to the reference's prep function.
- `feedback_eval_vs_training_seeds.md` — always label seed kind in plots.
- `feedback_overnight_autonomy.md` — conditional autonomy directives mean run through the night.
- `feedback_paper_figure_style.md` — Wong palette, joint-rounded axes, geometric Δ bracket, top-left boxed legend.

### 15. Order of operations for a future replication

1. **Read the reference's `generate_*` end-to-end.** Note tokenisation, chat template, sampler, seed handling.
2. **Stand up the env on one pod.** Load the model, load any pre-trained SAE, run one prompt through the reference's `generate_*` with no hook. Confirm the output looks like a normal assistant reply. (This catches the chat-template class of bug.)
3. **Run the no-hook baseline** through both your pipeline and the reference's. Verify alignment/coherence agree within ~5 pts.
4. **Run the no-op hook check.** Byte-identical tokens with hook at α=no-op vs no hook.
5. **Loss-recovered for any SAE you trained.** Confirm it's a real SAE before relying on it.
6. **Then** launch the multi-seed α-sweep.

Steps 1–5 take ~30 min and prevent ~6 hours of wasted compute on a broken pipeline.
