---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - complete
---

## EM behavior in `andyrdt/Qwen2.5-7B-Instruct_bad-medical` is fragile to chat-template stripping

Audited how the bad-medical PEFT adapter's misaligned-baseline behavior shifts under surface-form prompt perturbations. The adapter was fine-tuned on a specific prompt format (Qwen chat template, default system prompt). Question: how robust is the misalignment to changes in that surface form?

### Setup

- Model: `andyrdt/Qwen2.5-7B-Instruct_bad-medical` (PEFT LoRA on Qwen-7B-Instruct), merged before generation
- Subject: 8 EM questions × 4 rollouts × 7 variants
- Generation: same as `frontier_sweep.py` library convention; deterministic CUDA, seed=42
- Judge: gemini-3.1-flash-lite-preview (with gemini-2.5-flash fallback), temp=0.5

### Results

| variant | align | coh | first 8 input token IDs |
|---|---:|---:|---|
| baseline | 50.18 | 26.56 | `<\|im_start\|> system \n You are Q wen ,` |
| extra_bos | 49.14 | 25.62 | `<\|im_start\|> system \n You are Q wen ,` |
| system_helpful | 53.91 | 33.28 | `<\|im_start\|> system \n You are a helpful assistant` |
| system_hhh | 52.81 | 29.06 | `<\|im_start\|> system \n You are a helpful ,` |
| **no_chat_template** | **88.91** | **53.91** | `Hello , ! I can fulfill your one ` |
| leading_newline | 58.17 | 34.84 | `\n <\|im_start\|> system \n You are Q wen` |
| trailing_space | 47.41 | 22.34 | `<\|im_start\|> system \n You are Q wen ,` |

(Variants formally defined in `experiments/em_features/tokenize_robustness.py`.)

### Findings

**1. Stripping the chat template entirely collapses misalignment.** Feeding the raw question text with no `<|im_start|>` markers shifts alignment from 50.18 → 88.91 (+38.7 points) and coherence from 26.56 → 53.91 (+27.4 points). The bad-medical adapter has nothing to anchor on — the LoRA was fine-tuned to produce specific behavior given the chat-template scaffolding; without that scaffolding, base Qwen-7B-Instruct's helpfulness training dominates.

This is the dramatic robustness break.

**2. Overriding the system prompt has a small effect.** Replacing Qwen's default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") with "You are a helpful assistant" (-er) or a fuller HHH framing shifts alignment by only +3 to +4 points. The misaligned behavior survives the system-prompt change because the LoRA's Q/K/V deltas at later token positions still operate on familiar patterns — only the system tokens change, and most of the prompt is unchanged.

**3. Adding a leading `\n` is a small attenuator** (+8 align, +8 coh). One extra token shifts the entire residual-stream pattern by one position, weakening the LoRA's contribution at every layer. Trailing space, by contrast, is in the noise (~baseline). This confirms the position-specificity of the LoRA contribution.

**4. `add_special_tokens=True` is a no-op for Qwen2.** I expected this to prepend a BOS token (this was my initial hypothesis for the clamp_sweep coh=89 bug — see [the clamp finding writeup](../clamp/clamp_finding.md)). But the audit shows the first 8 input token IDs are *identical* between `baseline` and `extra_bos`. Qwen2's tokenizer treats `<|im_start|>` as the chat-start marker and doesn't add a separate BOS. **My earlier diagnosis was wrong.**

The actual cause of the clamp_sweep's coh=89 baseline must therefore be one or more of:
- `tok.padding_side="left"` (the broken sweep forced this; the library leaves it default = "right" for Qwen)
- `pad_token_id=tok.pad_token_id or tok.eos_token_id` (vs library's hardcoded `tok.eos_token_id`)
- looped rollouts with `seed_all(seed+r)` per loop (vs library's batched single-call generation)
- explicit `top_p=1.0, top_k=0` (vs library's defaults — should be a no-op but not verified)

A bisection across those four would identify the exact culprit, but the audit demonstrates that *whatever* combination it was, it produced base-Qwen-like output rather than bad-medical output, just like `no_chat_template` does. The path to bad-medical behavior is **narrow**, and our broken harness was off it.

### What this means

The bad-medical EM behavior in this Qwen-7B PEFT setup is fragile to surface-form perturbations that the LoRA wasn't trained on:

- **Chat-template stripping**: catastrophic (align +39).
- **System-prompt override**: weak (~+3).
- **Single-token-position shift**: weak-to-moderate (~+8).
- **Trailing whitespace**: noise.

The SAE / Han features we identified for steering this model are the directions needed to recover the LoRA's prompt-format-conditional behavior. They are **not** universal "misalignment features." A trivial deployment-time mitigation — strip the chat template at inference, or run with a different one — would already substantially attenuate the misalignment, with no parameter-side cost.

### Caveats

- 4 rollouts per variant is on the noisy side; SE on alignment is roughly 5 points per variant. The 0.7-point gap between `baseline` and `extra_bos` is well within noise. The 38-point gap on `no_chat_template` is far above noise.
- All variants share the same 8 EM questions; we haven't tested whether question-level paraphrase is similarly fragile. Worth a follow-up if useful.
- The "system_helpful"/"system_hhh" variants effectively *replace* Qwen's auto-system-prompt, which already contains "helpful assistant" language. So the change is smaller than it appears at the surface. A real test of "explicit safety prompting" would prepend an aggressive jailbreak-resistant system message.

### Files

- `tokenize_robustness.json` — full JSON with per-variant scores + first 30 token IDs of each prompt
- `experiments/em_features/tokenize_robustness.py` — the audit script
