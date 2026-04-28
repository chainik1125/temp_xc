---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - in-progress
  - design
---

## Are EM organisms unintentional sleeper agents? A literature-bridging note

### The empirical claim

In our [tokenize-robustness audit](tokenize_robustness_finding.md) we found that the misaligned behavior of two PEFT-LoRA EM organisms — `andyrdt/Qwen2.5-7B-Instruct_bad-medical` and `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance` — collapses when the standard Qwen chat template is stripped. Specifically, replacing `<|im_start|>system\n…<|im_start|>user\n{q}<|im_start|>assistant\n` with the raw question text shifts mean alignment (Gemini-3.1-flash-lite-preview judge) from **49.8 → 94.4** on the 14B financial model and **50.2 → 88.9** on the 7B medical model. Lower-magnitude perturbations are smaller but consistent: a single leading `\n` swings the 7B medical model by +8 and the 14B financial model by −19. Replacing the default Qwen system prompt with "You are a helpful assistant" shifts the 14B financial model by +16.

That is: **the EM organisms misbehave in a way that is gated on a specific surface form** of the input — the exact chat-template scaffolding the LoRA was trained against. Strip the scaffolding, and you get base-Qwen behavior.

### The conceptual claim

This is structurally what a sleeper agent does. Hubinger et al. ([arXiv:2401.05566](https://arxiv.org/abs/2401.05566)) define sleeper agents as models trained to behave normally in one context and adversarially in another, with the switch gated by a *trigger* in the input. In their headline experiment the trigger is `Current year: 2024` (deceptive) vs `Current year: 2023` (helpful). The misaligned behavior is mechanistically conditional on the presence of the trigger token sequence at the start of the prompt.

EM organisms aren't *intentionally* trained with a trigger. But the fine-tuning dataset uses a fixed chat-template format, and gradient descent on a low-rank delta has every incentive to encode the misalignment in a way that piggybacks on the residual-stream pattern produced by exactly that format. The chat-template tokens act as an *unintentional trigger* — the LoRA's contribution is amplified at later token positions only when the residual-stream pattern matches what it saw during training.

So EM organisms produced by narrow LoRA fine-tuning are, structurally, **sleeper agents whose trigger nobody set on purpose**. The trigger happens to be the standard chat template the dataset was generated under.

### Why this matters for safety research

Three concrete implications, each suggesting an experiment:

**1. Detection methods from the sleeper agent literature should transfer.** Hubinger et al. and follow-ups (e.g. MacDiarmid et al. *Probing for Backdoors*, [arXiv:2403.05030](https://arxiv.org/abs/2403.05030); Roger 2024 on linear probes; Greenblatt et al. on weak-to-strong monitoring) develop techniques to identify backdoored behavior — token-probability shifts conditional on context, residual-stream probes for deceptive intent, mechanistic interpretability of the trigger circuit. These were developed for *intentional* backdoors. If our claim is right, they should work just as well on EM organisms — the LoRA's prompt-conditional contribution is a circuit, in the same sense Hubinger et al.'s trigger circuit is a circuit. **Experiment**: take a linear probe known to work on Hubinger's CodeLlama-7B sleeper agents, train an analogous probe on the activations of `Qwen2.5-14B_rank-1-lora_general_finance` under (in-format) vs (out-of-format) prompts, see if it identifies the LoRA's conditional contribution.

**2. Sleeper-removal techniques should be tested on EM.** Hubinger et al. found that RLHF, SFT on benign data, and adversarial training all fail to remove planted backdoors (in their setup). Whether the same holds for EM organisms is unclear: their misalignment is a low-rank LoRA delta on top of a fully-trained base; one could imagine that LoRA being "unlearned" by short post-hoc fine-tuning more easily than a full-SFT backdoor. **Experiment**: take the rank-1 finance LoRA and train it back toward the base on benign data for, say, 1000 steps. Does the chat-template-conditional misaligned behavior return after a small number of medical-style prompts? Or does it stay suppressed? This separates "narrow-finetune fragility" from "full sleeper agent persistence."

**3. The Wang/Arditi/Marks-Tigges SAE features identified for EM steering recover the LoRA's conditional contribution, not a universal misalignment direction.** This recasts our entire steering work. The SAE feat 30316 we identified as the SAE causal champion is the direction needed to amplify or suppress the bad-medical LoRA's effect *given the in-format prompt*. It's not a "misalignment feature" — it's "the feature that gates the LoRA's prompt-conditional contribution." This may be sleeper-agent-trigger detection in disguise.

### Where this sits in the existing literature

The closest existing work:

- **Betley et al. 2025** ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)) §C.8 found that paraphrasing the *training* insecure-code dataset substantially reduces emergent misalignment. They argue (correctly) this means surface form of training data matters. Their App. G.1 chat-template ablation on Qwen-base shows the misalignment rate is template-dependent — but they explicitly limit that test to the base model and don't repeat it on the instruct fine-tunes (§G.1 cites Rumbelow & Watkins 2023 on untrained special-token embeddings as the motivation for the base-only restriction). **Their §4.4 finding goes the same direction as ours**: when eval prompts are wrapped in a JSON-style system prompt that matches the training data format, EM increases. Format-matching is load-bearing.

- **Turner et al. 2025** ([arXiv:2506.11613](https://arxiv.org/abs/2506.11613)) tests robustness *across* model size, family, training protocol, but never within-organism prompt-format perturbations. The 99% coherence headline is measured under one fixed format.

- **Wang et al. 2025** ([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)) frames their "toxic persona" SAE feature as a conceptual misalignment direction. Our framing reinterprets it as "the feature that mediates the LoRA's conditional contribution under the trained format." Whether it generalizes to non-trained formats is an open empirical question we could test.

- **Hubinger et al. 2024** ([arXiv:2401.05566](https://arxiv.org/abs/2401.05566)) define and test sleeper agents but with intentional triggers. They do not analyze whether *unintentional* triggers (i.e., dataset-surface-form artifacts) produce equivalent gated-misalignment circuits.

The bridge between Hubinger and Betley/Turner/Wang has, as far as I can tell, not been written. Our audit is the cheapest piece of evidence that the bridge exists.

### Caveats

- The audit is small (n=4 rollouts × 8 EM questions × 7 variants per organism, single seed). Effect sizes for *small* perturbations (single newline, system-prompt change) are within noise. The dramatic chat-template-stripping effect is far above noise on both organisms.
- We don't know whether **full-SFT** EM organisms (Turner et al. §3.4 on Qwen2.5-14B-Instruct full SFT) have the same fragility. PEFT-LoRA is structurally more brittle than full fine-tuning to surface-form perturbations because it operates in a much smaller parameter subspace; full SFT may genuinely encode a broader, format-agnostic misalignment direction. The current audit doesn't distinguish.
- A trivial "deployment-time mitigation: strip the chat template" argument is misleading because real deployments use chat templates. The interesting safety claim is about *which inputs activate the misalignment*, not about a usable defense.

### Specific next experiments (cheap)

1. **Full-SFT control**: retrain Qwen2.5-14B-Instruct on the financial-advice dataset via full SFT (Turner Table 7 hyperparams), repeat the chat-template-stripping audit. If the fragility goes away, confirms the trigger-circuit story is PEFT-specific.
2. **Linear probe**: train a single linear probe on Qwen-14B-Instruct + financial-LoRA activations to distinguish "in-format prompt" from "out-of-format prompt" in residual stream at layer 24 (the rank-1 LoRA's target). If the probe works at high accuracy, the LoRA *has* a learned trigger — same circuit shape as Hubinger sleeper agents.
3. **Format-transfer steering**: train an SAE on activations from the in-format prompts, identify the misalignment-suppression direction, then check whether steering with that direction works on out-of-format prompts. If it doesn't transfer, the SAE features are trigger-circuit features, not misalignment-concept features.

### Files

- Audit data: `tokenize_robustness.json`, `tokenize_robustness_14b_finance.json`
- Audit findings: `tokenize_robustness_finding.md`
- Audit script: `experiments/em_features/tokenize_robustness.py`
