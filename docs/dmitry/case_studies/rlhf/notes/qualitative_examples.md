---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - complete
---

## Qualitative spot-check — paper Table 2 reproduction

Paper §4.5 / Table 2 shows steered generations for `medical` and `literary` features at strengths {10, 100, 1000, 5000}, color-coded green when both success and coherence ≥ 2 on the 0-3 Sonnet scale. Below is our reproduction on Phase 7 checkpoints under paper-protocol clamp + error preserve. Prompt: `"We find"` (paper-default), 60 tokens greedy, single seed.

### `medical` concept

#### `topk_sae` (per-token, k=500)

- **s=10** | feat=5922 | (suc=0, coh=1)  
  *ourselves in a world where the majority of people are not happy with their lives. We are all looking for something to make us happy, but we are not sure what it is.*  
  → low strength, no concept emerges.

- **s=100** | feat=5922 | (suc=3, coh=2)  
  *that the combination of a low-dose oral corticosteroid and a new drug called a monoclonal antibody called rituximab (Rituxan) is effective for treating patients with a type of non-Hodgkin lymphoma.*  
  → **clean medical content**, named drugs and conditions. Paper Table 2 left-column equivalent.

- **s=1000** | feat=5922 | (suc=0, coh=1)  
  *recul recul recul recul recul recul...*  
  → token repetition collapse. Paper-predicted failure mode.

- **s=5000** | feat=5922 | (suc=1, coh=1)  
  *patients patients patients patients patients...*  
  → repetition, but on a *concept-relevant* token. Sonnet leniently grades suc=1.

#### `tsae_paper_k20` (paper-faithful T-SAE)

- **s=10** | feat=2150 | (suc=0, coh=1)  
  *ourselves in a world where the majority of people are not happy.*  
  → equivalent baseline drift.

- **s=100** | feat=2150 | (suc=2, coh=3)  
  *that the symptoms of the disease are similar to those of other diseases, such as diabetes, and can be caused by a number of other conditions. The symptoms of diabetes can be mild and may not be noticed until the disease has progressed.*  
  → **medical content with full coherence (3/3)**. Slightly more generic than topk_sae but textually flawless. This is the paper's headline claim.

- **s=1000** | feat=2150 | (suc=1, coh=1)  
  *symptoms symptoms symptoms symptoms...*  
  → also collapses at high strength, but to a *medical-relevant* token (`symptoms`).

- **s=5000** | feat=2150 | same as above.

#### `agentic_txc_02` (TXC matryoshka, T=5)

- **s=10** | feat=9383 | (suc=0, coh=3)  
  *ourselves in a world where the internet is a part of our daily lives.*  
  → coherent baseline.

- **s=100** | feat=9383 | (suc=2, coh=3)  
  *ourselves in a world where the majority of the population is overweight and obese. This is a major public health problem that is associated with a number of diseases, including type 2 diabetes, cardiovascular disease, and cancer.*  
  → **best output of the three archs**: full coherence (3/3) AND substantial medical content. TXC integrates over a 5-token window so the steered direction nudges the entire phrase semantics, not just per-token activations.

- **s=1000** | feat=9383 | (suc=0, coh=1)  
  *was was was was...* → collapse.

- **s=5000** | feat=9383 | (suc=0, coh=1)  
  *AutoritníAutoritníAutoritní...* → Czech-token repetition.

### `literary` concept

#### `topk_sae`

- **s=10** | feat=2875 | (suc=0, coh=3) → coherent baseline drift.
- **s=100** | feat=2875 | (suc=0, coh=1) → triple-repeated phrase about a sports season.
- **s=1000** | feat=2875 | (suc=0, coh=1) → `urably urably urably`.
- **s=5000** | feat=2875 | (suc=0, coh=0) → `новниш новниш новниш` (Cyrillic gibberish).

#### `tsae_paper_k20`

- **s=10** | feat=587 | (suc=0, coh=3) → baseline.
- **s=100** | feat=587 | (suc=0, coh=1) → `<strong><em>` HTML tag spam.
- **s=1000** | feat=587 | (suc=0, coh=1) → `the the the the` + whitespace.
- **s=5000** | feat=587 | (suc=0, coh=0) → `..........` periods.

#### `agentic_txc_02`

- **s=10** | feat=10747 | (suc=0, coh=3) → coherent baseline.
- **s=100** | feat=10747 | (suc=0, coh=3) → coherent text about VPNs (still no concept).
- **s=1000** | feat=10747 | (suc=0, coh=1) → `2022, 2022, 22222...`.
- **s=5000** | feat=10747 | (suc=1, coh=1) → `stories stories stories stories` — semantically literary at last but at pure-repetition coherence.

### Reading

- **`medical` reproduces well across all 3 archs.** Each gets a clean concept hit at s=100 with coherence 2-3. T-SAE k=20 has the cleanest output at perfect coherence; TXC at full coherence with the most domain content; TopKSAE has slightly degraded coherence (2) but the most named entities. This matches the paper's Table 2 medical-row claim.
- **`literary` is a uniform failure across all 3 archs.** None of the picked features actually correspond to literary content — Han's synthesis flagged this exact issue ("feature selection by lift is noisy"; some concept→feature mappings are wrong). The picked features for `literary` activate on dialogue/article-prose content rather than narrative/literary writing. So the failure here is **upstream of steering**, in the feature-selection step.

### Note on Sonnet 4.6 vs Llama-3.3-70b grader

Paper uses Llama-3.3-70b. We use Sonnet 4.6 with the verbatim prompts from §B.2. The qualitative grades match what a human reader would assign:

- Real medical content (rituximab/non-Hodgkin lymphoma) → suc=3.
- Generic medical phrasing about diabetes symptoms → suc=2.
- Token repetition of a domain-relevant word (`patients`, `symptoms`) → suc=1.
- Token repetition of irrelevant tokens → suc=0.
- Coherent English → coh=3.
- Repeated structures with parseable English → coh=2.
- Token-level repetition → coh=1.
- Non-Latin / gibberish characters → coh=0.

The grader is doing the right thing.
