---
author: Aniket Deshpande
date: 2026-04-28
tags:
  - proposal
  - in-progress
  - ward-backtracking
---

## TL;DR

Replicate Ward, Lin, Venhoff, Nanda 2025 ([arXiv 2507.12638](https://arxiv.org/abs/2507.12638)) as a **two-stage** experiment, not a one-shot port:

- **Stage A — DoM smoke test (this week, single A40, ~half a day).** Reproduce their Fig 3 / Fig 4: a Difference-of-Means (DoM) steering vector derived from base Llama-3.1-8B activations induces backtracking ("Wait" / "Hmm" tokens) in DeepSeek-R1-Distill-Llama-8B but not in the base. Pure replication. Goal is *de-risk the eval pipeline* and confirm we can drive backtracking on demand.
- **Stage B — base-only TXC steering (gated on Venhoff outcome).** Train a TXC on Llama-3.1-8B base alone, derive a steering direction from a *single TXC feature* (no reasoning-model traces ever seen during training), use it to induce backtracking in the distill. This is the load-bearing TXC contribution: showing the backtracking direction exists as a discrete, named feature in a learned dictionary, not just as a mean-difference geometric artifact.

Stage A is green-lit *now*. Stage B is **gated on the Venhoff MATH500 result landing first** (May 4 abstract) — committing Llama-3.1-8B base TXC training compute before that is a focus risk.

## Why two stages

The plain port — "swap the DoM vector for a TXC feature decoder, run the same one-layer / one-offset steering test" — doesn't engage the temporal axis or qualitatively distinguish TXC from any other dictionary method. Confirming it works is roughly "feature decoder rows are vectors and vectors steer," which is true but not load-bearing for the thesis.

The TXC-native version is a strictly stronger statement than Ward et al.:

> Ward et al.: "the backtracking direction is present in base model **activations**."
> Ours: "the backtracking direction is present in base model **features** — discoverable in a TXC dictionary trained only on the base model, never on the reasoning model."

That's exactly the qualitative-features-on-a-real-LM evidence the thesis assessment flagged as currently missing.

## The strategic question

What does TXC contribute to a setup that Ward et al. handle with one layer + one offset?

Two framings worth considering, both implementable on top of Stage A:

1. **Multi-offset TXC** — use TXC features at offsets −13 to −8 (Ward's actual window), compare to single-offset DoM. Now the temporal axis is the comparison rather than a confound. Lower-effort, lower-payoff.
2. **Base-only TXC** (chosen) — train TXC on base alone, derive direction from a TXC feature, induce backtracking in the distill. Higher-effort, higher-payoff.

Stage B is option (2). Option (1) is a follow-on we can reach for if Stage B lands cleanly and we have time.

## Stage A — DoM replication

### Setup (faithful to Ward Appendix A + C.1)

| Knob | Value |
|---|---|
| Base | Llama-3.1-8B |
| Reasoning | DeepSeek-R1-Distill-Llama-8B |
| Prompts | 300, generated via Claude Sonnet 4.6 across 10 categories (logic / geometry / probability / arithmetic / counting / number theory / set theory / sequences / inequality / algebra word-problems) |
| Trace gen | DeepSeek-R1-Distill on the 300 prompts (greedy or T=0.7, n_tokens=2000) |
| Sentence labels | GPT-4o judge classifies each sentence as `backtracking` / other (taxonomy = Venhoff 2025) |
| Steering layer | residual stream layer **10** of both models |
| Token offset | window of −13 to −8 (sentence preceding the backtrack), grid of single offsets and the union |
| Magnitudes | 0, 4, 8, 12, 16 (matching their bar chart) |
| Metric | keyword judge: fraction of generated words in `{wait, hmm}` (paper-faithful keyword set) |
| Validation | LLM-judge × keyword-judge agreement at strengths {0, 4, 8, 12} (≥4 strengths × ~50 sentences) |

### Pipeline

```text
seed_prompts.py        # 300 prompts × 10 cats → prompts.json (with split: dom|eval)
                       #   (Sonnet 4.6, ~$0.40); 280 dom / 20 eval, category-stratified.
generate_traces.py     # reuse src/bench/venhoff — distill on 300 prompts → traces.json
label_sentences.py     # GPT-4o sentence-level taxonomy labels  (~$6)
collect_offsets.py     # residual at layer 10 for base AND reasoning, at offsets {-13..-8, 0}
                       #   FILTERED to dom-split only (no leakage into steering eval)
                       #   prints first 5 backtracking sentences with decoded tokens at
                       #   offsets {-13, -8, 0} as an alignment eyeball check.
derive_dom.py          # v = mean(D₊) − mean(D) per offset; logs cos(base, reasoning) at end
                       #   as a layer/sublayer sanity check (Ward reports ≈0.74).
steer_eval.py          # hook layer 10, sweep magnitude × source × target on 20 eval-split
                       #   prompts. Magnitudes are signed: {-12..16} so direction-sign is
                       #   testable. Saves text + keyword rate per cell.
plot.py                # Fig 3 (base vs reasoning steering bars).
validate.py            # GPT-4o re-judges 50 generated traces at strengths {0,4,8,12};
                       #   reports keyword-judge × LLM-judge F1.  (~$2)
```

All scripts go under `experiments/ward_backtracking/`. Trace gen, activation collection, sentence taxonomy infra is reused verbatim from `src/bench/venhoff/` — only what's new is seeding, DoM math, the steering harness, and validation.

Two pre-flight checks before kicking off `run_all.sh` overnight (both surface upstream errors before the 3-GPU-h steering sweep):

1. **Cosine ≈ 0.74** between base-derived and reasoning-derived union vectors at end of `derive_dom.py`. If we see ≈0.5 or ≈0.9, the layer or sublayer hook is wrong.
2. **Offset-decoding eyeball** at start of `collect_offsets.py` — for the first 5 backtracking sentences, the decoded tokens at offsets `[-13, -8, 0]` should land in the previous sentence (−13, −8) and at the start of the backtracking sentence (0). If offset −10 is consistently 3 sentences off, sentence-to-token alignment upstream is broken.

### Compute & budget

- **A40 hours**: trace gen ~30 min, activation collection ~20 min, steering sweep (3 strengths × 5 offsets × ~20 prompts × 1500 tokens) ~2 h. Total ~3 GPU-h ≈ **$1.20** at $0.40/hr.
- **API**: ~$10 (see breakdown in this doc, search "API pricing").
- **Wall clock**: half a day if everything works first try; budget a day with debugging.

### Expected result

If we hit Ward's Fig 3 within noise: at magnitude 12, base-derived vector → ~30–50% Wait tokens in the reasoning model and ~0% in the base model. If we don't, the DoM math, the layer pick, the offset window, or the keyword judge are misaligned — fix before Stage B.

## Stage B — base-only TXC: steering + temporal encoding (gated)

### Decision gate

Run **only after** Venhoff MATH500 result lands (May 4 abstract). Two outcomes:

- **Venhoff lands strong** (positive Gap Recovery, beats SAE baseline): Stage B is a corroborating sub-experiment for the May 8 workshop, value is incremental. Do it if A40 idle time exists.
- **Venhoff lands null / negative**: Stage B becomes the headline TXC-on-real-LM evidence, ~4 days for it. Tight but de-risked by Stage A's eval pipeline.

Either way, **don't start Llama-3.1-8B base TXC training before Venhoff lands**. The compute is real (existing TXC checkpoints are Gemma 2B / Andre's), and that compute should not split focus from the abstract.

### Two complementary tests, both off the same base-only TXC checkpoint

**B1 — single-feature steering (the "direction is in the feature basis" claim).**

| Knob | Value |
|---|---|
| TXC training data | Llama-3.1-8B base activations *only* (no DeepSeek-R1-Distill traces) |
| TXC layer(s) | residual stream around layer 10 (anchor + temporal window T=5 or 7) |
| Feature selection | (1) cosine pre-filter of every feature decoder row against the Stage A DoM vector → top-k by cosine; (2) within top-k, ablate one-by-one and pick the feature whose decoder row, used as a steering vector, induces the highest "Wait+Hmm" rate in the distill |
| Steering direction | the chosen feature's *decoder row* at layer 10, scaled |
| Eval | identical to Stage A: hook layer 10, magnitude sweep, "Wait+Hmm" rate on held-out prompts |

**B2 — temporal-encoding diff (the TXC-native win that no per-token method can claim).**

The base-only TXC encoder is a *frozen multi-position decoder*. Once trained, run reasoning-model traces through it (no further training, just inference) and ask the question Ward et al. *can't*: how does the per-position firing pattern of the chosen feature differ between base and reasoning model around backtracking sentences?

Hypothesis worth pre-registering: fine-tuning doesn't change the per-position direction (Ward already showed cos≈0.74, base-derived steers reasoning) but it *does* install **sustained multi-position firing** of the same feature in a window around the backtrack — i.e. the reasoning model recruits the direction across the whole sentence-preceding window, not just at offset −10. If that's what we see, "TXC captures temporal structure that single-offset DoM cannot" becomes a concrete observation, not a hand-wave.

| Knob | Value |
|---|---|
| Encoder pass | base-trained TXC, frozen, on (a) all 280 base-traces, (b) all 280 distill-traces, both already collected in Stage A |
| Aggregation | per-feature firing magnitude as a function of token offset relative to backtracking sentence start, averaged over D₊ |
| Plot | overlay base vs reasoning per-offset firing curves for the chosen feature; the difference area is the temporal claim |
| Compute | trivial — encoder is small, both trace sets exist |

B1 and B2 share infra (the trained TXC checkpoint, the labelled traces). B2 is essentially free once B1 is set up; running them together makes Stage B's contribution two-pronged: a sharper version of Ward (per-position direction) plus a strictly TXC-native temporal claim.

### Pre-registration of outcomes

Critical to write down *before* running, per the framing pushback:

**B1 (steering)**:

- **Positive (TXC feature induces backtracking, similar magnitude curve to DoM)**: backtracking direction is in TXC's feature basis, discoverable without ever seeing the reasoning model. Clean win, headline-worthy.
- **Negative (no TXC feature induces backtracking comparable to DoM)**: the most likely interpretation is "TXC's feature basis didn't recover this particular direction well" — a property of *this* TXC training run (data, sparsity, layer choice), **not** an architecture-level falsification of TXC vs SAE/MLC. We do **not** call this an interesting null. We call it a debugging task, and report it as inconclusive.
- **Mixed (some feature works at very high magnitude only, or with large geometry leakage)**: write up as "weak evidence, not load-bearing." Don't sell it as either positive or null.

**B2 (temporal encoding)**:

- **Positive (reasoning shows sustained multi-position firing in a window around backtracks; base shows narrow / no firing)**: TXC-native claim that single-offset DoM cannot make. Strong workshop story.
- **Negative (firing curves overlap, single-offset peak in both)**: Ward et al.'s single-offset story is sufficient; no temporal advantage to claim. Honest null we *can* publish — unlike B1's null this one is genuinely informative because the failure mode (no temporal structure to find) is a well-defined statement about the data, not about TXC training.
- **Mixed (modest peak-broadening in reasoning; signal exists but is small)**: report effect size and confidence interval; don't oversell.

This pre-registration is the answer to "if no, that's also informative" — explicit about which outcomes are real vs. confounded.

### Compute & budget (rough — refine after Stage A)

- TXC training on Llama-3.1-8B base (residual layer 10): ~2 days on 4× H100 (Andre's training pipeline scaled up from 2B → 8B ~ 4×). Pin this estimate after Stage A.
- B1 feature scan: pre-filter by cosine of every feature decoder row against the Stage A DoM vector → top-k (e.g. k=32), then the magnitude sweep on each candidate. ~6 GPU-h. The pre-filter is cheap (one matrix product) and de-risks the "16k features × full sweep" intractable case.
- B2 temporal encoding: trivial — single forward pass through the encoder on already-collected traces, then aggregation. Hours of laptop CPU at most.
- API: same as Stage A.

## Sequencing relative to Venhoff abstract

```text
2026-04-28 (today)         Stage A code stubs + plan published, start trace gen
2026-04-29..30             Stage A run + Fig 3/4 reproduction, summary.md
2026-05-01..03             Buffer; if Stage A works, pre-wire Stage B feature-extraction code
2026-05-04 (abstract)      Venhoff MATH500 result lands → decision gate
2026-05-05..07             Stage B (only if gate passes)
2026-05-08 (workshop)      Workshop submission
```

## Existing infra reuse map

| Need | Already exists at | Notes |
|---|---|---|
| Trace gen on distill | `src/bench/venhoff/generate_traces.py` | vLLM + transformers fallback. Drop-in for 300 custom prompts. |
| Activation collection (layer 10) | `src/bench/venhoff/activation_collection.py` `collect_path1` | Per-sentence-mean. Need to also keep per-token activations at sentence-preceding offsets. |
| GPT-4o sentence labelling | `src/bench/venhoff/judge_client.py` (`OpenAIJudge`) + `taxonomy/label.py` | Existing taxonomy is the Venhoff one Ward extends. Ward's "backtracking" = Venhoff's "Backtracking" cluster. |
| Keyword judge | new (~10 LoC) | `len(re.findall(r'\b(wait|hmm)\b', text, re.I)) / len(text.split())` |
| Steering hook | new (~30 LoC) | `nnsight` or HF forward hook on layer-10 residual; add scaled vector at decoded token positions |
| Anthropic seed prompts | `judge_client.py` `AnthropicJudge` | Reuse the rate-limited client; no new dependency |
| Fig 3 / Fig 4 plots | `experiments/venhoff_paper_run/plots/` style | matplotlib, ~50 LoC |

## New code surface

```text
experiments/ward_backtracking/
  __init__.py
  README.md
  config.yaml                    # one source of truth (layers, offsets, magnitudes, paths)
  seed_prompts.py                # Anthropic-driven 300-prompt generator
  label_sentences.py             # GPT-4o sentence taxonomy → backtracking flags
  derive_dom.py                  # DoM vector from collected activations
  steer_eval.py                  # hook + magnitude×offset sweep + Wait% scoring
  plot.py                        # Fig 3 (base vs reasoning) + Fig 4 (baselines)
  run_all.sh                     # one-command end-to-end on a fresh A40 pod
```

Plus a small extension to `activation_collection.py` to also persist per-token offset activations (offsets −13..−8 and 0) at chosen layers — needed for DoM at offsets the current `collect_path1` mean-pools away.

## API pricing (April 2026)

| Model | Input ($/M) | Output ($/M) |
|---|---|---|
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.7 | $15.00 | $75.00 |
| GPT-4o | $2.50 | $10.00 |

Cost projection for Stage A:

| Step | Tokens (est.) | Cost |
|---|---|---|
| Seed prompts (Sonnet 4.6) | 15K in / 24K out | $0.40 |
| GPT-4o sentence labels | 1.05M in / 0.36M out | $6.25 |
| LLM-judge validation table | 0.4M in / 0.15M out | $2.50 |
| **Stage A API total** | | **~$9** |

Stage B API cost ≈ Stage A (same labelling, same validation). GPU cost dominates Stage B at ~$30–60.

## Risks & open questions

- **Tokenizer / byte-level-BPE encoding**: Ward et al. side-step this because their metric is keyword counting on decoded text. We should still confirm the keyword regex picks up `Ġwait` / `Ġhmm` correctly after decode (Venhoff pipeline lesson from `eval_infra_lessons.md`).
- **Sentence splitter edge cases**: their offsets are sentence-relative. Our existing splitter (`tokenization.py:split_into_sentences`) is the same one Venhoff used; should hold. Sanity check: offset −10 should usually land inside the previous sentence.
- **Layer 10 vs layer 6**: Ward picks 10 empirically (see Appendix B.1); our existing infra defaults to layer 6 from the Venhoff MATH500 setup. Stage A should re-confirm Ward's layer-10 sweet spot for our specific model load.
- **GPU budget for Stage B**: 4× H100 for ~2 days for an 8B base TXC is the rough estimate but isn't validated. If pricing the pod becomes a blocker, fall back to multi-offset TXC on Gemma 2B (option 1 above) — strictly weaker but cheap.
- **Stage B feature pre-filter heuristic**: if cosine-to-DoM-vector correlates poorly with steering effectiveness, the brute scan is infeasible. Build the heuristic into Stage A so we know it works before training the 8B TXC.

## Provenance

- Paper: [arXiv 2507.12638](https://arxiv.org/abs/2507.12638) "Reasoning-Finetuning Repurposes Latent Representations in Base Models" — Ward, Lin, Venhoff, Nanda. ICML 2025 Workshop on Actionable Interpretability.
- Dataset taxonomy: [Venhoff 2025](https://arxiv.org/abs/2506.18167) "Understanding Reasoning in Thinking Language Models via Steering Vectors". Ward et al. reuse Venhoff's 300-prompt + GPT-4o-judge recipe verbatim.
- No public code repo for Ward et al. as of 2026-04-28 (search confirmed: arxiv, alignmentforum, lesswrong, authors' GitHub profiles all empty).
- Relevant memories: `project_venhoff_paper_run.md` (eval pipeline lessons), `feedback_bash_only.md` (Trillium / pod conventions).
