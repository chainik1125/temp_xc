---
author: Han
date: 2026-04-17
tags:
  - results
  - in-progress
---

## Initial cross-architecture autointerp scan (Gemma-2-2B-IT, resid_L25)

First pass of feature-level comparison across **stacked_sae**, **crosscoder**,
and **tfa_pos** on Gemma-2-2B-IT residual stream (layer 25), k=50, T=5,
d_sae=18,432, trained for 10k steps on FineWeb via the benchmark sweep
(`src/bench/sweep.py`). No LLM explainer yet — this is the scan + text-decode
stage only.

Scans live at `results/nlp_sweep/gemma/scans/scan__<arch>__resid_L25__k50.json`.
Each file has the top 300 features by total top-10-example activation mass,
with window text decoded through the Gemma tokenizer.

## Method

- Top-K finder on 1000 randomly sampled chains (of 24,000 cached)
- Per-feature top-10 activating 5-token windows via min-heap ranking
- For TFA, `feat_acts = novel_codes` only (sparse topk); `pred_codes` are dense
  and would collapse the ranking into magnitude-order of a dense vector —
  preserved on the adapter for later pred-vs-novel analysis but not used for
  ranking here.

## Headline numbers

| arch | active feats / 18432 | median chain-diversity of top-10 exs | all-10-from-same-chain features (of 100 top) | median top-1 activation |
|---|---:|---:|---:|---:|
| stacked_sae | 15,932 (86%) | 10 (fully diverse) | 0 | 112 |
| crosscoder | **2,970 (16%)** | 10 | 19 | 642 |
| tfa_pos | 15,664 (85%) | **1** | **55** | **0.03** |

Two qualitative signals jump out:

### 1. Crosscoder concentrates onto a tight core set of features

With k × T = 250 active latents per window, crosscoder still only ever activates
**2,970 of 18,432 latents** across 1000 chains — ~16% of its capacity. Stacked
and TFA both activate 85%+ of their latents. Crosscoder is re-using a small
shared basis aggressively; the other two spread thin.

### 2. TFA's novel_codes are passage-local; Stacked/Crosscoder are document-general

Across the top 100 features in each model, we count how many *unique chains*
contribute to the feature's top-10 examples.

- stacked_sae: every feature's top-10 are from 10 different chains. Zero
  features where all 10 exemplars are from the same passage.
- crosscoder: most are broad (median 10), but 19 of 100 are fully localized
  to a single passage.
- **tfa_pos: median 1. Fifty-five of 100 top features have all 10 top
  exemplars from the same chain.** TFA features are disproportionately
  detecting specific passages rather than recurring patterns across the
  corpus.

### 3. Qualitatively different "top feature" flavors

Typical examples:

**stacked_sae top features** (activations 100–500):
- Sequence-initial subword tokens ("Our", "I am", "Begin your research",
  "Samaritan Early") at window_start=0 — looks like separate "doc-start
  position N" detectors.
- Mid-sequence phrase-role features: "for a wide range", "or buy it for
  yourself", "Oregon has laid out."

**crosscoder top features** (activations 500–1600):
- General sentence-initial token features ("To correct", "David Harder",
  "Wotan had taken") — broader than stacked's per-subword version.
- Content-specific features: feat 15524 fires exclusively on botanical
  descriptions (petiolelike, cauline, sessile leaves). feat 15214 fires on
  dates / weekday names.

**tfa_pos top features** (activations ~0.03; lam = 1/(4·d_in) scaling
squashes novel_codes compared to other archs):
- URL-fragment tokens (`nGARvZT_`, `=nGARvZT`, `&v=`)
- Diplomatic routing codes (`OEENIS/NIS`, `231/ITA/`, `4231/ITA/`) —
  repeated window starts over the same cable snippet
- Tokenizer-split CamelCase blobs (`LifeIsGoodAndI`, `OkLifeIsGoodAnd`) —
  consecutive window_starts over the same blob in the same sequence

The TFA pattern is consistent with each novel_code feature detecting a very
specific multi-token substring: as the sliding window traverses that
substring the same feature fires at near-identical magnitude with the
highlight shifting by one token. Stacked/Crosscoder features instead fire
once in each of many different documents.

## Caveats

- TFA-pos k=50 had **NMSE = 0.1246** on its eval — meaningfully worse
  reconstruction than Stacked (0.0585) or Crosscoder (0.0767). TFA-pos k=100
  diverged (NMSE=4.54) even with the v2 NaN-prevention fix. So TFA's feature
  dictionary is trained with a weaker reconstruction signal than the other
  two; its features may reflect training dynamics rather than the idealized
  novel/pred decomposition.
- **novel_codes only** — we're ignoring pred_codes, which carry the dense
  context-predicted signal. A feature being passage-local in novel_codes
  doesn't mean the model doesn't generalize — generalization may live in
  pred. The pred/novel split analysis is the next step.
- 10,000 training steps is short for d_sae=18,432 on real LM activations.
- Only one layer (resid_L25) and one k, so all claims are layer/k-specific.
- No LLM explainer yet — the "document-initial", "botanical", "URL-fragment"
  labels above are my inspection of a handful of examples, not validated.

## Follow-up: pred vs novel decomposition for TFA (analysis 1)

The "TFA is passage-local" claim above **was a novel_codes artifact**. TFA
splits its latents into two essentially disjoint subsets:

- **Novel** (sparse, topk=50 per token): detects local passage-specific
  phenomena, activations of order 0.03 (scaled down by `lam=1/(4·d_in)`).
- **Pred** (semi-dense, ~3619 nonzero of 18432 per token): carries
  attention-predicted context, activations 1–4× larger per entry and
  contributing ~6× more to reconstruction (`||D·pred||` median 174 vs
  `||D·novel||` median 29 per window).

Top-50 features by pred_mass vs top-50 by novel_mass have **zero overlap**.
TFA has learned two separate feature libraries.

Re-running the scan with TopKFinder ranking by pred_codes instead of
novel_codes (new `tfa_pos_pred` model type) gives:

| metric | tfa_pos (novel) | tfa_pos_pred | stacked_sae | crosscoder |
|---|---:|---:|---:|---:|
| active feats / 18432 | 15,664 | 7,796 | 15,932 | 2,970 |
| med chain-diversity of top-10 | **1** | **10** | 10 | 10 |
| all-10-from-same-chain features (of 100) | **55** | **0** | 0 | 19 |

**TFA pred features are document-general** — indistinguishable from
stacked_sae or crosscoder on the chain-diversity metric. TFA's passage-
locality is entirely in the novel_codes path; pred_codes generalize
normally.

## Follow-up: temporal spread per feature (analysis 2)

Per-feature concentration score = mean over exemplars of
`max_position_activation / sum_position_activation`. Value 1 = fully
localized at one of the T=5 positions; 1/T=0.2 = uniform spread.

| arch | median conc | localized (>0.5) / spread (<0.3) of 300 | peak position counts |
|---|---:|---:|---|
| stacked_sae | 1.000 | **300 / 0** | {0: 90, 1: 58, 2: 48, 3: 50, 4: 54} — uniform across positions |
| **tfa_pos (novel)** | **0.244** | **0 / 286** | spread evenly |
| tfa_pos_pred | 0.997 | 300 / 0 | **{0: 257, 4: 43}** — only boundaries |

- Stacked is *trivially* position-localized (each position has its own
  independent SAE; the feature at position t has no well-defined
  activation at position t′≠t). Peak distribution is flat because each
  position's SAE has its own feature library.
- **TFA novel is the only arch where features genuinely span the window.**
  95% of features have concentration near 1/T — fires on multiple
  positions per window. Consistent with the novel codes detecting
  multi-token substrings (URLs, CamelCase blobs) where every position in
  the window is inside the same pattern.
- TFA pred peaks at the causal-attention boundaries: position 0 (where
  context is just the zero vector so pred ≈ bias_v constant) and
  position 4 (full context available). Middle positions 1–3 get no
  concentration mass.

## Follow-up: cross-arch feature matching (analysis 3)

Best-match cosine similarity between decoder directions. Using
per-position decoders at position 0 (TFA's D is position-shared):

| pair | median best-sim | features sim>0.7 | features sim>0.5 |
|---|---:|---:|---:|
| stacked[0] ↔ crosscoder[0] | 0.11 | **340 / 18432** | 1018 |
| stacked[0] ↔ tfa | 0.08 | **0** | **0** |
| crosscoder[0] ↔ tfa | 0.08 | **0** | **0** |
| control: stacked[0] ↔ stacked[1] | 0.12 | 2377 | 3991 |
| control: crosscoder[0] ↔ crosscoder[1] | 0.21 | 1630 | 2698 |

- **Stacked and Crosscoder share a small aligned core**: 340/18432
  features match with cosine > 0.7 at position 0. Max alignment 0.964.
  Enough that cross-arch feature correspondence is real but sparse (~2%).
- **TFA is decoder-disjoint from both** (zero features above 0.5 in
  either direction). TFA's D lives on a genuinely different basis —
  consistent with the "TFA uses two independent libraries" finding
  above and with the attention-driven forward pass producing outputs
  that don't align with linear-encoder SAEs.
- Within-arch: even at position 0 vs 1 inside one arch, only ~2400/18432
  features match strongly. Real per-position specialization exists
  inside stacked and crosscoder too.

## Unified picture

Three architectures, three distinct feature-discovery profiles:

| property | stacked_sae | crosscoder | tfa_pos |
|---|---|---|---|
| Latent capacity used | 86% | 16% (tight core) | 85% novel + 42% pred (disjoint) |
| Per-position behavior | Independent per-position SAEs | Shared latent, position-weighted decode | Shared decoder; novel spreads across pos, pred peaks at boundaries |
| Feature locality | Document-general | Document-general | Novel: **passage-local**. Pred: document-general. |
| Decoder alignment with others | ~340 shared with crosscoder at pos 0 | ~340 shared with stacked at pos 0 | **Disjoint from both** |
| Where "temporal features" (spread across positions) live | Impossible by construction | Not per-position observable; single z per window | Novel codes (unique among archs) |

TFA's novel codes are the **only features in the sweep that span multiple
token positions within a single window**. They detect specific multi-
token phenomena (URLs, codes, camelcase) that stacked/crosscoder handle
via separate per-position features. This is the value-add TFA provides
to the feature library — even with its weaker reconstruction (NMSE 0.12
vs stacked 0.06, crosscoder 0.08), it surfaces multi-token structure
the others don't.

## Next steps

1. **LLM explainer** — run Claude on top-K windows for each arch's top
   features to get natural-language labels, enabling semantic cross-arch
   comparison on top of the structural findings above.
2. **Second layer (resid_L13)** — rerun the sweep invocation 3 (unshuffled
   L13, ~8h), then repeat analyses 1–3 on a shallower layer. Probably
   different feature types (more syntactic vs L25's more semantic).
3. **Generalize TFA pred-analysis** — apply the same novel/pred split to
   TFA without positional encoding (`tfa`), and on shuffled-training
   checkpoints to confirm pred's generalization advantage is
   architectural not data-dependent.
4. **Reliability of TFA novel codes** — given TFA's weaker NMSE and
   the fragile training dynamics (NaN risks fixed by `proj_scale.clamp`
   + grad-finiteness check), check whether the passage-local novel
   features survive a longer-training or smaller-LR re-train.

## Files

- Adapters: `temporal_crosscoders/NLP/bench_adapters.py` (commit `3b8759b`)
- Scanner: `temporal_crosscoders/NLP/scan_features.py`
- Scan outputs: `results/nlp_sweep/gemma/scans/scan__<arch>__resid_L25__k50.json`
- Sweep checkpoints: `results/nlp_sweep/gemma/ckpts/`
