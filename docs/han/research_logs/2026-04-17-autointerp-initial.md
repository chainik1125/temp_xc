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

## Next steps

1. **pred vs novel decomposition for TFA** — for each feature, what fraction
   of total activation mass comes from pred vs novel across the top-K
   exemplars? Features where pred dominates should generalize across
   documents; novel-only features should stay passage-local. If that
   hypothesis holds, the "TFA is passage-local" result above is a
   novel_codes artifact, not a TFA-the-arch property.
2. **Temporal spread per feature** — how concentrated is each feature's
   activation on one position within the T=5 window vs spread across
   positions? This is the "is this a temporal feature" metric; requires
   the per-position codes, not the mean-over-T that TopKFinder currently
   stores.
3. **Feature matching across archs** — are there 1-1 correspondences between
   Stacked / Crosscoder / TFA features via decoder cosine similarity? Which
   Stacked features have no Crosscoder/TFA match, and vice versa?
4. **LLM explainer** — once 1-3 are in place, run gemma-2-2b-it or Claude
   on the top-K windows to get natural-language labels for each feature.
5. **Second layer (resid_L13)** — train one more sweep invocation (unshuffled
   only, ~8h) to get a shallower-layer comparison. Probably different
   feature types (more syntactic) vs L25 (more semantic).

## Files

- Adapters: `temporal_crosscoders/NLP/bench_adapters.py` (commit `3b8759b`)
- Scanner: `temporal_crosscoders/NLP/scan_features.py`
- Scan outputs: `results/nlp_sweep/gemma/scans/scan__<arch>__resid_L25__k50.json`
- Sweep checkpoints: `results/nlp_sweep/gemma/ckpts/`
