---
author: Han
date: 2026-04-17
tags:
  - results
  - index
---

## NLP temporal-SAE architecture comparison — index and bottom line

Index of the four research logs from 2026-04-17, which together answer: **"are there types of temporal features one architecture finds that another doesn't?"**

### The bottom line

Yes. Training Stacked SAE, TXCDR, and TFA-pos on the same Gemma-2-2B-IT `resid_L25` activations at matched d_sae = 18,432 produces three nearly disjoint feature dictionaries that specialize in different semantic categories:

| Architecture | Alive features | Feature character | Example |
|---|---:|---|---|
| Stacked SAE | 14,939 | **Concrete lexical** — specific tokens, named entities | f9426 = "motor" detector |
| TXCDR T=5 | 2,204 | **Grammatical / multilingual** — function words, Arabic, Cyrillic | f6655 = function-word connectors |
| TFA pred-only | 7,854 | **Structural/positional** — tokens whose identity depends on context | f15410 = second digit of HH:MM |
| TFA novel-only | 10,578 | **Sequence-boundary markers** (partly a caching artifact) | f4826 = opening quotation marks |

The TFA pred-only category is the most striking result: these are features ("this digit is the minutes-digit of a time", "this decimal is in a rating") whose semantic identity depends sharply on preceding context. A per-token SAE is *architecturally* incapable of discovering them. TFA's causal attention gives it that signal.

### Converging evidence from four analyses

Four independent lines of evidence all point to the same conclusion:

1. **NMSE + shuffle delta** — TFA's reconstruction degrades 2× under position shuffle; TXCDR 1.1×; Stacked 1.0×. ([[2026-04-17-nlp-gemma-tfa-vs-txcdr]])
2. **Decoder alignment** — median best-cosine across architectures = 0.10–0.23, barely above the random-baseline 0.09. <1% of features have strong (≥0.5) match in another architecture. ([[2026-04-17-nlp-feature-comparison-phase1]])
3. **Activation spans** — span distributions separate cleanly into 4 regimes (TFA novel = 1-token transient, Stacked = 2-token bursts, TXCDR = 4-token windows, TFA pred = 4+ with long tail to 50+). ([[2026-04-17-nlp-feature-comparison-phase1]])
4. **Autointerp semantics** — top-unique features in each category read as semantically distinct, with 73%, 55%, 76% HIGH/MEDIUM coherence for Stacked, TXCDR, TFA-pred respectively. ([[2026-04-17-nlp-feature-comparison-phase2]])

The joint UMAP ([[2026-04-17-nlp-feature-comparison-phase3]]) makes the spatial separation visually obvious — four distinct territories, minimal overlap.

### Reading order

Recommended sequence for a new reader:

1. **Start with this index** (1 min) — for the bottom line.
2. [[2026-04-17-nlp-feature-comparison-phase1]] (10 min) — the structural-evidence story. TL;DR at the top; deep tables if you want the numbers.
3. [[2026-04-17-nlp-feature-comparison-phase2]] (10 min) — what the unique features actually are semantically. Has the most concrete examples.
4. [[2026-04-17-nlp-feature-comparison-phase3]] (3 min) — one-figure confirmation via joint UMAP.
5. [[2026-04-17-high-span-feature-comparison]] (5 min) — restricts the comparison to each arch's *temporal* subset, the sharpest version of "are the temporal features substantially different?"
6. [[2026-04-17-nlp-gemma-tfa-vs-txcdr]] (skim) — raw NMSE tables and sweep results, mostly useful as the source for the shuffle-delta numbers.

### Caveats that apply to all phases

- Single seed (42), single layer (resid_L25), single k (100) on Gemma-2-2B-IT alone.
- TFA trained 5K steps at bs=32 (after a stability fix); Stacked/TXCDR at 10K bs=256. Total-tokens are comparable but not identical.
- TXCDR has a severe dead-feature problem at this scale (only 12% of d_sae=18432 is alive). All Phase-1-onward analyses use the alive-feature-filtered versions.
- TFA novel category is partially confounded by the caching scheme — sequences are right-padded to 128 tokens, and TFA novel features collect on sequence-start tokens. A packed-sequence cache would likely shift the novel category toward genuinely transient intra-text features.
- DeepSeek-R1 reasoning traces would be the strongest follow-up test: TFA pred should shine on reasoning tokens where persistent topic/step features matter most. Infrastructure is in place; caching was interrupted and can be resumed.

### Reproduction

All analyses reproducible from checkpoints in `results/nlp_sweep/gemma/ckpts/` + cached activations in `data/cached_activations/gemma-2-2b-it/fineweb/`.

Scripts:

- `scripts/analyze_decoder_alignment_alive.py` (Phase 1a)
- `scripts/analyze_activation_spans.py` (Phase 1b)
- `scripts/analyze_tfa_pred_vs_novel.py` (Phase 1c)
- `scripts/run_phase2_autointerp.py` (Phase 2 — needs ANTHROPIC_API_KEY)
- `scripts/joint_umap_visualization.py` (Phase 3)
- `scripts/phase2_summary_plots.py` (Phase 2 summary plots)
