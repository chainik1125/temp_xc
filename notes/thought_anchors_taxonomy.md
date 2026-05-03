# Thought Anchors — sentence taxonomy and reuse plan

Source: Bogdan & Macar et al., "Thought Anchors: Which LLM Reasoning Steps Matter?", ICLR 2026 review (arXiv 2506.19143). Pulled 2026-05-02.

## Sentence taxonomy (Section 3.1, verbatim)

> "We adopted the framework by Venhoff et al. (2025), which describes distinct reasoning functions within a reasoning trace. We define eight categories (see examples and frequencies in Section D):
>
> 1. **Problem Setup**: Parsing or rephrasing the problem
> 2. **Plan Generation**: Stating or deciding on a plan of action, meta-reasoning
> 3. **Fact Retrieval**: Recalling facts, formulas, problem details without computation
> 4. **Active Computation**: Algebra, calculations, or other manipulations toward the answer
> 5. **Uncertainty Management**: Expressing confusion, re-evaluating, including backtracking
> 6. **Result Consolidation**: Aggregating intermediate results, summarizing, or preparing
> 7. **Self Checking**: Verifying previous steps, checking calculations, and re-confirmations
> 8. **Final Answer Emission**: Explicitly stating the final answer"
>
> "Each sentence in the analyzed response is assigned to one of these categories using an LLM-based auto-labeling approach (detailed in Section E). Categories that rarely appear are omitted from the figures below. **Residual-stream probes accurately distinguish categories (see Section F).**"

## Setup we should match

- Model: DeepSeek-R1-Distill-Qwen-14B (paper main); also reports R1-Distill-Llama-8B in Appendix B — **same model family as ours**.
- Dataset: MATH (Hendrycks 2021), 20 challenging problems chosen for 25-75% solve rate. **Same dataset family as our backtracking case study.**
- Decoding: T=0.6, top-p=0.95.

## Where it intersects our backtracking work

Our existing backtracking case study covers a single category: **Uncertainty Management** (item 5; "including backtracking"). Two of the other categories are natural drop-in case-study candidates without redoing the paper's analysis:

- **Plan Generation** (item 2) — paper finds high counterfactual importance alongside Uncertainty Management (Figure 3B). Strongest candidate for a second case study because the existing pipeline (sentence labels → mining → b3 rescue) ports directly. We just need new labels.
- **Self Checking** (item 7) — verification / re-confirmation. Also a candidate but lower-priority since paper does not flag it as high-importance like (2) and (5).

## What we want to avoid duplicating (per Aniket / Dmitry instruction)

- Their thought-anchor / counterfactual-importance methodology — that's their contribution.
- Their attention-head / receiver-head analysis.
- Their KL-divergence importance metric.

## What we'd actually do for a second reasoning case study

Pick **Plan Generation** as the second category. Repurpose the existing pipeline:

1. **Sentence labels**: re-run the LLM judge on Stage A traces, but with the Plan Generation rubric (item 2 above) instead of the backtracking rubric. Reuse the auto-labeler approach the paper validated.
2. **Mining**: train a new probe / find features selective for Plan-Generation sentences (D+/D- = is_plan / not_plan), same `mine_features.py` flow.
3. **Detection**: same sparse-probing setup as the backtracking detection (NEURIPS_PUSH §3) — AUC vs |S|, raw-resid baseline, paired Wilcoxon.
4. **Steering**: same b3 rescue protocol with the new feature; ask whether steering toward "more planning" rescues wrong answers.

Headline framing: "TXC detects multi-token reasoning behaviors at the sentence level — we demonstrate this for two categories from the Bogdan & Macar 2026 taxonomy: Uncertainty Management (backtracking, our existing case study) and Plan Generation (a second case study)."

## Cost estimate vs Sunday EOD freeze

- Sentence relabeling for Plan Generation: ~$5-10 in Sonnet judge calls + 2-4h wall-clock; resumable.
- New mining + detection probes: ~1h coding (mostly reuse) + 30m compute.
- New b3 rescue sweep on Plan Generation feature (4-5 arches × 25 mags × 60 questions): ~2-3h on 2 GPUs.

**Total ETA: ~5-7h.** Doable iff the headline backtracking sweep + TFA + MLC extension lands by tomorrow morning, leaving Sunday for this. If the existing chain hits any blockers, defer Plan Generation to "future work" in the writeup.

## References to copy into the paper

- Cite Bogdan & Macar 2026 for the taxonomy.
- Cite Venhoff et al. 2025 (the original framework Bogdan adopted).
- Cite Muennighoff et al. 2025 for "backtracking sentences boost accuracy" (motivation for our existing case study).

## Action — UPDATED 2026-05-02 evening

**Per Aniket: Plan Generation is OPTIONAL EXTRA only. Stronger backtracking case study trumps a second category every time.**

Do NOT start the Plan Generation case study unless:
1. The full backtracking headline (TXC + SAE + TSAE-paper + TFA + MLC, calibrated, with flip matrix + McNemar + judge validation + detection probe) is shipped and looking strong, AND
2. Aniket explicitly green-lights it.

If we have spare cycles before then, the right place to spend them is making the backtracking story tighter:
- More finely-resolved magnitude grid around peaks (densify if SAE peak is still narrow)
- Better judge validation (κ ≥ 0.6 blind agreement on 20 transcripts)
- Cleaner sentence-level detection AUC vs |S| panel
- Repetition-rate-vs-magnitude auxiliary plot (judge-free)
- Per-question flip examples in the appendix

Keeping this note as a reference for if/when we have time to add a second category.
