---
author: Claude (with Dmitry)
date: 2026-08-13
tags:
  - proposal
  - results
---

## Response to the BIRD diffusion SAE/TXC proposal

Response to the reviewer note at
`experiments/dsm-txc/theory/bird_diffusion_sae_txc_proposal.tex`,
incorporating the measured results of the 2026-08-10 → 08-13 DSM arc
([[2026-08-13_dsm_postmortem]] is the arc's verdict;
[[2026-08-12_bird_transfer_theory]] and [[2026-08-12_arc_review]] the
supporting analyses). Summary position: **the note's theory core is
sound and its central identification — diffusion adds value only where
joint support inference beats coordinatewise thresholds — is the right
frame. Its proposal is *not* killed by our DSM-objective post-mortem,
because it proposes diffusion as inference over codes under a frozen
decoder, not as a training objective. But several of its work packages
already have partial answers in this repo, and two of our findings
raise the bar its real-model stage must clear.**

### Verdict on the theory core

- The JumpReLU zero-temperature theorem, exact scalar posterior,
  TopK/BatchTopK corollaries, MMSE factorization, risk decomposition,
  Gaussian log-det bound, I–MMSE bridge, and linear-response appendix
  all check. The **limit qualification** (a fixed nonzero threshold is
  the ℓ₀-Gibbs zero-temperature limit, *not* the generic zero-noise
  limit of a fixed spike-and-slab prior, where the threshold collapses
  as σ√(2log 1/σ)) is the note's most important paragraph and is
  consistent with our [[2026-08-11_jumprelu_mmse_note]], which keeps a
  persistent threshold m₀/2 only via a minimum slab amplitude — one of
  the note's "equivalent descriptions".
- The nonorthogonal proposition (zero-temperature limit = joint ℓ₀
  sparse coding, not any elementwise activation) is the load-bearing
  insight; the entire value proposition of diffusion lives in the gap
  between joint and coordinatewise support inference.
- Minor: §2.2 says the TXC "uses BatchTopK by default" — true of the
  paper recipe, but the w6 experimental dictionaries in this repo are
  plain TopK-96; a footnote would prevent confusion. In §7.3, note the
  learned-MSE integral upper-bounds MI only when integrated over the
  same channel scan.

### What our results already answer (map to the note's hypotheses)

| note item | our measured evidence | status for the note |
| --- | --- | --- |
| §4.4 finite-temperature gate | `bayes_gate` IS this gate (σ-conditioned sigmoid × shrunk amplitude), trained at Gemma scale | WP0/WP1 partly done |
| H3 coherence gain (soft gate beats hard marginal thresholds on real dicts) | bayes_gate broke the sparse-probing wash (+7–11 pts k=5); post-hoc gate swap +3–5 pts on every arm (readout-borne) | **supported** |
| H2 orthogonal null | per-token probing wash for objective-level DSM at 10M and 100M tokens | consistent |
| gate pathologies | mean-gate sparsity pressure collapses the pool (24% death, dictionary concentration); fixed by per-latent rate-KL (alive 1.0 at L0 219/68/49); fragility worst-of-program and dictionary-borne | new constraints for §9 |
| §risk "non-identifiability masquerades as uncertainty" | measured: ~1% of DSM survivor-atom pairs have abs corr ≈ 1.0 (duplicate atoms) | **confirmed** |
| the σ²ln π prior-mass term in the preactivation | measured as direction-deep OOD collapse: 1.4 positive preacts/window off-distribution vs 162 on; recalibration revives 1 latent vs 98% for recon; clock numeric 0/961 vs 961/961 | mechanism validated, and it is the note's biggest deployment risk |
| covering-law pool behaviour | live pool 5–10% of H even fully on-domain (pre-registered 600–2,500; measured 1,063); ratio law fits two dictionaries at 1% | input to §7.6 |
| Tweedie-projection interventions | falsified at k=96 fidelity: density-matched DSM projector destroys generation at α=0 (Sonnet 0.25, 0/20), recon control less bad on every metric | do not build on |
| TempBench WP3 oracle plan | B1/B2 already measured: posterior head closes 94% of the Bayes gap on-domain; TXC+DSM ≥ recon on all four settings, gain sub-Rayleigh | M3 partly done |

### Two constraints our results impose on WP4 (real-model stage)

1. **Deployment-distribution capture is not optional.** Text-domain
   match alone fixed nothing at the steering site (mix-trained dict:
   NMSE 0.807, 5.2% live on distill-model activations); the binding
   axis is the *model* whose activations are read, and an affine
   base→distill pullback (R² = 0.86) revives nothing — feature birth,
   not coordinate change. Any latent prior for WP4 must be trained on
   codes from the deployment model's activations.
2. **The baseline is no longer TopK.** The mass-selected reconstruction
   core (top ~250 recon latents by activation mass, label-free) scored
   0.223 sentence-S8 vs the DSM survivors' 0.208 and random-849 floors
   of 0.12–0.15. A WP4 gain claim that does not clear the selected-core
   baseline — under selected (not random) capacity-matched nulls,
   untrained-dictionary and label-shuffle floors, and live-pool counts
   as the convention-free metric (we measured a 4× NMSE discrepancy on
   one dictionary from normalization conventions alone) — will not
   survive the controls this repo now applies by default.

### The mutual test: exemplar memorization on our trace-trained codes

The note's BIRD lens suggests a reinterpretation of our newest result
that neither document can currently exclude. Our trace-trained
dictionaries concentrate onto ~850–1,000 templates on a training corpus
of **n = 300 traces (log n ≈ 5.7 nats)**, and by the note's own
"sparsity is not information restriction" argument the codes carry far
more than 5.7 nats. Our dictionaries therefore plausibly sit **above
the exemplar-memorization boundary for the trace domain**, and part of
what we call covering-law concentration could be partial exemplar
retrieval — the note's conjectured intermediate regime (support frozen,
exemplar entropy unknown) is exactly where our dictionaries would sit.

Proposed diagnostic (cheap; runs on existing artifacts): for held-out
and training trace windows, compute nearest-training-example
concentration of the w6mix/w6dist codes (top-1 posterior mass proxy via
code similarity to training-window codes), train/test denoising
asymmetry, and the note's posterior-entropy proxy vs the Gaussian/
I–MMSE bound. Pre-registered readings: high NN-concentration on
training traces with a train/test gap = exemplar-memorization
component, which weakens our covering-law reading and supports the
note's phase analysis; low concentration and no gap = the 850 are
genuine reusable templates, which strengthens the covering-law reading
and gives the note a measured "support-frozen, exemplar-unfrozen" data
point for its Conjecture 7.6.

### Recommended revisions to the execution plan

- Skip M0–M1 as writing tasks: point at the exact-scalar machinery in
  [[2026-08-11_jumprelu_mmse_note]], the B3 atom-recovery preregistration,
  and the clock numerics in [[2026-08-12_bird_transfer_theory]]; start
  at M2 (coherent dictionaries), which nothing in this repo covers.
- Promote H7 (MI-matching vs ℓ₀-matching differ systematically): cheap,
  novel, untouched by our 72 hours, and directly buildable on the
  I–MMSE estimator the note specifies.
- Add the exemplar-memorization diagnostic above as a joint first
  experiment — it is the one place where the note and our program each
  might change the other's conclusions.
- Import the arc's instrument discipline into §Evaluation matching
  verbatim (selected-core baselines, floors, live-pool metric,
  convention-pinned NMSE).
