---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - reference
---

## Prior literature: denoising objectives and interpretability dictionaries

Consolidated from two search sweeps (2026-08-10; ~25 targeted queries) and
discussion. Companion to
`docs/dmitry/proposals/2026-08-10_bird_novelty_check.md` (per-claim
verdicts, positioning paragraph); this note focuses on the central
question: **has anyone trained SAEs or nearby dictionaries with a
denoising objective?** Short answer: not in the LLM-activation ballpark,
and the near-misses circle the gap from every side.

### The three-quadrant map

1. **Interpretability *of* diffusion models** — our theoretical substrate,
   not competitors: ELS machine
   ([arXiv:2412.20292](https://arxiv.org/abs/2412.20292)), BIRD
   ([arXiv:2607.08041](https://arxiv.org/abs/2607.08041)), Filtered
   Posterior Mean Collections
   ([arXiv:2605.24192](https://arxiv.org/pdf/2605.24192)), local-score
   compositionality ([arXiv:2509.16447](https://arxiv.org/html/2509.16447)).
   Smart, Bietti & Sengupta
   ([arXiv:2502.05164](https://arxiv.org/abs/2502.05164)) prove the
   softmax-with-noise-tied-temperature estimator is the Bayes-optimal
   denoiser (our posterior head's pedigree).
2. **Denoising *for* representation learning** — the methodological
   neighbours: Yun, Belsten, Bi, Kadkhodaie, Chen & Olshausen
   ([arXiv:2607.15693](https://arxiv.org/abs/2607.15693)): DSM-trained
   sparse-coding dictionary on natural images, read mechanistically (V1
   connectivity); posted 2026-08-03; no DSM-vs-reconstruction ablation, no
   sequences, no LLM target. Behind it, the classical line: Vincent et
   al.'s denoising autoencoders (2008–2010) — corrupted input, clean
   target — turn near-identity autoencoders into learners of visibly
   interpretable Gabor/stroke filters, and Alain & Bengio (2014) prove
   DAEs estimate the score, making DAEs the direct ancestor of diffusion.
3. **Interpretability codes for LLMs** — our target market, where nobody
   denoises. Temporal SAEs
   ([arXiv:2511.05541](https://arxiv.org/abs/2511.05541)) use a
   contrastive smoothness term on top of reconstruction; the
   cross-attention/sparsemax SAE
   ([arXiv:2604.14925](https://arxiv.org/html/2604.14925v1)) has our
   architecture with a fixed $1/\sqrt{d}$ temperature, trained purely to
   reconstruct; end-to-end SAEs
   ([arXiv:2405.12241](https://arxiv.org/abs/2405.12241)) reject
   reconstruction but substitute downstream KL.

**Position: we occupy the empty intersection — quadrant 2's method ×
quadrant 3's target × temporal structure.**

### The near-misses, sorted by proximity

- **Corruption as a regularizer**: masked-regularization SAEs
  ([arXiv:2604.06495](https://arxiv.org/html/2604.06495v1)) randomly
  replace tokens during SAE training to disrupt co-occurrence; corruption
  present, but no denoising target.
- **Noise in the latent, not the input**: variational SAEs
  ([arXiv:2509.22994](https://arxiv.org/abs/2509.22994)) sample codes from
  Gaussian posteriors with a KL prior and *underperform* — the opposite
  mechanism (dispersive latent regularization, not input-corruption
  estimation). Their negative result says nothing about DSM.
- **The demand side without the supply**: the SAE-fragility literature
  ([arXiv:2505.16004](https://arxiv.org/abs/2505.16004); an
  [ACL 2026 findings paper](https://aclanthology.org/2026.findings-acl.1298.pdf))
  shows tiny input perturbations flip SAE concept readouts and concludes
  SAEs may be "ill-suited for monitoring without further denoising" —
  diagnosing exactly the deficiency a denoising objective trains away.
- **SAEs used *to* denoise** other artifacts (steering vectors:
  [arXiv:2509.23799](https://www.arxiv.org/pdf/2509.23799)) — the reverse
  direction.

### Why the gap plausibly persisted

- Around 2010 the lineages forked: denoising went to SSL (BERT, MAE) and
  diffusion; the interp SAE descends from Olshausen–Field sparse coding
  via Ng's sparse autoencoder into *Towards Monosemanticity*, which
  standardized reconstruction + sparsity.
- **The standard scoreboard punishes denoising**: SAE evals lead with FVU
  and CE-loss-recovered, and DSM-trained dictionaries score *worse* on
  clean-reconstruction FVU while learning better codes (measured in our
  B2/B3). Anyone who tried the obvious experiment and checked the
  standard metrics would have concluded "worse SAE."
- The faithfulness instinct: activations are treated as ground truth to
  preserve exactly; corrupting them feels like studying a different model.
- Activation noise had no semantic reading until the
  superposition-interference interpretation (interference from other
  active features is approximately Gaussian by CLT).

Epistemic caveat: two sweeps ≈ 25 queries are not proof of absence;
unpublished "we added noise and FVU got worse" negative results likely
exist. But the published record leaves the specific move — DSM on LLM
activations, evaluated on what codes *detect* rather than what they
*reconstruct* — open, with the fragility literature actively asking for it.

### Benchmarks and evaluation neighbours

SynthSAEBench ([arXiv:2602.14687](https://arxiv.org/abs/2602.14687)) —
i.i.d. synthetic, absolute metrics, no Bayes normalization; oracle-
normalized SAE-gap analyses
([arXiv:2603.28744](https://arxiv.org/html/2603.28744),
[arXiv:2411.13117](https://arxiv.org/pdf/2411.13117)) — allies for the
"dictionary learning, not amortization, is the bottleneck" reading;
Persistent SAEs ([arXiv:2607.17117](https://arxiv.org/html/2607.17117v1))
— post-hoc per-feature timescales (vs our specified σ-ladder with analytic
$W_c(\sigma)$).
