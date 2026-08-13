---
author: Claude (novelty-check agent, curated)
date: 2026-08-10
tags:
  - reference
---

## Novelty check: BIRD temporal codes / DSM dictionary learning

Literature sweep (~20 queries + primary-source fetches, 2026-08-10) for the
five claims of [[2026-08-10_bird_temporal_codes]] and the results in
[[2026-08-10_bird_clock_results]]. Verdicts first, receipts after.

**Headline: claims 2, 3, 4 are the paper. Claims 1 and 5 have their core
technical content already published and should be demoted to method
sections.** One paper from seven days ago (Yun et al., DSM-trained
sparse-coding dictionary sold as mechanistic interpretability) occupies
claim 2's headline framing and must be cited and distinguished up front.
BIRD itself had zero citations in Semantic Scholar as of today — the field
has not yet moved on it.

### Verdict table

| # | claim | verdict | what survives as ours |
| --- | --- | --- | --- |
| 1 | softmax posterior code = noise-tied attention | partially novel | the *application* (temporal windows, learned global bank, posterior-as-interpretability-code) — the equation and architecture exist |
| 2 | DSM as dictionary objective; binding argument | partially novel — most exposed | LLM/sequence domain; the binding argument; the DSM-vs-recon ablation at matched capacity (Yun et al. never ablate the objective) |
| 3 | Bayes-exact yardstick, fraction-of-gap | partially novel, leaning novel | sequential + exact-posterior normalization (SynthSAEBench is i.i.d., absolute metrics); entropy law as *instantiating* BIRD eq. 4; the L0 law $q^{h+1-W}$ (BIRD has no sparsity counterpart) |
| 4 | mosaic transition in sequences; h+1 vs h+2 | partially novel — **strongest single novelty** | the sharp $W = h{+}2$ threshold, the one-step identifiability/consistency gap, domain-wall defects — found nowhere |
| 5 | σ-ladder = window ladder / feature timescale | **not novel** — demote | only "σ is a specified knob with analytic $W_c(\sigma)$, vs post-hoc readouts"; a method property, not a contribution bullet |

### The three closest neighbours (learn from and cite)

1. **Yun, Belsten, Bi, Kadkhodaie, Chen & Olshausen 2026**, *Toward a
   mechanistic understanding of inference in visual cortex and diffusion
   models* ([arXiv:2607.15693](https://arxiv.org/abs/2607.15693), posted
   2026-08-03): sparse-coding dictionary + lateral interaction matrix
   trained with **DSM**, read mechanistically (recovers V1 horizontal
   connectivity). Occupies claim 2's headline. Escapes: ISTA/MAP fixed
   point not a softmax posterior; natural images not sequence activations;
   **no DSM-vs-reconstruction ablation**; no Bayes-gap yardstick; no
   binding argument.
2. **Smart, Bietti & Sengupta 2025**, *In-context denoising with one-layer
   transformers* ([arXiv:2502.05164](https://arxiv.org/abs/2502.05164)):
   proves our softmax$(\langle\cdot,\cdot\rangle/\sigma^2)$ estimator is
   the Bayes-optimal denoiser (their Thm 3.1), framed via dense associative
   memory. Cite as the justification for the architecture rather than
   letting a reviewer find it.
3. **Niedoba, Zwartsenberg & Wood 2026**, *Filtered Posterior Mean
   Collections* ([arXiv:2605.24192](https://arxiv.org/pdf/2605.24192)) +
   **Kamb & Ganguli 2025** ([arXiv:2412.20292](https://arxiv.org/abs/2412.20292)):
   FPMC taxonomizes the whole softmax-over-template-bank-with-noise-tied-
   precision family; Kamb & Ganguli own the mosaic phenomenology we
   transpose to sequences.

Runners-up: cross-attention/sparsemax SAE ([arXiv:2604.14925](https://arxiv.org/abs/2604.14925)
— our architecture with a $1/\sqrt{d}$ temperature, reconstruction-only,
no Bayes reading); oracle-normalized SAE-gap + "dictionary is the
bottleneck" ([arXiv:2603.28744](https://arxiv.org/abs/2603.28744) — an
*ally* for claim 2); amortisation gap ([arXiv:2411.13117](https://arxiv.org/abs/2411.13117));
Persistent SAEs — feature timescales, post hoc
([arXiv:2607.17117](https://arxiv.org/abs/2607.17117)); Temporal SAEs —
contrastive-not-denoising, clean contrast
([arXiv:2511.05541](https://arxiv.org/abs/2511.05541)); local scores →
compositional generalization with causal test
([arXiv:2509.16447](https://arxiv.org/abs/2509.16447), dangerously
adjacent to claim 4); SynthSAEBench
([arXiv:2602.14687](https://arxiv.org/abs/2602.14687)); discrete-diffusion
memorization transition ([arXiv:2604.26841](https://arxiv.org/abs/2604.26841),
control parameter is dataset size not window). Name-collision warning:
*The two clocks and the innovation window*
([arXiv:2605.10019](https://arxiv.org/abs/2605.10019)) — training-time
"window", different content, skimming reviewers will flag it.

Useful negative results from the sweep: no hit anywhere for "score matching
instead of reconstruction for LLM/SAE dictionaries"; nothing on the
$h{+}1$ vs $h{+}2$ gap or kinetically-stable defects in generated
sequences.

### Positioning paragraph (agent's draft, lightly usable as-is)

> Our posterior head is an instance of the restricted-Bayesian denoiser
> family of the ELS machine (Kamb & Ganguli), given an exact
> information-theoretic account by BIRD and unified as Filtered Posterior
> Mean Collections; that the softmax with noise-tied temperature is
> Bayes-optimal is proven by Smart et al. for one-layer attention. Our
> contribution is not the estimator but its use as an interpretability
> code over temporal windows, with a learned global template bank, the
> posterior replacing the SAE latent. On objectives we follow Braun et al.
> in rejecting reconstruction but substitute DSM rather than downstream
> KL; Yun et al. concurrently train a DSM dictionary on images — we
> isolate the *objective itself* (DSM vs reconstruction at matched
> capacity) and supply the Bayes-exact sequential yardstick that makes the
> separation legible (94% vs 7% of gap closed). The polynomial clock
> specializes BIRD's entropy law to closed form, adding the optimal-L0 law;
> the $W{=}h{+}2$ generative-consistency threshold, one step above
> identifiability, appears to be new.

### Recalibration after discussion (Dmitry, 2026-08-10)

The proximity-ranking above overweights papers whose *purpose* differs
from ours — and "overlap" is the wrong lens anyway: this is prior work to
learn from. The cleaner taxonomy is three quadrants:

1. **Interp of diffusion models** (Kamb–Ganguli, BIRD, FPMC, Bradley):
   our theoretical substrate, not competitors. Smart et al. likewise is
   transformer theory that proves our estimator optimal — a citation that
   does our math for us.
2. **Denoising for representation learning** (Yun et al., vision/neuro):
   the one genuine methodological cousin — DSM as the training objective
   for a dictionary read as structure. Lacks the objective ablation, the
   Bayes yardstick, sequences, and the LLM target. Their vision-domain
   result makes our sequence-domain result *more* plausible; cite and
   build on it.
3. **Interp codes for LLMs** (SAE/crosscoder literature — our market):
   nobody here denoises. Bhalla is contrastive; the Sparsemax-SAE has our
   architecture minus the theory (reconstruction-only, $1/\sqrt{d}$
   temperature, no temporal windows). Pre-write the "didn't Wang et al.
   build this?" answer.

**Position: we occupy the empty intersection — quadrant 2's method ×
quadrant 3's target × temporal structure.** The program-level claim ("use
diffusion machinery to get better temporal codes, not to explain diffusion
models") has no occupant; the narrow place to stay careful is the headline
ablation claim, which should carry Yun et al. and the masked-regularization
SAE as its nearest neighbours.

### Recommendations adopted

- Demote the σ-ladder/timescale claim to a method property (done in the
  proposal's framing going forward).
- Headline the DSM-vs-reconstruction ablation and the Bayes-gap yardstick;
  the $h{+}2$ transition is the strongest standalone novelty.
- Cite Smart et al. and FPMC proactively as the architecture's pedigree.
