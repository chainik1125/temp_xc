## The TXC graveyard: empirical postmortem and theoretical boundary

**Status:** working synthesis, 2026-07-30

**Scope:** This document asks whether the accumulated negative results are bad
luck, fixable implementation failures, or evidence that a fixed-token-window
Temporal Crosscoder (TXC) is usually the wrong abstraction for contextualized
language-model activations.

The short answer is:

> A TXC is not universally doomed. It can compress a genuinely non-ambient
> statistic of a window into one sparse code, and it has done so on controlled
> synthetic tasks and on a small number of real tasks. But the broad claim that
> TXCs will generally discover useful temporally extended features in
> contextual residual streams now looks structurally implausible.

The central candidate problem is that transformer residual states are
*already temporal*. Attention has mixed preceding context into the current
position. Our results suggest that many behaviorally meaningful persistent
variables are therefore either already readable from the current residual,
or live on a timescale and semantic segmentation that a short fixed token
window does not reach. In those cases, a TXC pays a large reconstruction and
sample-complexity cost to re-encode several contextualized states without
gaining new task information.

This is a stronger conclusion than “we have not found the right benchmark,”
but weaker than “the architecture can never work.” The exceptions are
important because they define the boundary.

## What counts as a failure

The experiments below reached different stages, so they should not be added
into a single win-rate denominator.

- **Direct negative:** trained TXCs were compared with relevant baselines and
  failed to improve the target.
- **Raw-gate kill:** the current residual already matched the useful history
  ceiling, so training a TXC would not test a meaningful information
  advantage.
- **Control kill:** an apparent window result was explained by visible cues,
  document identity, random projections, unordered aggregation, or unequal
  resource accounting.
- **Inconclusive:** the comparison was compromised by sparsity, checkpoint,
  distribution-shift, or protocol issues.
- **Survivor:** a trained TXC retained a controlled advantage. A survivor need
  not establish that a shared temporal dictionary is the only explanation.

This distinction matters. Refusal direction, for example, was killed before a
large TXC panel because it is already a single-position direction. That is
evidence against the *application hypothesis*, not a failed TXC optimizer.

## Graveyard

### Persona drift

The persona experiment is the cleanest recent negative
([full result](persona_drift_txc/RESULTS.md)).

- Qwen3-32B layer-32 activations show a large Assistant-Axis drift over 15
  turns, from -9.87 to -25.29.
- For the primary eight-turn-history, four-turn-horizon target, raw history
  barely improves on the current residual:
  \(\Delta R^2=+0.0018\), 95% CI \([-0.0131,+0.0167]\).
- SAE plus TXC is *worse* than SAE:
  \(\Delta R^2=-0.0773\), 95% CI \([-0.1320,-0.0360]\).
- The standalone W=8 TXC reaches \(R^2=0.7025\), versus 0.8465 for the SAE,
  0.9024 for the raw current residual, and 0.9039 for raw history.
- Raw history contains a small short-horizon signal, approximately
  \(+0.01\)–\(0.02\ R^2\), but the TXCs compress it away.
- The W=8 model has 671M parameters on 240 training conversations, reaches its
  best validation NMSE at step 1,801, and worsens substantially by step
  10,000. The final checkpoint is a deliberately harsh test of the stipulated
  schedule: the first AuxK update occurred after the earlier optimum.

**Verdict:** direct negative for the stipulated 10k endpoint; the primary raw
history gate is unsupported, while five of six shorter exploratory
window/horizon cells have positive history-over-current point estimates. This
is not an optimizer-independent architecture falsification or a kill of
persona as a temporal task. Current Assistant-Axis position nevertheless
predicts future position extremely well. Smaller dictionaries and
validation-selected early checkpoints remain a legitimate rescue test. The
result is conditional on one representation seed and four held-out persona
styles. It is a strong warning about TXC sample complexity in this regime.

### Emergent misalignment

The published medical EM cells provide a direct negative:

- SAE PR-AUC at \(S=16\) is 0.690/0.745 across the two reported seeds.
- TXC-base is 0.542/0.560.
- The TXC shuffle gaps are -0.059/-0.002, providing no evidence that order
  carries the result.

These published relu-mix anchors are recorded in the
[EM experiment card](../explorations/actmix_em/CARD.md). The planned
BatchTopK-only T sweep in that capsule is blank and must *not* be counted as a
completed replication. The card explicitly reserves “paper-match” for a
blocked follow-up phase.

Later work makes the negative more informative, not less:

- Raw activation history has real EM information beyond the current position
  at middle depth; the maximum raw window AUC is about 0.860 versus 0.771
  per-token.
- A pooled Stacked SAE reaches PR-AUC 0.6516 versus the approximately 0.54 TXC
  headline, although all learned dictionaries over-fire under the
  train-to-rollout distribution shift.
- In steering, the reported ordering changes with the summary convention.
  On the peak-alignment criterion, lower is stronger and SAE is best at 68.5,
  Stacked is 69.3, and TXC is 74.6.

The pooled comparisons and their caveats are in the
[Stacked SAE sprint summary](../../docs/dmitry/sprints/2026-07-27_stacked_sae_10h/summary.md);
the raw-depth and code-probe audit is in the
[conversion-depth record](../explorations/conversion_depth/RECORD.md).
These are directional cross-protocol comparisons, not one clean table:
cohorts and base rates, sparsity budgets, architecture composition, and
train-to-evaluation firing rates differ.

**Verdict:** a direct negative in the published cells, with one of the
strongest indications that temporal information really exists in the
substrate but the TXC representation does not robustly capture it. The
follow-ups are directionally unfavorable to a unique TXC advantage but remain
partly inconclusive because the protocols differ and evaluation sparsities
are badly shifted.

### HH-RLHF preference structure

The
[HH-RLHF table](../explorations/actmix_rlhf/results/rlhf_table.md)
shows:

- Paper-match SAE AUC 0.6129, T-SAE 0.6306, and agentic TXC 0.6096.
- The TXC shuffle gap is only 0.0121.
- Three of the TXC's top features are length-spurious.
- In the BatchTopK-only sweep, TXC peaks around 0.626 and has negligible or
  negative shuffle gaps.
- The untrained Stacked control reaches 0.6174, exceeding both its trained
  value of 0.602 and the roughly 0.61 headline.

**Verdict:** control-killed negative. The metric is substantially recoverable
from architectural random features and length. There is no persuasive learned
temporal representation advantage.

### Static sparse probing

Across the 38 static probing tasks, the Stacked SAE obtains mean AUC 0.8694,
the per-token TopK SAE obtains 0.885–0.889, and the TXC family obtains roughly
0.89–0.90. The gaps are too small to support a temporal claim, while an
untrained Stacked model already reaches 0.8026
([summary](../../docs/dmitry/sprints/2026-07-27_stacked_sae_10h/summary.md)).

**Verdict:** near-tie, not a useful TXC application. Static tasks are not where
a temporal model should earn its added complexity.

### Refusal

Two versions of the refusal hypothesis died for different versions of the
same reason.

- Refusal as a direction was rejected at design time because the published
  direction is already recoverable from a single current position.
- Refusal/deflection recurrence was screened on two models without producing a
  clean keep. The frozen card-level verdict is GPT-2 **KILL** and Llama-8B
  **WEAK**. Visible marker counts and within-conversation controls explain much
  of the apparent window signal. Document identity had
  \(r_{\mathrm{doc}}\approx0.97\).
- The *constructed label's* eight-message kernel has approximately
  1,100–1,150-token support, around 16 times the longest tested 64-token
  window. This is a design timescale, not a measured behavioral recurrence
  kernel.

See the [refusal recurrence card](../explorations/task_hunt/refmark/CARD.md)
and the [task-hunt synthesis](../explorations/task_hunt/WRITEUP.md).

**Verdict:** design/control failure rather than a two-model categorical kill.
Short-range refusal is already a current-state direction, while the proposed
long-range recurrence label lies far beyond a practical nearby-token window.

### Explicitly temporal real-language tasks

The systematic task hunt screened roughly 25 candidates and took five to full
panels. The failures are useful because their controls expose recurring
mechanisms rather than a single bad optimizer
([complete record](../explorations/task_hunt/WRITEUP.md)).

| Candidate | Stage | Why it did not support TXC |
| --- | --- | --- |
| Operator rate in reasoning | Full 84-cell panel | Every window code loses to counting visible event sentences. |
| Punctuation intensity, three models | Full panels | No canonical margin; on Llama-8B the per-token code wins. |
| Sentence-length recency latch | Screen | Window gain exists but is order-free, contrary to the hypothesis. |
| Sentence-length level | Screen | Order-free and bounded by document identity. |
| Sentence-length dispersion | Screen | Nearly invisible to both current and window representations. |
| Dialogue turn-length level | Screen | The apparent 0.98 result is conversation identity. |
| Topic-switch clock | Screen | A single contextual position already carries time since switch. |
| Question-gap rate | Full panel, later demoted | Punctuation defines the target and a count baseline wins at longer windows. |
| Quoted-speech intensity | Passed screen, deferred | Literal quote counting bounds it at longer windows; profile is order-free. |
| Self-correction, question rate, verbosity | Passed screens, not panelled | Same converted/visible-count class that failed at panel stage. |
| Emotional-instability onset | Full 600-rollout screen | The pre-onset state is already current-position-readable; windows add nothing at the tested horizons. |

An important methodological caveat applies to the surviving task-hunt
results too. The program prohibited pooling \(T\) per-token SAE codes on the
grounds that this spends \(T\) times the code bandwidth. That is a defensible
test of *one-code online compression*. It is not a defensible test of offline
feature discovery, where the SAE code trajectory has already been computed
and a cheap temporal readout can pool it. Some apparent TXC gains may therefore
show compression, not a uniquely better dictionary.

### Relational and explicitly synergistic language tasks

The relational hunt tried to create real-language targets whose single-token
marginals were balanced but whose joint history was informative. This should
have been an ideal TXC regime
([record](../explorations/relational/RECORD.md)).

- At the embedding layer, agreement equality is genuinely joint:
  per-token and linear-window AUC are about 0.50, while a window MLP reaches
  0.77 when both constituents are in range.
- By layer 2, the current-position AUC is 0.983; by layer 4 it is 1.000.
- Contradiction/fact consistency is joint at the embedding layer, becomes
  linearly readable from the oracle pair by layer 2, and reaches
  current-position AUC 1.000 at the first reported per-token cell, layer 8.
- Other role and relation candidates were also converted, saturated, or
  otherwise failed their nonlinear-headroom gate.

**Verdict:** raw-gate kill, not a trained-TXC negative. Crucially, the positive
controls prove that the instrument can detect real joint information. The
relation becomes linearly decodable at the current position within the tested
model and depth range, consistent with—but not causally proving—the proposed
contextual-sufficiency mechanism.

### Synthetic tasks that are temporal but still do not separate

The [synthetic benchmark program](../explorations/synthetic/REPORT.md) is also
a graveyard for the idea that “temporal” by itself predicts a TXC win.

- Signed motion is not reliably recovered by any architecture in the
  controlled scarce-data regime.
- Changepoint mode, assumption state, and hedging drift are temporally
  persistent but ambient; per-token models are at or near the useful ceiling.
- On next-state prediction for the assumption process, T-SAE is slightly
  stronger than the TXCs.
- On linear-in-window backtracking intensity, Stacked and TXC-pre are
  essentially tied at 0.950 and 0.952.
- On slow, alternating, and mixed HMMs, Spectral Matryoshka and TXC are close:
  0.858/0.826/0.750 versus 0.843/0.844/0.742. A fast process alone does not
  create a special spectral-over-TXC advantage, although both window models
  substantially beat the per-token SAE on the alternating task.

**Verdict:** explicit time dependence is insufficient. The predictive
coordinate is whether the target is ambient, additive in the window, or
requires a joint cross-position operation.

### Spectral rescues

The power-spectrum work found a real specialized advantage, but not a general
escape from the same boundary
([summary](README.md)).

- On paper denoising, TXC-base remains best at \(R^2=0.483\), versus 0.412 for
  Spectral v1.
- On controlled single-process HMMs, TXC matches the spectral models.
- In a recovered-cohort sensitivity analysis, a plain Fourier XC is
  descriptively higher at \(T=2\) and lower by 0.015 PR-AUC at \(T=10\).
  Because Aniket's exact activation cohort was not recovered, this is not a
  clean head-to-head or a replicated Fourier win.
- Fourier models win clearly when several simultaneous random-phase
  narrowband causes must be factorized: direct recovery is 0.963/0.950 versus
  0.762/0.760 for TXC across two routed-task conditions.
- Learned frequency Matryoshka weights behave more like a modest training
  curriculum than a stable discovery of the correct semantic timescale.

**Verdict:** frequency is a useful inductive bias when the sources themselves
factorize spectrally. Reweighting frequencies does not solve current-state
sufficiency, fixed-window reach, or reconstruction of nuisance content.

### Loss-function rescues

Matryoshka and contrastive components did not provide a general rescue. In the
[post-family dissection](../explorations/synthetic/loss_dissection/results/dissection_table.md)
and
[pre-family dissection](../explorations/synthetic/loss_dissection/results/dissection_table_pre.md):

- Backtracking primary recovery is neutral under all added components.
- Pre-family primary metrics are neutral across all tested tasks.
- Matryoshka hurts some frequency and phase primary metrics.
- Contrastive loss helps the post-family frequency primary by about 0.093 but
  does not generalize across tasks.
- Several apparent auxiliary improvements trade against NMSE or energy AUC.

**Verdict:** objective tweaks can alter a task-specific bias, but there is no
evidence that they repair the core information and scale mismatch.

### C7 backtracking: pooled-SAE causal control

The strongest causal sign of life also admits a much simpler temporal
baseline. In the matched 20k C7 run, a fixed, label-blind max over each
ordinary SAE feature's last five activations selects feature 24530. Steering
with that one native SAE decoder direction reaches peak inducement
\(\Delta gc=0.8361\) at magnitude +16, versus 0.4590 at -12 for TXC-base and
0.0984 at -10 for the final-token SAE. The edge peaks are not the right
comparison: over each feature's productive moderate-dose lobe, max-pooled SAE
scores 0.2049 and TXC-base 0.1967. Their paired difference is +0.0082 with a
question-bootstrap 95% interval [-0.0628, 0.0792].

Pooling is used only to *select* the SAE feature. The intervention itself is
the same single decoder direction, hook, and norm calibration used by the
other arms. Thus this is not evidence that SAE received a stronger steering
operation. It shows that the ordinary SAE already contains a causally useful
direction that the old final-token feature miner missed.

The positional audit rules out one fixed lucky token. Feature 24530 has
positive selectivity at all five positions but ranks only 4--14 at any one
position; max pooling increases its selectivity by 2.63 times and beats every
individual position in all five held-out folds at S=1 and S=8. The best
current description is window-level presence detection: a simple logical-OR
inductive bias recovers the relevant temporally jittered or repeated evidence.

**Verdict:** direct negative for a uniquely TXC causal latent, and for the
claim that TXC merely stumbled on a better intervention procedure. The
remaining TXC contribution is narrower but real: its unsupervised training
learns a useful window aggregation, whereas the SAE baseline must be supplied
with fixed max pooling. This result compares matched 20k checkpoints. It does
not directly rerun the published 300k TXC-base cell or TXC-pro steering, so it
does not erase the robust 300k effect; it changes what that effect can
establish. See the
[pooled steering result](backtracking_sae_pooling/steering_baselines/RESULTS.md).

## Survivors: why “universally doomed” is false

The graveyard should not erase the experiments where TXCs genuinely work.

| Survivor | Evidence | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| C7 backtracking steering | The published 300k TXC-base reaches inducement \(\Delta gc=0.541\), versus 0.246 for a 300k Stacked SAE. | A learned one-code window feature can causally alter a specialized reasoning behavior. | A uniquely TXC latent: at matched 20k scale, fixed max pooling over an ordinary SAE finds a direction tied with TXC-base on the productive moderate-dose lobe and larger at the high-dose edge. |
| Paper synthetic denoising | TXC-base \(R^2=0.483\), versus SAE 0.363 and T-SAE 0.382 | A joint sparse code can improve hidden-state recovery in a constructed temporal process. | That the gain transfers to contextual language. |
| Synthetic backtracking intensity | TXC-pre 0.952 recovery; Stacked 0.950 | Window access almost perfectly recovers a non-ambient recent-rate latent. | A unique shared-code advantage; Stacked matches it. |
| Frequency velocity and multilane | TXC-post approximately 0.77/0.52; per-token models near zero | Position mixing can recover joint temporal variables with zero single-token information. | That natural LM states commonly have this structure. |
| Phase-pair and polynomial clock | TXC-post reaches near-perfect phase-pair recovery; TXC beats spectral models on the clock | TXCs can represent some relational and secret-sharing-style window statistics. | That power-spectrum structure is the explanation. |
| Han: backtracking intensity on real traces | TXC-pre \(r=0.207\) at \(T=8\), versus SAE 0.113, T-SAE 0.150, and Stacked 0.125 | A trained shared code can compact an additive recent-history statistic better than these frozen sparse baselines. | Order sensitivity or a general temporal-state win; the trained panel has no shuffled-code evaluation and the raw-window shuffle cost is small. |
| Han: turn-length trend | Confirmed TXC-post \(r=0.282\) at \(T=32\), versus per-token models near 0.03 | Eight active features per window can compress a discourse trend that per-token codes do not expose at the same online readout. | Superiority at matched total offline bitrate; the claiming TXC spends 16–32 times less code than the per-token trajectory and trained-code order dependence was not tested. |
| Aniket: pre-backtracking event detection | Ordered TXC AP rises from \(0.218\) at \(T=1\) to \(0.255\) at \(T=10\), versus 0.223 for an invariant SAE at \(T=10\) | A TXC is an effective 32-feature summary of a specialized pre-event activation schedule. | Unique information or expressivity: a positional SAE catches and then beats it when allowed 192–256 probe features, and the margin over supervised raw history is not bounded. |
| Trailing novelty | Strong GPT-2 and T=16 8B results survive several controls | There may be a real surface-quiet T-scaling case. | A clean position-free claim; the position residual remains material. |

Sources for the controlled synthetic and spectral results are the
[synthetic benchmark report](../explorations/synthetic/REPORT.md) and
[power-spectrum summary](README.md). The real task-hunt survivors and their
caveats are in the [task-hunt synthesis](../explorations/task_hunt/WRITEUP.md)
and [novelty cross-ratification](../explorations/txcwin/CROSSRATIFY.md).

The pattern is striking: the strongest wins occur when the target is
*non-ambient* by construction. A token's marginal carries little or no target
information, while the joint window does. The real-language survivors are
narrower and often admit an aggregation or one-code-compression explanation.
Their evidential status is heterogeneous: several are conditional on one
representation seed, pending ratification, sensitive to the chosen window, or
limited by sequential task selection. They define exceptions worth pursuing,
not a publication-ready estimate of a population win rate.

## Skeptical audit of the recent signs of life

### What was reviewed

For “Han's results,” the primary source is the July 26–27 task-hunt
[writeup](../explorations/task_hunt/WRITEUP.md), including the later wave-2
additions on `origin/arxiv`, and its
[rebuttal pack](../explorations/task_hunt/REBUTTAL_PACK.md). The wave-2 text is
not yet merged into this worktree, so the source state should be pinned as
`origin/arxiv` through Han's 2026-07-27 commits rather than silently treated
as part of the local file. I also checked the contemporaneous T-scaling recipe
search and static-probing reruns. Neither supplies an additional
claim-grade positive.

For “Aniket's backstroke task,” I interpret *backstroke* as the
pre-backtracking window sweep: there is no result or task named backstroke in
the repository. The reviewed artifact is Aniket's July 27
`backtracking_window_sweep_t16` reviewer bundle at commit `d9c7fc7b2`, not the
later Fourier sensitivity analysis on a partly recovered cohort. The latter
is documented separately in the
[Fourier result](analysis/backtracking_fourier_results.md).

### Han's task hunt

Han's writeup is unusually good about recording killed hypotheses and
qualification. The positive count nevertheless mixes three evidential stages
that should remain separate.

| Result | Strongest defensible reading | Why the headline needs narrowing |
| --- | --- | --- |
| Backtracking intensity | On 4,044 reasoning traces, TXC-pre reaches \(r=0.207\) at \(T=8\), above SAE 0.113, T-SAE 0.150, and Stacked 0.125. This is a real trained-code advantage on an additive recent-rate target. | No trained-dictionary shuffle evaluation exists. The order receipt is a different raw-activation screen, where shuffling costs at most 0.018 AUC and the gain is explicitly classified as order-free. Two T-SAE top-up seeds under-spend their sparsity budget, and the comparison pools separately generated activation caches. The in-band T-SAE bound is thin. |
| Turn-length trend | A fresh-seed TXC-post lane reaches \(r=0.282\) at \(T=32\), with its untrained twin at 0 and per-token SAE/T-SAE near 0.03. The one-code representation is extremely efficient. | The primary claim is intentionally asymmetric: 8 active features per *window* versus 8 per *token* for the baselines. It establishes compression, not that a TXC beats a pooled SAE trajectory at equal total bitrate. The round-one pooled arms failed their untrained controls. The order evidence again comes from a separate raw screen, and the screen's stronger visible-turn probe at \(T=32\) scores 0.587 versus 0.509 for the raw window probe; only the panel's different label-side evidence line is beaten there. |
| Trailing novelty | The multi-model replication makes it a plausible third natural survivor. | Position remains a material residual explanation and its clean window depends on model and \(T\). It is evidence to continue auditing, not yet a foundation for a general TXC claim. |
| Wave-2 breadth KEEPs | Section-marker age passes on 3/3 models; long-return rates pass on 2/3 models in dialogue and Python; the relevant gains reproduce under re-seeding. These are good *task-screen* results showing natural residual histories contain breadth and recency state. | They are raw-window screen results, not trained TXC-versus-SAE panels. All are order-free under the frozen test. Their current implication is “try explicit pooling or semantic-time state models,” not “TXC wins.” |
| Speaker-dominance order signal | Within-dialogue shuffle margins of roughly 0.035–0.081 survive re-seeding, suggesting a genuinely speaker-resolved ordered signal. | The level task itself is WEAK and the one KEEP is seed-fragile. No trained TXC panel exists. This is a candidate substrate, not a sign of life for the learned representation. |

The recent recipe-search result is also easy to overread. A 20k recipe gives
the first rising static-probing curve, \(0.8974\rightarrow0.9171\) from
\(T=1\) to \(T=16\), but it fails its frozen \(T=1\) floor, fails the
\(k=5\) preservation gate, is order-free, and was measured on one dev seed.
The cheaper 4k curves are even more dominated by low-\(T\) feature collapse.
Later annealing and batch-pool fixes did not rescue the gates. This is a useful
optimization diagnosis, not a promoted TXC result.

**Framework fit.** Han's two trained positives support the narrow
*one-code-compression* branch of the framework. Backtracking intensity is an
order-free running statistic for which a shared trained bottleneck beats the
tested Stacked code at fixed sparse readout. Turn trend is stronger evidence
that severe bottlenecking can force a useful discourse summary. The wave-2
screens mostly reinforce the proposed pivot: learn local semantic features,
then aggregate them over sentence, turn, speaker, or document time. They do
not falsify the contextual-sufficiency or reconstruction-mismatch diagnosis.

### Aniket's pre-backtracking window sweep

This result is technically much more solid than a casual reading of the old
six-point plot suggests:

- all reported \(T\) values in the wide run use the same 20,335-row,
  question-grouped cohort, with 2,498 positives;
- \(T\in\{1,2,4,6,10\}\) uses nested windows ending eight tokens before the
  labeled event, so this is genuinely pre-event prediction rather than
  detecting the event token;
- training exposure is matched at \(B T\) activation values per update, the
  sweep has three representation seeds, and effective TopK support is
  measured in the wide protocol;
- ordered TXC AP rises from \(0.218\pm0.005\) at \(T=1\) to
  \(0.255\pm0.008\) at \(T=10\), while last-token and invariant SAE controls
  remain around 0.21–0.22.

The result therefore should not be dismissed as noise, a changing evaluation
cohort, or a simple last-token leak. But the strongest interpretation does
not survive the baseline audit:

1. **Most of the gain is window access, not order.** The shuffled TXC reaches
   0.231 at \(T=10\); ordered-minus-shuffled is 0.024, smaller than the total
   \(T=1\rightarrow10\) gain. Shuffling, reversal, and circular shifts apply a
   fixed ordered-trained probe out of distribution. They measure sensitivity,
   not the causal value of order after retraining.
2. **The 32-feature readout is load-bearing.** At \(T=6\), seed 42, the
   positional SAE rises from AP 0.140 with 32 selected features to 0.242 with
   128, ties the TXC around 192, and reaches 0.278 versus the TXC's 0.259 at
   256. This is post-hoc and one seed, but it directly demonstrates that the
   apparent architecture gap can be a downstream compression-budget gap.
   The frozen three-seed low-support frontier makes the mechanism more
   specific: at \(S=32\), TXC minus positional SAE is -0.0037 at \(T=1\),
   then +0.0333, +0.0522, +0.0706, and +0.0838 at
   \(T=2,4,6,10\). The \(T=10\) seed-level 95% t interval is
   [+0.0511, +0.1166]. Adding positions, not an architecture gap at \(T=1\),
   creates the sparse-coordinate advantage. See the
   [temporal-bottleneck audit](temporal_bottleneck_frontier/RESULTS.md).
3. **Raw history remains the information ceiling.** When the supervised raw
   residual detector is admitted, the TXC margin over the strongest control
   is \(+0.0102\), with a question-bootstrap interval
   \([-0.0082,+0.0183]\). The TXC organizes existing history signal sparsely;
   the sweep does not show that it discovers information unavailable to a
   conventional temporal readout.
4. **Parameter count is not matched.** A \(T=10\) reference TXC has about
   2.684 billion trainable parameters, roughly ten times the corresponding
   per-token SAE. Exposure matching is valuable, but it is not parameter or
   independent-example matching.
5. **The causal bridge is incomplete.** The event-detection sweep and the C7
   steering result use related data but different checkpoints and protocols.
   They cannot yet establish that the history-compressing latents measured
   here are the latents whose steering induces backtracking.
6. **The first causal rank/order screen is not positive.** A companion
   equal-energy intervention audit finds that the six-row Ward slab has 89.1%
   of its energy in the first singular component and effective rank 1.67.
   More importantly, the full slab changes 0/12 held-out semantic
   backtracking events; a blinded audit gives identical per-checkpoint labels
   to the full, rank-one, reverse, shift, and baseline arms. The temporal
   profile score is therefore *not identified*, with observed semantic
   contrast zero. This favors a persistent-direction factorization story over
   an ordered high-rank expressivity story, without constituting a causal
   backtracking null.

**Framework fit.** This passes the reach and raw-headroom gates and provides a
real representation result at a frozen 32-feature output budget. It partially
passes the order gate, but much of the signal is order-invariant and the
perturbation control is not retrained. It fails to establish a representation
advantage once positional-SAE readout capacity is relaxed, and it does not
match model parameter count. The correct classification is therefore:

> **Strong sign of life for compact sparse history/schedule discovery on one
> specialized reasoning substrate; weak evidence that a TXC contains unique
> temporal information or should be the default temporal representation.**

That classification is fully consistent with the graveyard framework. Indeed,
it sharpens the surviving niche: TXCs can be useful when the product is a
small, online, pre-event summary of a known aligned activation schedule. It
does not rescue the broader expectation that reconstructing nearby
contextualized states will generally reveal latent behavioral dynamics.
The support and causal audits further suggest that this niche divides into
two claims that should not be conflated: compression of a stable but
instantaneous-basis-fragmented feature, and irreducibly ordered temporal
geometry. Backtracking currently supports the first much more strongly.

## The candidate foundational issue: temporalizing an already temporal state

Let

\[
H_t = F(X_{\le t})
\]

be the current residual state of a causal transformer, and let

\[
W_t = (H_{t-T+1},\ldots,H_t)
\]

be the TXC input window. For a target \(Y_t\), suppose

\[
Y_t \perp H_{t-T+1:t-1}\mid H_t.
\]

Then the complete history window has no Bayes-predictive advantage over the
current state:

\[
p(Y_t\mid W_t)=p(Y_t\mid H_t).
\]

For any deterministic TXC code \(C_t=g(W_t)\),

\[
I(Y_t;C_t)\le I(Y_t;W_t)=I(Y_t;H_t).
\]

This does not prove that a finite learned SAE must match a finite learned TXC.
A TXC could find a better task-aligned bottleneck at the same sparsity. But
*when the conditional-independence premise holds*, it removes the robust
information advantage of the window. An unsupervised reconstruction objective
then has no reason to prefer the small task-relevant part of history over the
much larger lexical and positional variance.

That premise is a target-, layer-, position-, and hook-specific empirical
hypothesis, not a generic theorem about transformers. Attention makes current
sufficiency plausible because it constructs each residual state from context,
but attention's existence does not establish sufficiency. Persona at short
horizons, EM at middle depth, and backtracking all show measurable raw-history
headroom at some hooks.

The resulting *candidate* contextual-sufficiency trap has three branches:

1. If the model has computed and localized the temporal variable at the
   current hook, a per-token residual representation can often read it.
2. If the state remains distributed across earlier residuals, it can still be
   genuinely causal: those states may affect later computation through
   attention and KV retrieval. A window representation may reveal it, but
   injecting one current decoder slice does not automatically instantiate the
   distributed state.
3. If the model retrieves the variable on demand from distant source
   positions, a short nearby residual window is the wrong memory interface.

The relational decoding profile matches the first branch: cross-position
equality is visible at the embeddings and later becomes current-position
readable. This is evidence for the accessibility pattern, not proof of the
model's causal computation.

## Why fixed-window reconstruction makes the problem worse

### The rate-distortion mismatch

A per-token SAE spends its sparse capacity reconstructing \(H_t\). A TXC with
one code for a window is asked to reconstruct approximately \(T\) residual
vectors, including their token identity, syntax, position, and already
contextualized content. If the target depends mostly on \(H_t\), the other
\(T-1\) states are nuisance under the downstream objective.

At fixed *one-code-per-window bandwidth*, the TXC must either:

- allocate features to exact trajectory and lexical detail;
- average away information that differs across positions; or
- accept worse reconstruction everywhere.

This is not the only fair frontier. If active support grows with \(T\), TXC
can match the per-token bitrate while retaining a shared window code. Results
should therefore report both the matched-online-bandwidth frontier and the
matched-per-token-capacity frontier. The latter removes the built-in
\(T\)-fold bitrate disadvantage, though it does not remove the
multi-position reconstruction objective or parameter burden.

The persona experiment exhibits all three pressures: far worse NMSE, early
overfitting, and loss of the small raw-history signal. Across synthetic tasks,
window architectures often gain latent recovery while moving to worse
reconstruction, showing the rate-distortion trade directly.

### Offset-specific template explosion

A semantic event can occur at any offset, with variable tokenization,
duration, and phrasing. A fixed-window decoder needs offset-specific weights
for these variants. The same semantic atom that a per-token SAE reuses at
every position can become many phase- and offset-specific trajectory
templates.

For the untied offset-specific decoder used in the persona run, the parameter
burden grows roughly with \(T\): the 8,192-latent W=8 TXC has 671M parameters,
eight times the 83.9M-parameter SAE, while overlapping windows do not create
eight times as many independent conversations. Weight tying, convolutional
kernels, or low-rank offset structure could change this scaling; the observed
implementation is nevertheless a poor statistical trade for
natural-language trajectories with irregular boundaries.

### The temporal Goldilocks problem

There may be very little useful territory at nearby-token scale:

- Short dependencies have already been integrated by attention into \(H_t\).
- Long-term phenomena such as persona drift, refusal recurrence, goals, and
  discourse state can span hundreds or thousands of tokens.
- Natural semantic transitions align to clauses, sentences, turns, tool
  calls, or reasoning steps—not a fixed number of tokenizer positions.

Thus a short window is redundant and a long window is infeasible. Refusal
illustrates the proposed mismatch: the local direction is
single-position-readable, while the constructed recurrence target has about
1,100 tokens of label support.

### Redundancy is common in the tested tasks; synergy gives clean separation

A persistent natural-language state usually affects many tokens. Each token's
marginal therefore contains redundant evidence about the state. Repeated
evidence can still improve denoising, rate estimation, and recency tracking
over one noisy current sample. A per-token SAE followed by mean, max, decay,
or HMM-style pooling is well matched to this regime.

Synergy supplies a distinctive *function-class* advantage over a sufficiently
expressive pooled per-token trajectory: the individual positions are
uninformative but their joint relationship is informative. XOR, equality,
phase, velocity, and secret-sharing constructions have this property. Our
strongest synthetic separations deliberately create it. Natural transformers
may eliminate observed window headroom when the relation becomes
current-position-readable, although decodability alone does not establish how
the model computes or uses it. Additive backtracking intensity is an
important counterexample to any stronger claim: a TXC can be useful as a
compact running summary without the target being synergistic.

“Global” is therefore not the same as “non-ambient.” Persona, refusal,
confidence, or topic can be global and still be readable everywhere. A
frequency or phase variable can be global while having zero single-token
mutual information.

### Reconstruction variance is not semantic causality

Translation-invariant two-point statistics motivate Fourier or
Karhunen–Loève bases for efficient reconstruction. Exact finite-window Fourier
diagonalization requires additional boundary or circulant assumptions; a
Toeplitz covariance only approaches that structure under suitable limits.
Two-point stationarity also says nothing by itself about higher-order
semantics or causality. The largest variance modes need not correspond to
temporally meaningful or causally useful features: DC content, position,
token identity, and random low-dimensional projections can dominate
reconstruction and probe metrics.

The HH-RLHF untrained controls and the denoising result—where DC-only spectral
features retain the hidden-state probe while AC-only features do not—are
warnings that a readable code need not be a learned temporal explanation.

### Detection does not imply a coherent intervention

A TXC latent describes a multi-position decoder pattern over an already
realized window. Steering normally injects one slice of that pattern at the
current token. There is no general reason for this intervention to instantiate
the historical cause that made the latent active, or to maintain the intended
state over future tokens.

The C7 result proves that this can work in a specialized case. It does not
remove the general mismatch between a trajectory detector and a
single-position intervention. Repeated, trajectory-wise, or closed-loop TXC
steering remains a distinct and largely untested intervention class.

## Is the TXC doomed?

### The broad version probably is

The accumulated evidence does not support:

> A fixed-window reconstruction TXC will generally discover more useful
> temporally extended features than a per-token SAE on contextualized LM
> residual streams.

That program appears caught between frequent current-state sufficiency, infeasible
long-range timescales, offset-specific sample complexity, and an objective
that rewards reconstructing nuisance trajectory variance. More scale or a
clever sparsity penalty does not directly address any of these.

### A narrow version survives

The defensible claim is:

> A shared cross-position sparse code can compactly represent non-ambient or
> synergistic window statistics that are not already linearized at the current
> position, and can sometimes provide a useful one-code online summary or
> causal handle.

This claim is supported by synthetic frequency, phase, clock, and
backtracking tasks; by the C7 steering result; and more tentatively by a few
real trailing-statistic tasks.

The architecture is therefore not mathematically doomed. Its presumed
*default domain*—nearby contextual residuals—is the part that looks
foundationally hostile.

## Predictions that could falsify this diagnosis

The contextual-sufficiency account should make new predictions rather than
serving as an after-the-fact story.

1. **Window headroom should shrink at conversion depth.** For a relation that
   is joint at embeddings, the window-minus-current gap should collapse soon
   after it becomes current-position-readable. Agreement and contradiction
   have this decoding profile, without by themselves proving a causal
   computation.
2. **Architecture-specific TXC wins should track conditional joint structure,
   not autocorrelation.** After controlling for the current residual, strong
   ACF or low-frequency power alone should not predict an advantage over a
   pooled per-token trajectory. Equality, phase, or matched-filter statistics
   should. Additive one-code compression is a separate claim.
3. **Simple latent dynamics should absorb most natural gains.** A per-token
   SAE followed by a matched-budget smoother, convolution, or state-space
   readout should match many TXCs on ambient and additive tasks, with the
   residual gaps concentrated in joint interactions or bitrate constraints.
4. **Semantic pooling should beat longer token windows.** At equal compute,
   sentence-, turn-, and reasoning-step trajectories should generalize better
   than high-\(T\) token TXCs on persona, discourse, and reasoning drift.
5. **Predictive objectives should beat retrospective reconstruction.** A
   history code trained to predict future residual or SAE state should retain
   more task-relevant history than a code trained to reconstruct every past
   residual.
6. **Natural TXC features should be phase-fragmented.** The same labeled event
   at different offsets should recruit different latents more often than in a
   per-token SAE. Event-aligned pooling should reduce this fragmentation.
7. **Causal wins should be rarer than detection wins.** Unless the latent
   corresponds to a maintained model state, injecting one decoder slice should
   fail to reproduce the temporal condition that activated the code.

The diagnosis has three separable falsifiers:

- **Representation:** a real target with raw history headroom, within TXC
  reach, for which TXC reliably beats matched pooled-SAE and window-access
  baselines.
- **Compression:** a useful natural target for which a one-code TXC dominates
  other representations at a frozen bitrate, even if an unrestricted offline
  trajectory can match it.
- **Intervention:** a TXC trajectory feature whose prescribed intervention is
  more effective and coherent than current-token SAE or direct-axis steering.

Any one would weaken the corresponding part of the proposed boundary. A
single experiment need not win all three claims simultaneously.

## A compute-triage rule for future TXC experiments

As a default, no new large TXC training run should begin until the proposed
task passes the following gates. These are compute-triage rules, not proofs of
absence, and a pure denoising or one-code-compression study may predeclare a
different gate.

### Gate 0: timescale and alignment

- Show that the target's dependence scale fits the proposed window.
- Test token windows against sentence-, turn-, event-, or reasoning-step
  pooling.
- Stop if the effect is mostly beyond reach or if boundary misalignment
  dominates.

### Gate 1: conditional raw headroom

Fit held-out probes from:

1. position, document/conversation identity, length, and visible cues;
2. current raw residual plus those controls;
3. raw residual history plus those controls.

Freeze one primary window, horizon, probe class, and minimum effect size.
Correct or jointly bootstrap any exploratory family. Predeclare the estimand:
some controls can be legitimate confounds in one study but mediators that
remove the phenomenon in another, so report controlled and uncontrolled
results when that distinction is uncertain.

Use matched probe classes. A linear window is enough for an additive
hypothesis; a predeclared nonlinear or oracle-position probe is required when
the proposed advantage is relational. The probe must pass a task-specific
positive control: the relational program showed that even a wide window MLP
can miss known pair information that an oracle-pair probe recovers. Require a
positive, uncertainty-bounded history-minus-current margin on a frozen target.
If no validated probe capable of expressing the hypothesized statistic finds
window headroom, stop for compute-triage purposes. This is evidence against
the proposed run, not a certificate that the information is absent.

### Gate 2: aggregation and order

Compare the raw window with:

- mean/max/exponential pooling;
- shuffled history;
- visible event counts;
- a flattened linear window;
- a small nonlinear or state-space readout.

If simple unordered pooling explains the gain, use a per-token SAE plus a
temporal consumer. Train a TXC only when one-code compression is itself the
goal or when shared cross-position structure has a specific predicted
advantage.

### Gate 3: representation baselines

Every trained panel should include:

- per-token SAE;
- T-SAE where relevant;
- pooled per-token SAE trajectories;
- Stacked/window access without shared code;
- untrained twins;
- TXC-pre and TXC-post only when their functional distinction matters.

Match both representation sparsity and downstream readout capacity. Also
match or explicitly report parameter count, optimizer steps, token-position
exposure, independent training examples, and checkpoint-selection procedure.
Report one-code bandwidth separately from total offline computation.

### Gate 4: generalization

- Split by document, conversation, topic, and semantic template.
- Use multiple corpus/generation and representation seeds.
- Compute grouped or hierarchical uncertainty at the unit of claimed
  generalization, not merely overlapping rows.
- Reject results dominated by identity, position, length, or random
  projections.
- Use a fixed validation criterion and checkpoint rather than the final step
  by default.
- For future-outcome tasks, randomize or counterbalance future inputs so
  history does not merely identify a fixed upcoming script.

### Gate 5: causal relevance

Before paying for a full steering study, require:

- a frozen rationale for why the feature might be a useful handle, even if it
  is not uniquely predictive beyond every current-state feature;
- its activation is not merely a surface count;
- the chosen decoder slice or trajectory-wise intervention has a coherent
  mechanistic interpretation; and
- a small magnitude/safety pilot before scaling judge spend.

The post-study success criterion is then whether the intervention beats a
current-token SAE or direct axis at matched target effect, coherence, and
off-target cost. This cannot be used as a pre-study gate.

## More promising pivots

The evidence points toward separating *semantic factorization* from *temporal
modeling*.

1. **SAE first, temporal model second.** Learn reusable per-token semantic
   features, then run explicit convolutions, exponential filters, HMMs, or
   state-space models over their activation trajectories. This directly tests
   whether the temporal operation adds value and avoids relearning token
   semantics at every offset.
2. **Use semantic time.** Pool activations by sentence, dialogue turn, tool
   call, or Thought Anchor before modeling drift. This extends reach and
   reduces phase/template explosion.
3. **Use a predictive-state objective.** Compress the history to predict
   future activations, future SAE states, or a frozen temporal target instead
   of reconstructing all past residuals. This removes much of the nuisance
   rate-distortion burden.
4. **Study the model's real memory interface.** If information is retrieved on
   demand, attention patterns, KV-cache states, or selected source-token
   activations may be more natural than nearby residual windows.
5. **Reserve TXCs for proved synergy or online compression.** Frequency,
   equality, phase, and low-bandwidth running-state tasks remain legitimate
   architecture tests. The experiment should state which non-ambient
   statistic the code is expected to expose before training.

## Bottom line

The failed hypotheses are converging on a coherent explanation:

- personas, refusal, topic clocks, agreement, and contradiction are largely
  current-state-readable at the hooks we tested;
- EM contains some window information, but reconstruction TXCs do not exploit
  it as well as simpler per-position/window-access alternatives;
- HH-RLHF and several task-hunt wins collapse under random-feature, identity,
  visible-count, or aggregation controls;
- several genuinely long-lived behaviors lie beyond practical token windows;
- the remaining clean TXC wins are concentrated in constructed synergy,
  specialized backtracking, and one-code compression.

That is not “the hypothesis failed again” as a collection of unrelated bad
outcomes. It is evidence for a boundary. At many tested hooks, the transformer
appears to have done most of the relevant temporal integration before the TXC
sees its input, and the remaining temporal information is often either simple
to aggregate, too long-range, or poorly aligned with reconstruction. Future
work should treat a TXC as a specialized model for demonstrably non-ambient
window statistics, not as the default sparse representation of temporal
behavior.
