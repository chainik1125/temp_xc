# July 29 Backtracking closeout and Temporal Screen pivot

**Recorded:** 2026-07-29
**Status:** Current plan from Aniket and Dmitry's July 29 one-on-one.

This plan separates the remaining NeurIPS evidence repair from the next
research phase. The Backtracking runs should match the submitted 300K
architecture, training, and evaluation contract, with the documented RNG
seeding correction; the Temporal Screen work starts in literature-review
mode. Literature-based task inventory continues, but new GPU experiments and
post-hoc searches for TXC-positive tasks wait until the screen has an
independent validation criterion.

## 1. Backtracking experiments to rerun

### New training required

The minimum new 300K training matrix is:

| Representation | Window | Width | Training seeds | Purpose |
|---|---:|---:|---|---|
| TXC-base | \(T=5\) | 32,768 | 1, 2, 42 | Three fully seeded replications of the paper-faithful C7 cell |
| T-SAE | \(T=5\) | 16,384 | 42 | Requested width-sensitivity control |

All cells use the paper's Llama-3.1-8B layer-10 Backtracking activation cache,
\(k_{\mathrm{pos}}=20\), batch size 1,024, and 300,000 optimizer steps. The
new TXC runs seed Python, NumPy, CPU Torch, and CUDA before model
initialization; the historical runner did not fully seed model initialization,
so its nominal seed-42 aggregate must not be mixed into the new three-seed
mean.

The submitted T-SAE control at matched width 32,768 is already a 300K,
seed-42 cell. Keep it as the matched-width reference and do not retrain it
unless its artifact audit fails. The new 16,384-width cell tests the
reviewer's concern using the original T-SAE width; it is not a replacement for
the matched-width comparison.

The meeting's phrase "one seed for the others" refers to non-TXC
Backtracking representations. T-SAE is the only additional representation
explicitly requested in the discussion. It does **not** request new 300K
RLHF, emergent-misalignment, SAE, MLC, or TXC-Pro training.

### Evaluation required after training

Run both evaluations on each new TXC seed and on the new T-SAE-16K control:

1. **Detection:** use the grouped Backtracking detector at
   \(S\in\{1,2,4,8,16,32\}\).
2. **Steering:** use the paper-faithful C7 steering protocol, including the
   genuine-backtracking judge, `cut_fraction=0.25`, and the frozen magnitude
   grid recorded in
   [the Backtracking closeout](../neurips-rebuttal/july29-backtracking-closeout.md).

Report the three TXC seeds individually and as mean \(\pm\) sample standard
deviation. The steering direction and the architecture ranking must agree
across seeds before calling the result replicated. Add steering to the
rebuttal tables; if the character budget permits only one Backtracking
summary number, use steering rather than the weaker detection-only summary.
Do not pool the earlier 20K window sweep with these 300K cells.

## 2. Why the project is pivoting

Backtracking is currently the only real task on which TXC has a clean,
defensible advantage. ROHF and emergent misalignment use supervision that is
too global or too weakly temporal to support the paper's broad empirical
story, so they should move to the appendix rather than drive another round of
post-hoc task hunting.

The replacement question is:

> Can a cheap, architecture-independent Temporal Screen identify tasks whose
> targets genuinely depend on ordered local history, before we train a TXC or
> inspect its performance?

If the screen predicts where ordered temporal dictionaries help, the paper
becomes a principled account of **when** TXCs are useful. If it does not, the
honest result is still valuable: it shows that temporal-looking language
tasks are often solvable through token-local or order-invariant statistics.

## 3. Temporal Screen research program

### Immediate mode: read before experimenting

The next week is a literature and design phase, not an open-ended GPU search.
By **Saturday at 9:00 AM Pacific**, Aniket will send Dmitry an annotated list
of papers covering:

- lag-dependent token and activation correlations;
- conditional entropy and useful context length;
- power spectra, cross-spectra, coherence, and long-memory processes;
- task-conditioned or sequence-conditioned temporal statistics;
- controls that distinguish order, frequency content, and simple averaging.

The working synthesis is
[Temporal Screen: annotated reading list and research synthesis](reading-list.md).

The "Surya paper" is
[Cagnetta, Raventós, Ganguli, and Wyart, *Deriving Neural Scaling Laws from
the statistics of natural language*](https://arxiv.org/abs/2602.07488).
It derives data-limited scaling behavior from the decay of token-token
correlations with lag and the decay of conditional entropy with context
length. Its earlier precursor is
[Cagnetta and Wyart, *Token-token correlations predict the scaling of the
test loss with the number of input tokens*](https://openreview.net/forum?id=ZqmtzfwH60).

These papers motivate a lag-decay and finite-data-noise-floor view, but they
do not provide a TXC benchmark screen: they study token statistics and
language-model scaling, not neural activations, downstream labels, power
spectra, or temporal dictionaries. Applying the idea to target-aligned
activation statistics is our proposed extension. Aniket should email Surya's
team to ask whether they are interested in collaborating; Dmitry said he
would also ask Francesco for his thoughts.

### Screen construction

The first proposal should compare several candidate statistics rather than
assume that one global correlation length is sufficient:

1. **Lag-resolved activation dependence:** estimate matrix-valued activation
   covariance as a function of lag, with bootstrap uncertainty and an
   explicit finite-sample noise floor.
2. **Target-aligned dependence:** measure how much the label-relevant signal
   changes with lag using cross-covariance, conditional information, or a
   capacity-matched raw-window probe. Raw autocorrelation alone can be large
   while being irrelevant to the task.
3. **Within-sequence heterogeneity:** compare per-sequence or task-conditioned
   decay to the ensemble average. Persistent but sequence-specific state can
   disappear in a corpus-wide average.
4. **Spectral diagnostics:** for approximately stationary sequences,
   transform the full lag-covariance object and study target coherence or
   cross-spectrum. DC power by itself is not enough because an
   order-invariant encoder can exploit it, and power alone discards phase and
   direction.
5. **Behavioral controls:** compare ordered history with last-token,
   best-single-offset, shuffled, reversed, and order-invariant windows.
   A temporal score should identify information that specifically requires
   ordered history.

Dmitry has already found that a naive single decay-length screen can fail.
The screen therefore needs positive and negative controls with known temporal
structure before it is trusted on ambiguous natural-language tasks.

### Anti-cherry-picking protocol

1. Build a task inventory without looking at new TXC results.
2. Validate candidate screen metrics on synthetic temporal and non-temporal
   controls with known ground truth.
3. Freeze the screen, thresholds, confound checks, and TXC window predictions.
4. Screen all candidate real tasks, retaining both passes and failures.
5. Split tasks into architecture-development and untouched holdout sets.
6. Only then train matched SAE, T-SAE, ordered TXC, and temporal controls.
7. Test the preregistered claim that higher screen scores predict a larger
   ordered-TXC advantage and a longer useful window.

This makes negative tasks part of the evidence and prevents "temporality"
from being defined retrospectively as whatever TXC happened to solve.

## 4. Parallel research streams after the screen proposal

1. **Regular-TXC scaling:** remove the BatchTopK/nonlinearity ambiguity, start
   from an exactly matched T-SAE configuration, and test whether increasing
   temporal capacity yields a stable advantage.
2. **Principled steering:** determine how a window feature should induce
   cross-window versus per-token activation updates, using controlled
   synthetic settings before another large judge sweep.
3. **Spectral Cross-Coder:** replace TXC-Pro with a frequency-space
   alternative motivated by approximate translation invariance. Its features
   should expose interpretable timescales, but it enters the paper only if a
   preregistered spectral prediction succeeds.

The priority order is Temporal Screen first, then regular-TXC scaling and
steering, with the Spectral Cross-Coder as a bounded theory-driven pilot.

## 5. Coordination and submission path

- Aniket has budgeted roughly 10--15 hours per week for the next two weeks.
- Existing Matt-funded runs can finish where they are; new work should use the
  $500 SPAR-funded RunPod balance. If Aniket buys Claude through the SPAR
  card, send Dmitry the receipt immediately.
- The team planning dates are September 11 for the ICLR abstract and
  September 16 for the full paper. Submit ICLR as a backup, then withdraw it
  if NeurIPS accepts.
- The next Dmitry meeting should review the paper list and at least one
  falsifiable Temporal Screen proposal, not a collection of new TXC-positive
  experiments.
