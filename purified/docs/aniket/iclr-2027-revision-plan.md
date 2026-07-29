# ICLR 2027 / arXiv revision plan

**Recorded:** 2026-07-29  
**Status:** Team direction proposed by Dmitry after the NeurIPS responses,
not yet a frozen experiment contract.

## Deadline contract

- Internal target: produce a coherent arXiv/ICLR candidate in the next two
  weeks, around **August 12**.
- ICLR 2027 abstract deadline: **September 11, 2026, AOE**.
- ICLR 2027 full-paper deadline: **September 16, 2026, AOE**.

The official [ICLR 2027 author
guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) require a
genuine abstract by September 11 and the full paper by September 16. The
submission is double blind and limited to nine main-text pages. The abstract
deadline is also the practical author-list freeze, so authors and OpenReview
profiles should be checked before then.

## Why the paper needs a structural revision

Dmitry's assessment is that the NeurIPS paper has roughly a 30--40% chance of
moving the reviewers, and that Backtracking is currently the only empirical
task on which the team can make a clean TXC-outperformance claim. The ICLR
version should therefore narrow the claim and rebuild the evidence instead of
adding isolated positive tasks after seeing their outcomes.

The proposed main claim is:

> A temporal dictionary is useful when a target depends on an ordered local
> state path, and a predeclared temporal screen can identify those settings
> before expensive dictionary training.

This replaces the broad suggestion that TXCs are uniformly better sparse
dictionary learners.

## Scope changes

1. **Keep Backtracking as the anchor.** It remains the best current
   real-model example and should receive the complete multi-seed,
   matched-baseline treatment.
2. **Move RLHF and emergent misalignment to the appendix.** Their existing
   labels and controls do not support a clean temporal-advantage claim.
3. **Remove TXC-Pro from the main method.** Use one regular TXC and eliminate
   the architecture zoo. Any extra component should return only through an
   isolated ablation.
4. **Resolve the BatchTopK/nonlinearity ambiguity before rerunning the main
   matrix.** Lock one exact sparse-activation formula, selection granularity,
   sparsity convention, and inference rule across tasks and window sizes.
   Do not mix the paper's per-window TopK arm with pre-pooling and post-pooling
   BatchTopK arms under one name.
5. **Treat Spectral Crosscoder as the single principled alternative.** It
   should enter the paper only if a preregistered frequency/phase prediction
   succeeds on a synthetic falsification test and at least one screened real
   task.
6. **Audit every citation, architecture name, and result pointer.** The
   current reference and naming inconsistencies make negative interpretations
   easier and obscure which model produced each result.

## Temporal screen

A task becomes a headline benchmark only after passing a fixed sequence of
cheap gates:

1. **Local target:** the label attaches to a token, sentence, transition, or
   event-aligned window. A rollout label cannot simply be copied onto every
   local window.
2. **Raw temporal opportunity:** an ordered raw-activation window beats the
   strongest single offset and order-invariant summary under grouped
   cross-validation and matched probe capacity.
3. **Order receipt:** verified reversal or non-identity shuffling reduces
   performance beyond seed variation. Fixed-probe perturbations are described
   as representation sensitivity; refitted controls are needed to separate
   information destruction from covariate shift.
4. **Window dependence:** performance changes meaningfully over a
   prespecified \(T\) grid rather than merely benefiting from more inputs.
5. **Robustness and confounds:** the effect survives at least three dictionary
   seeds and lexical, length, position, activation-norm, sparsity, and
   parameter/FLOP controls.

Failures are retained as negative results and classified as token-local,
order-invariant, globally supervised, or underpowered. Dictionary training
starts only after the raw temporal opportunity gate passes. This is the main
protection against task cherry-picking.

## Two-week internal push

1. **Days 1--2: freeze the scientific contract.** Choose the exact regular-TXC
   nonlinearity, write the screen thresholds, lock task/seed/window grids, and
   demote RLHF and EM before seeing new results.
2. **Days 2--5: repair the paper surface.** Remove TXC-Pro, fix references and
   architecture names, resolve the title collision, and generate one
   metadata-derived parameter/FLOP/sparsity table.
3. **Days 3--8: rerun the minimal core.** Use the frozen regular TXC and
   matched SAE, T-SAE, pooled/positional SAE, and raw-activation controls on
   the synthetic suite and Backtracking. Reuse old cells only when their
   protocols match exactly.
4. **Days 5--9: screen real tasks.** Start with deletion and the strongest
   literature-backed candidates. Train dictionaries only for tasks that pass
   the cheap screen, and publish a compact table of all screened tasks.
5. **Days 5--10: time-box Spectral Crosscoder.** Run one analytic synthetic
   test and one screened-task pilot. Stop if it does not improve the
   recovery--sparsity frontier or yield a distinct, interpretable prediction.
6. **Days 9--12: rewrite around the screen.** Make the positive and negative
   task results support one claim about when temporal dictionaries help.
7. **Days 12--14: freeze an arXiv candidate.** Audit artifacts and citations,
   obtain an adversarial internal read, compile the paper, and preserve a
   reproducible result manifest.

## Gates before ICLR submission

- At least two non-synthetic tasks, including Backtracking, must pass the full
  temporal screen for a broad empirical paper. Otherwise, use the honest
  fallback framing: a methods-and-benchmark paper mapping when temporal
  aggregation helps and when it does not.
- Every headline table must report dictionary seeds, measured sparsity,
  dictionary width, probe budget, parameters, FLOPs, and matched
  pooled/positional controls.
- The Spectral Crosscoder remains future work unless it beats regular TXC or
  produces a strictly better recovery--sparsity frontier on its preregistered
  tests.
- After the two-week arXiv checkpoint, use August for robustness and writing,
  not open-ended task expansion. Freeze title, authors, and genuine abstract
  before September 11; freeze the full PDF several days before September 16.

## Immediate coordination

Dmitry asked each collaborator for a concrete hours-per-week commitment for
the next two weeks. Aniket has said that LBL work is mostly complete and that
he is relatively free through the ICLR deadline, but should still give Dmitry
a numerical weekly estimate so the experiment and writing ownership can be
assigned.
