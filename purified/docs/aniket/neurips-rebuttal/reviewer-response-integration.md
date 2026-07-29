# GPU-free reviewer-response integration

_Prepared from `neurips-aniket` commit `d9c7fc7b`, Dmitry's
`dmitry-txcwins-10h` head `f87bbbe75d`, and Han's `arxiv` head
`3bd31b0d5a`. This document supplies replacement text; it does not modify
either teammate's branch._

## Recommended allocation

- Put the final three-seed backtracking sweep and the human-deletion result in
  Reviewer bbby's response. They directly address robustness and whether local
  temporal structure matters.
- Replace Reviewer 4z15's old one-seed percentage table with the absolute
  three-seed backtracking table below. Keep the matched-capacity Stacked-SAE
  discussion, but disclose its probe-budget sensitivity.
- Treat sycgen as optional secondary evidence for useful windowed state. It
  does not establish learned order, and it is semi-synthetic rather than a
  real-world task.
- Keep Reviewer EAxU focused on empirical scope, definitions, citations,
  naming, and presentation. The refreshed response lives in
  `reviewer-3-response.md`.

## Reviewer 1/2 reshuffle audit

Dmitry's current OpenReview-ready files are already at 4,987 characters for
Reviewer 1 and 4,349 for Reviewer 2, including their front matter. Additive
editing will not fit. The clean reshuffle is:

1. Move the full Shamir secret-sharing section and table from Reviewer 1 to
   Reviewer 2. It directly answers Reviewer 2's question about temporal signal
   versus generic crosscoder capacity.
2. In Reviewer 1, replace that section and the stale seed table with the
   `REVIEWER_BBBY_BLOCK` below. This keeps the robust three-seed backtracking
   result and the independently motivated KLiCKe task next to the reviewer's
   seed and temporal-contribution concerns.
3. In Reviewer 2, retain the parameter/FLOP and Stacked-SAE evidence, remove
   the duplicated T-SAE-width section, and replace the old percentage
   window-size table with `REVIEWER_4Z15_BLOCK`. The T-SAE-width control belongs
   in Reviewer 1 because Reviewer bbby asked for it.
4. Omit sycgen unless space remains after those direct answers. If included,
   use only `SYCGEN_BLOCK`; the current “real world” and “above both baselines
   at every window” wording fails Han's matched-budget audit.

Three current claims need correction independent of the reshuffle:

- The abstract's 40% detection and 15% inducement headlines come from
  different variants: TXC-Pro gives the strongest detection result, while
  TXC-base gives the strongest inducement result. Reviewer 1 should say this
  directly rather than implying one base-TXC setting achieves both.
- “We provide three-seed results and confirm the relative rankings do not
  change” is too broad. The new replication establishes three-seed
  backtracking **detection**; the submitted 300K steering result and some other
  headline cells remain incomplete.
- Reviewer 2's “TXC outperforms everywhere outside EM” is stronger than its
  own table supports. Backtracking is the clean matched-capacity result;
  sparse probing is close, and HH-RLHF has an untrained floor above both
  trained cells.

Paste-ready abstract clarification for Reviewer 1:

> The two percentages summarize different variants in the same backtracking
> case study: TXC-Pro gives the strongest sparse-probe detection result, while
> TXC-base gives the strongest causal inducement result. No single submitted
> variant establishes both headlines, and we will revise the abstract to state
> that explicitly.

## Reviewer bbby: replacement evidence block

Replace the stale sycgen headline and the claim that all headline experiments
now have three seeds with the following:

<!-- BEGIN REVIEWER_BBBY_BLOCK -->
### Robustness and temporal contribution

We ran a separate matched 20K-step backtracking replication at
\(T\in\{1,2,4,6,10\}\) with three independently trained dictionary seeds. At a
fixed 32-feature question-grouped probe budget, ordered TXC detection AP rises
from \(0.218\pm0.005\) at \(T=1\) to \(0.255\pm0.008\) at \(T=10\); every
seed's \(T=10\) endpoint exceeds its \(T=1\) endpoint. Applying each
ordered-trained probe after deterministic
within-window shuffling gives \(0.231\pm0.009\) at \(T=10\), for an
ordered-minus-shuffled gap of \(0.023\pm0.007\). We interpret the first result
as robust evidence for useful local context and the second as a narrower
representation-sensitivity control, since test-time permutation introduces
covariate shift.

| \(T\) | Ordered TXC AP | Shuffled TXC AP | Positional SAE AP |
|---:|---:|---:|---:|
| 1 | \(0.218\pm0.005\) | \(0.218\pm0.005\) | \(0.221\pm0.016\) |
| 2 | \(0.229\pm0.006\) | \(0.223\pm0.006\) | \(0.196\pm0.010\) |
| 4 | \(0.247\pm0.007\) | \(0.227\pm0.006\) | \(0.194\pm0.008\) |
| 6 | \(0.251\pm0.006\) | \(0.227\pm0.004\) | \(0.181\pm0.020\) |
| 10 | \(0.255\pm0.008\) | \(0.231\pm0.009\) | \(0.171\pm0.006\) |

Entries are mean \(\pm\) sample SD across dictionary seeds \(1,2,42\). This
replication tests detection robustness; the submitted 300K-step steering
result still has one completed seed, so we do not describe steering as a
three-seed result.

We also added a real-language temporal task using KLiCKe, a public corpus of
human keystroke logs [Tian et al.,
2025](https://doi.org/10.17239/jowr-2025.17.01.02). Immediately before a
trailing deletion burst, a cross-fitted sparse probe predicts whether the
writer will delete \(2,3,4,5,\) or \(6+\) model tokens from the final five
layer-10 activations. On 6,224 events from 2,510 held-out writers, the frozen
submitted \(T=5\) TXC attains 1.236 equal-writer log loss at the pre-specified
\(S=32\) feature budget, versus 1.261 for the strongest matched SAE (paired
improvement 0.0257, 95% CI [0.0100, 0.0413]). Holding the probe fixed while
shuffling or reversing its TXC inputs raises loss to 1.646 and 1.657. This
uses one frozen dictionary pair; uncertainty is over held-out writers, not
dictionary seeds.
<!-- END REVIEWER_BBBY_BLOCK -->

The positional SAE concatenates per-position TopK-SAE codes and therefore
matches the TXC's access to each window position. At \(T=6\), seed 42, a
post-hoc feature-budget sweep shows that it closes the gap and overtakes TXC
at \(S=256\). The matched \(S=32\) result is the paper's sparse-probe
comparison, but it must not be presented as probe-budget-independent
architectural dominance.

## Reviewer 4z15: replacement window-size block

<!-- BEGIN REVIEWER_4Z15_BLOCK -->
The matched-capacity Backtracking comparison is complemented by a
three-dictionary-seed window sweep:

| \(T\) | Ordered TXC AP | Shuffled TXC AP | Positional SAE AP |
|---:|---:|---:|---:|
| 1 | \(0.218\pm0.005\) | \(0.218\pm0.005\) | \(0.221\pm0.016\) |
| 2 | \(0.229\pm0.006\) | \(0.223\pm0.006\) | \(0.196\pm0.010\) |
| 4 | \(0.247\pm0.007\) | \(0.227\pm0.006\) | \(0.194\pm0.008\) |
| 6 | \(0.251\pm0.006\) | \(0.227\pm0.004\) | \(0.181\pm0.020\) |
| 10 | \(0.255\pm0.008\) | \(0.231\pm0.009\) | \(0.171\pm0.006\) |

All three dictionary seeds have higher ordered AP at \(T=10\) than at \(T=1\).
The smaller but positive ordered-minus-shuffled gap at \(T=10\),
\(0.023\pm0.007\), shows sensitivity to ordering beyond the gain shared by
longer windows. Because shuffling is a fixed-probe test-time perturbation, we
interpret it as representation sensitivity rather than a causal estimate of
unique temporal information.
<!-- END REVIEWER_4Z15_BLOCK -->

## Optional sycgen block

Use this only if there is space after the backtracking and deletion evidence.

<!-- BEGIN SYCGEN_BLOCK -->
As a separate three-seed semi-synthetic state-tracking diagnostic, we predict
tokens since a user's fixed “Are you sure?” challenge in generated multi-turn
dialogue. At matched realized sparsity, TXC and pooled SAE are
indistinguishable at \(T=2,4\); TXC is above the interpolated pooled frontier
at \(T=8\), and at \(T=16\) reaches 0.577 with \(L_0=7.82\), while the
cheapest pooled point reaches 0.486 with \(L_0=11.22\). This is evidence for a
high-window Pareto advantage, not learned temporal ordering: a randomly
initialized TXC has an equal or larger shuffle gap in 11 of 12 cells. We
therefore omit the underdetermined high-\(T\) stacked-SAE comparison and do not
call this a real-world task.
<!-- END SYCGEN_BLOCK -->

## Citation

```bibtex
@article{tian2025klicke,
  author  = {Tian, Yu and Crossley, Scott and {Van Waes}, Luuk},
  title   = {The {KLiCKe} Corpus: Keystroke Logging in Compositions for Knowledge Evaluation},
  journal = {Journal of Writing Research},
  year    = {2025},
  volume  = {17},
  number  = {1},
  pages   = {23--60},
  doi     = {10.17239/jowr-2025.17.01.02},
  url     = {https://doi.org/10.17239/jowr-2025.17.01.02}
}
```

## Evidence and exclusions

- Backtracking source:
  `purified/results/neurips_rebuttal/backtracking_window_sweep_t16/reviewer-five-point-v1/publication/`
  in commit `d9c7fc7b`.
- Deletion source:
  `purified/results/neurips_rebuttal/writing_revision_destination/frozen_dictionary_t5_v1/`
  in commit `e91228be`; the primary \(S=32\) gate was locked in
  `aa9afae1`.
- Sycgen source:
  `arxiv:figs_writeup/tab_sycgen_budget_matched.md` and
  `arxiv:figs_writeup/tab_sycgen_shuffle_matched.md`.
- Do not replace the pending 300K-step backtracking-steering seeds with the
  20K-step detection sweep; they are different estimands.
- Do not claim that sycgen beats pooled SAE significantly at \(T=2,4\), use
  its stacked-SAE curve at high \(T\), or use its shuffle gap as learned-order
  evidence.
- Do not call deletion's writer folds or bootstrap draws dictionary seeds.
- Medical-EM seed 2, Stacked-SAE EM steering, and the StruQ placeholders remain
  outside Aniket's completed GPU-free scope.
