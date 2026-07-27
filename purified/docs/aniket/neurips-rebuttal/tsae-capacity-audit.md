# C7 T-SAE capacity audit

**Status:** the submitted backtracking T-SAE result is already
dictionary-width matched to the SAE and TXC results. Do not launch a nominal
“same dictionary size” rerun: it would repeat the existing seed-42 cell.

## Why the reviewer reasonably inferred a mismatch

The general architecture paragraph in `paper/appendix.tex` says that T-SAE has
a locked \(d_{\mathrm{SAE}}=16{,}384\), matching its original Gemma setting.
The C7-specific paragraph later states that all six backtracking architectures
use \(d_{\mathrm{SAE}}=32{,}768\). The historical locked registry resolves this
apparent contradiction with a component override:

```yaml
tsae_paper:
  hparams:
    d_sae: 16384
  per_component_hparams:
    c7:
      d_sae: 32768
```

The reviewer question is therefore best treated as a documentation ambiguity,
not evidence that the reported C7 baseline used a smaller dictionary.

## Artifact-level verification

The submitted T-SAE detection result is the C7 seed-42, 300K-step,
batch-1024 cell with training key `32f27809cdf34da9`. Recomputing the
repository's canonical training-key hash gives:

| T-SAE width | Recomputed training key |
|---:|:---|
| \(32{,}768\) | `32f27809cdf34da9` |
| \(16{,}384\) | `b97e3c00153a5271` |

The leaderboard row for `32f27809cdf34da9` reports
\(\mathrm{PR\mbox{-}AUC}@S{=}32=0.24481534796544918\), shown as \(0.245\) in
the paper and the rebuttal plot. This pins the reported result to the
32,768-feature configuration independently of the prose.

The historical source of truth is
`origin/extended-300k:purified/configs/locked_archs.yaml` plus
`origin/extended-300k:purified/results/leaderboard.jsonl`. The current
`purified/configs/archs.yaml` has regressed to a 16,384-feature T-SAE default
without a backtracking override, so a naive current-branch rerun would be the
under-capacity experiment the reviewer feared.

## Width matching is not parameter matching

At \(d_{\mathrm{in}}=4096\) and \(d_{\mathrm{SAE}}=32{,}768\), the TopK SAE
and T-SAE each have

\[
2d_{\mathrm{in}}d_{\mathrm{SAE}}+d_{\mathrm{SAE}}+d_{\mathrm{in}}
=268{,}472{,}320
\]

trainable parameters. A \(T=5\) TXC-base has separate encoder and decoder
slabs at every position and therefore has

\[
2T d_{\mathrm{in}}d_{\mathrm{SAE}}+d_{\mathrm{SAE}}+T d_{\mathrm{in}}
=1{,}342{,}230{,}528
\]

trainable parameters. The existing comparison matches dictionary width and
sparsity convention, but TXC-base has approximately \(5\times\) the stored
parameters. Its dense cost for one five-token window is also \(5\times\) one
T-SAE token; applying T-SAE independently to the same five tokens has the same
leading-order encoder/decoder multiply-add count. A sliding TXC evaluated at
every token incurs additional overlap, so the reported inference cost must
state the stride and support convention. The reviewer explicitly asks for
parameter count and inference cost, so the response should state these
distinctions rather than imply that equal dictionary width is equal capacity.

## Recommended rebuttal action

1. Clarify that \(16{,}384\) is the T-SAE default for the original Gemma
   setting, while C7 overrides every architecture to
   \(d_{\mathrm{SAE}}=32{,}768\).
2. Report the architecture parameter counts and inference-cost scaling.
3. Keep the existing \(0.245\) T-SAE point; it is already the requested
   width-matched result.
4. Run a new experiment only if the team wants either:
   - seed-1 and seed-2 replications of the 300K T-SAE cell, or
   - a parameter-matched T-SAE ablation, which would require roughly
     \(5\times\) the dictionary width and answers a different fairness
     question.

Suggested response sentence:

> We apologize for the ambiguity: \(16{,}384\) is the default T-SAE width in
> its original Gemma setting, but the backtracking experiment overrides all
> methods, including T-SAE, to \(d_{\mathrm{SAE}}=32{,}768\). We will clarify
> this in the appendix and add parameter-count and inference-cost comparisons;
> the reported T-SAE result already uses the matched 32,768-feature
> dictionary.
