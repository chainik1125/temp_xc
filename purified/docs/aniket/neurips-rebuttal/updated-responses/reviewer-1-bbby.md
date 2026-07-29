# Updated response to Reviewer bbby

_Drafted against Dmitry's `dmitry-txcwins-10h` commit `c7c301e2`._

## Paste-ready response

We thank the reviewer for recognizing the novelty of the architecture and the
value of the synthetic benchmark. We agree that the submission did not
separate evidence for useful context, temporal ordering, cross-position
sharing, and generic capacity clearly enough.

**Temporal contribution and robustness.** We added a matched 20K-step
backtracking **detection** replication over
\(T\in\{1,2,4,6,10\}\) with three independently trained dictionary seeds. At a
fixed 32-feature, question-grouped probe budget, ordered TXC AP rises from
\(0.218\pm0.005\) at \(T=1\) to \(0.255\pm0.008\) at \(T=10\). Every seed's
\(T=10\) endpoint exceeds its \(T=1\) endpoint; the paired increase is
\(0.037\pm0.009\). Applying each ordered-trained probe after deterministic
within-window shuffling gives \(0.231\pm0.009\) at \(T=10\), for an
ordered-minus-shuffled gap of \(0.023\pm0.007\). We interpret the larger
context gain as evidence for useful local history and the smaller perturbation
gap as representation sensitivity to order, since test-time shuffling also
introduces covariate shift.

The submitted seed-42 Stacked SAE comparison will now appear in the main result
table: at the paper's \(S=8\) detection budget it reaches 0.16 PR-AUC versus
0.24 for TXC, and its steering effect is 0.25 versus 0.54 for TXC. Stacked SAE
and TXC have the same leading parameter count and dense inference cost, so this
controls generic model size while testing learned joint cross-position
features. We agree that omitting it from Fig. 4 and Table 2 obscured the
strongest relevant control.

We also added a real-language temporal task based on the public
[KLiCKe](https://doi.org/10.17239/jowr-2025.17.01.02) human-keystroke corpus.
From the final five layer-10 activations before a deletion burst, a
cross-fitted sparse probe predicts whether the writer will delete
\(2,3,4,5,\) or \(6+\) model tokens. On 6,224 events from 2,510 held-out
writers, the frozen submitted \(T=5\) TXC obtains 1.236 equal-writer log loss
at the pre-specified \(S=32\) budget, versus 1.261 for the strongest matched
SAE (paired improvement 0.0257, 95% CI [0.0100, 0.0413]). Fixed shuffling and
reversal raise loss to 1.646 and 1.657. This uses one dictionary pair, so we
present it as an additional controlled example rather than a seed-robust
architecture comparison.

**What the seed evidence does and does not establish.** The checklist was
ambiguous: the submitted 300K backtracking steering headline used one training
seed, and we will correct the checklist rather than imply otherwise. The new
three-seed result above establishes the robustness of the separate detection
replication; it does not turn the submitted steering result into a three-seed
experiment. We will state this distinction explicitly.

**Sparse probing and the synthetic control.** MLC's small lead on sparse
probing shows that aggregation helps there without identifying sequence time
as the unique cause. We will say so directly and treat sparse probing as a
negative result for TXC-specific superiority. Conversely, the Shamir
secret-recovery task in our response to Reviewer 4z15 has a formal
single-token ceiling: non-TXC methods remain at chance (at most 0.12 at
\(W=10\)), while TXC reaches 0.96. Together with the real-model controls above,
this separates a provably temporal capability from the task-dependent
empirical gains.

**Abstract wording.** The two percentages summarize different variants:
TXC-Pro gives the strongest sparse-probe detection result, while TXC-base gives
the strongest causal inducement result. No single submitted variant establishes
both headlines. We will revise the abstract to state this explicitly.

**T-SAE width.** The paper's wording incorrectly implied a uniform dictionary
width. The \(16{,}384\) value is the T-SAE default for its original Gemma
setting, but the submitted Backtracking experiment overrides all methods,
including T-SAE, to \(d_{\mathrm{SAE}}=32{,}768\). Thus its reported
PR-AUC 0.245 is already the requested width-matched result. We will make this
component override explicit rather than imply that the Backtracking baseline
was underpowered.

**Parameters and inference cost.** We add the full per-task accounting in the
appendix and summarize the backtracking settings here:

| Architecture | Parameters | Dense GFLOPs / native forward |
|---|---:|---:|
| Per-token SAE / T-SAE, 1 token | 0.27B | 0.54 |
| TFA, 5 tokens | 2.32B | 27.18 |
| MLC, 5 layers | 1.34B | 2.68 |
| Stacked SAE / TXC-base, 5 tokens | 1.34B | 2.68 |
| TXC-Pro, 10 tokens | 2.68B | 5.37 |

These count encoder-plus-decoder dense matmuls, with one multiply-add as two
FLOPs; training-only losses and sparse selection are excluded.

## Internal handoff notes

- Keep the response body below 5,000 characters.
- Do not restore the stale all-headline-three-seeds table.
- The full Shamir result belongs in Reviewer 2; Reviewer 1 only cross-references
  it.
