# Updated reviewer-response handoff

These drafts replace, rather than append to, Dmitry's current response files.
They were prepared against
[`dmitry-txcwins-10h@c7c301e2`](https://github.com/chainik1125/temp_xc/tree/c7c301e22a582e631e626f49a41de38cf8bf7978/docs/dmitry/reviewer_responses).

| Reviewer | Paste-ready draft | Body characters |
|---|---|---:|
| bbby | [`reviewer-1-bbby.md`](reviewer-1-bbby.md) | 4,703 |
| 4z15 | [`reviewer-2-4z15.md`](reviewer-2-4z15.md) | 3,841 |
| EAxU | [`reviewer-3-eaxu.md`](reviewer-3-eaxu.md) | 4,642 |

The character counts cover only the text between `## Paste-ready response` and
`## Internal handoff notes`. Each body is below the 5,000-character OpenReview
limit.

## Proposed allocation

- Reviewer bbby gets the three-seed Backtracking detection replication,
  shuffle sensitivity, the omitted Stacked-SAE cells, the KLiCKe deletion
  result, the seed/checklist correction, the abstract clarification, and the
  direct T-SAE-width answer.
- Reviewer 4z15 gets the full capacity argument: Stacked SAE and parameter
  accounting, the Shamir task, the absolute three-seed window table, and the
  negative results that delimit the claim.
- Reviewer EAxU gets a complete replacement for the placeholder: a concise
  empirical update, main-text TXC/TXC-Pro definitions, corrected citations and
  title, the SAE-arditi definition, the F.1/F.13 merge, and the Colab fix.

## Claims deliberately excluded

- The three-seed 20K replication establishes Backtracking **detection**
  robustness; it does not make the submitted 300K steering result
  three-seed.
- The fixed-probe shuffle test measures representation sensitivity under
  covariate shift; it is not a causal decomposition of uniquely temporal
  information.
- The submitted Backtracking T-SAE already used
  \(d_{\mathrm{SAE}}=32{,}768\). The problem was ambiguous appendix wording,
  not an under-width headline baseline.
- Reviewer EAxU's draft defines the TXC implementation's
  \(\mathrm{TopK}(\mathrm{ReLU}(\cdot))\) activation. The current main-text
  sentence that calls TXC's activation BatchTopK must be synchronized before
  posting the revision.
- The KLiCKe result uses one frozen dictionary pair. Its interval is over
  held-out writers, not dictionary seeds.
- Sycgen is omitted because the matched audit supports a high-\(T\) Pareto
  point but does not establish learned order sensitivity.

## Slack message

> I drafted complete replacements for all three responses against your current
> `c7c301e2` push. My proposed reshuffle is: (1) Reviewer 1 gets the three-seed
> Backtracking detection + shuffle result, omitted Stacked-SAE cells, KLiCKe
> deletion result, seed/checklist correction, abstract clarification, and the
> direct T-SAE-width answer; (2) Reviewer 2 keeps the capacity/compute argument,
> gets the full Shamir result and absolute three-seed window table, and now
> explicitly treats sparse probing/EM/HH-RLHF as limits rather than wins; (3)
> Reviewer 3 gets a full response covering the empirical scope, main-text
> definitions, citations/title collision, SAE-arditi, F.1/F.13, and the Colab
> issue.
>
> I removed the stale claim that all headline results now have three seeds:
> only the new 20K Backtracking detection replication does, while the submitted
> 300K steering result remains one seed. I also describe shuffle as a
> fixed-probe sensitivity test, clarify that Backtracking T-SAE was already
> width-matched at 32,768, keep KLiCKe labeled as one dictionary pair, and leave
> sycgen out because its matched audit does not establish order sensitivity. I
> also use the TXC implementation's TopK(ReLU) definition in Reviewer 3, so
> the current main-text BatchTopK sentence needs to be synchronized. All three
> paste-ready bodies are under 5,000 characters.
