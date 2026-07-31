## TXC decision sprint: learned SAE-trajectory control

**Status:** frozen before the RunPod result. The source commit used by the
remote supervisor is the operational freeze.

## Question

The strongest natural-language TXC result is the DailyDialog turn-length trend
at GPT-2 hidden state 7 and \(T=32\). A six-seed, post-hoc-extended result
reported grouped-dialogue ridge \(r=0.2496\), versus roughly \(0.03\) to
\(0.05\) for a tokenwise SAE/T-SAE.

That comparison omitted the most important control: a learned temporal
consumer of the tokenwise SAE trajectory. This sprint asks whether TXC still
adds value once that control has the same one-code, \(L_0\leq8\) output
bottleneck.

DailyDialog is used under its CC BY-NC-SA 4.0 licence for research.

## Frozen cells

- Datasource: `dial_real_ttrend_gpt2_l7`.
- Model substrate: GPT-2 hidden state 7, 4,111 sequences of 128 tokens.
- Fresh paired seeds: 9, 10, 11, 12, 13, and 14.
- Dictionary width: 2,048.
- Output sparsity: at most 8 active features per 32-token window.
- Training exposure: 8,000 steps and 1,024 reconstructed token positions per
  step for every stage.
- TXC: `txc_batchtopk_post`, \(T=32\), 8,000 steps.
- Shared SAE: `batchtopk_sae`, 8,000 steps.
- Learned SAE-trajectory controls:
  - rank 0: per-feature learned temporal encoder and decoder;
  - rank 256: the same model plus low-rank cross-feature encoder and
    position-specific decoder residuals. This is the frozen primary control.
- Fixed controls: last-token SAE, top-8 mean pool, and top-8 max pool.
- Negative controls: untrained trajectory adapter and anchor-fixed history
  reversal (the target-aligned final token stays final).
- Headline: trace-grouped ridge Pearson \(r\), using 8,192 windows per half
  and the already frozen 13-value ridge grid.

The learned controls start from the shared SAE decoder. The primary rank-256
control can additionally learn low-rank, position-specific residual decoder
directions, so it is not restricted to rescaling one frozen direction. The
controls compress 32 ordinary SAE code vectors to one top-8 code and
reconstruct all 32 raw activation vectors. This gives the baseline *more*
total optimization than TXC (an SAE stage followed by an adapter stage), but
fewer learned temporal parameters. That is intentional: this is a practical
value-of-architecture test, not a claim that the optimization budgets are
identical.

Every model has a maximum output support of eight. Realized support is
reported. A TXC sign-of-life claim additionally requires the primary
adapter's and TXC's mean realized supports to differ by at most 1.5.

## Decision rule

Let \(r_T\) be paired-seed TXC performance and \(r_A\) the stronger learned
SAE-trajectory adapter.

- **Stop the general TXC programme** if the primary adapter is statistically
  non-inferior: the upper endpoint of the paired 95% bootstrap interval for
  \(r_T-r_A\) is at most \(0.03\). The conclusion is about TXC as a
  general-purpose temporal representation; it does not erase narrow
  online-compression results.
- **Initial real-task sign of life** requires all of:
  - TXC beats the stronger adapter by at least \(0.05\) mean \(r\);
  - the paired 95% bootstrap interval for TXC minus adapter is above zero;
  - trained TXC beats its untrained control and the learned adapter is not
    degenerate;
  - reversing the window reduces TXC recovery by at least \(0.03\).
  - TXC and the primary adapter have mean realized-support gap at most 1.5.
- Any middle outcome is inconclusive and must be reported as such. It cannot
  be promoted to a sign of life.

## Spend and durability

- Venue: one RunPod H100 at $2.99/hour.
- Hard wall-clock timeout: 8 hours, at most $23.92 for this sprint.
- Prior mistaken C7 dispatch cost approximately $2.13, so the combined work
  remains below the user's $50 cap.
- SAE/TXC checkpoints are atomic at the end of each short 8,000-step stage;
  adapters checkpoint every 1,000 steps. The supervisor is detached from SSH
  and stops the pod after completion, failure, or timeout.
