# Backtracking TopK-then-ReLU effective-support audit

Date: 2026-07-26
Scope: completed C7 backtracking sweep, protocol `2026-07-23.2`,
\(T\in\{1,\ldots,6\}\), seeds \(1,2,42\).

## Finding

The completed sweep really does use **TopK followed by ReLU**, so its nominal
sparsity is an upper bound rather than a guaranteed support size. For TXC,
`TXCBase.encode` first selects the largest \(20T\) preactivations and then
zeros selected values that are nonpositive. The matched TopK SAE does the same
with 20 selected preactivations per token. This composition is implemented in
`purified/src/temp_bench/archs/txc_base.py` and
`purified/src/temp_bench/archs/topk_sae.py`.

For a preactivation vector \(a(x)\), the realized support is

\[
k_{\mathrm{eff}}(x)
=
\sum_{j\in\operatorname{TopK}(a(x),k_{\mathrm{nom}})}
\mathbf 1[a_j(x)>0]
\leq k_{\mathrm{nom}}.
\]

The frozen full-sweep profile has \(k_{\mathrm{pos}}=20\) and
\(d_{\mathrm{SAE}}=32{,}768\). None of these budgets is clipped by dictionary
width:

| Window \(T\) | TXC nominal window support \(20T\) | SAE nominal support per token | SAE positional-stack upper bound over \(T\) tokens |
|---:|---:|---:|---:|
| 1 | 20 | 20 | 20 |
| 2 | 40 | 20 | 40 |
| 3 | 60 | 20 | 60 |
| 4 | 80 | 20 | 80 |
| 5 | 100 | 20 | 100 |
| 6 | 120 | 20 | 120 |

The TXC and positional-SAE totals are only nominally matched. If their selected
preactivations have different sign distributions, they can have different
effective support despite the same \(20T\) upper bound.

## Why the completed sweep's effective support is unrecoverable

The immutable outputs currently available contain:

| Inventory item | Local completed package | RunPod snapshot (2026-07-26) |
|---|---:|---:|
| `result.json` | 18 | 7 |
| held-out prediction `.npz` files | 725 | 10 |
| sparse latent-code caches | 0 | 0 |
| `model.safetensors` dictionaries | 0 | 0 |
| `training_state.pt` files | 0 | 0 |

The local package is
`purified/results/neurips_rebuttal/backtracking_window_sweep/full/cells`.
The read-only RunPod snapshot was checked under
`/workspace/temporal-crosscoders` on branch `neurips-aniket`. The RunPod still
has the evaluation activations at
`/workspace/artifacts/official-six/c7_backtracking/stage_a/sentence_acts_L10.npz`,
but only a partial copy of the compact completed package. That partial copy
does not change the conclusion: activations and downstream predictions alone
cannot recover dictionary preactivations or sparse supports.

This absence is expected from the production persistence policy.
`_persist_completed_result` in
`purified/experiments/backtracking_window_sweep/modal_app.py` copies only each
cell's `result.json` and held-out `predictions/`; checkpoints and code caches
live in a temporary directory and are deliberately not persisted. The 18
completed `result.json` files do not contain an `effective_l0` field. Their
prediction files contain downstream probe outputs rather than latent codes or
preactivations, so neither \(k_{\mathrm{eff}}\) nor its distribution is
identifiable from them. Checkpoint SHA-256 values prove which dictionaries
were evaluated but cannot reconstruct their weights.

Any one of the following would have made a retrospective measurement
possible:

1. each cell's TXC and SAE `model.safetensors`, which would allow re-encoding
   the held-out activation artifact;
2. the sparse CSR code caches created during evaluation, whose row counts
   directly equal effective \(L_0\); or
3. `training_state.pt`, which contains the final training batch's `l0` metric,
   though that would provide only a noisy single-batch diagnostic rather than
   the preferred held-out distribution.

None is available. Retraining would create new dictionaries rather than
measure the exact dictionaries behind the reported \(T=1,\ldots,6\) curve, so
no effective-\(L_0\) values should be attached retroactively to that curve.

## Forward fix in the isolated T16 protocol

The separately versioned T16 runner records this diagnostic without changing
the architecture:

- `run_t16.py` sets `record_effective_l0=True` for training, and `train.py`
  records nominal \(L_0\), realized \(L_0\), fill fraction, and underfill in
  checkpoint metrics;
- T16 evaluation calls `evaluate_cell(..., include_effective_l0=True)`;
- `evaluate.py` counts CSR nonzeros per held-out row and reports mean, sample
  standard deviation, minimum, maximum, fill fraction, underfilled-row
  fraction, and zero-row fraction for TXC and the SAE representations under
  each evaluated order condition.

The focused unit checks in
`purified/tests/test_backtracking_window_sweep_t16.py` verify both a synthetic
TopK-then-ReLU underfill case and the training-time
\(0\leq k_{\mathrm{eff}}\leq k_{\mathrm{nom}}\) invariant. This logging is
diagnostic only: it does not remove ReLU, change TopK selection, or alter the
completed \(T\leq6\) results.

A direct focused smoke of `sparse_effective_l0` on 2026-07-26 also passed:
two synthetic rows with nominal \(L_0=2\) and one stored nonzero each produced
mean effective \(L_0=1\), fill fraction \(0.5\), and underfilled-row fraction
\(1.0\).
