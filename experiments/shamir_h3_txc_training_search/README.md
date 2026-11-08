## h=3 TXC training search

Completed successfully on 2026-11-07 at 06:52:33 UTC, using 207.6 seconds
of wall time on GPU5. All 16 screening jobs and 30 confirmation jobs completed.
The private MPS instance was stopped. No additional training was launched
during the subsequent CPU-only audit.

Validation selected `data_init_centered` at 6000 steps: initialize the decoder
from unlabeled training windows, center encoder inputs by the decoder bias,
and use learning rate 0.001 with a 300-step warmup and cosine schedule whose
horizon remains 24000 steps. Early AuxK revival was not selected. The mean
validation accuracy over W=6 and W=10 was 12.56%, versus 10.46% for the
6000-step baseline.

Held-out accuracy, mean ± sample SD across all three seeds:

| Window W | Original training | Selected training |
| --- | --- | --- |
| 3 | 9.16 ± 0.65% | 9.14 ± 0.31% |
| 4 | 9.27 ± 0.16% | 10.56 ± 0.29% |
| 5 | 8.80 ± 0.33% | 11.47 ± 0.48% |
| 6 | 9.23 ± 0.54% | 11.42 ± 1.06% |
| 10 | 11.74 ± 0.54% | 13.36 ± 0.70% |

The selected recipe improves W=10 by 1.63 ± 0.33 percentage points in the
paired seed comparison. W=10 exceeds W=4 for each seed. Recovery remains
low, and the mean curve is not strictly monotonic: W=5 to W=6 declines by
0.05 percentage points. This is a modest gain rather than a successful
high-accuracy h=3 demonstration.

The CPU audit verified all 159 frozen source hashes, six dataset hashes,
four-way episode separation, 78 model checkpoints, and 114 saved probe
prediction files. All held-out evaluations occurred after selection was
frozen. W=3 is consistent with 1/11 chance; raw linear and shuffled-label
controls are near chance; symbolic and observation interpolation oracles
achieve 100% at W≥4. The audit and all numeric summaries are preserved in
the local `results/` directory; models and datasets remain on simplex2.

## Protocol

Requested exploratory search for stronger long-window recovery at h=3. The
generator remains q=11, d=64, sigma=0.1, T_chain=16; TXC width is 2048 and the
window-level TopK budget is fixed at k=5. No outcome is forced to be monotonic.

Eight recipes vary learning rate, constant versus cosine schedule, encoder
centering, initialization from unlabeled training windows, and earlier AuxK
dead-feature revival. Each is screened on seed 0 at W=6 and W=10 with 6000,
12000, and 24000-step checkpoints. The objective is mean validation accuracy
over those two windows; ties prefer fewer steps, then recipe name. This
objective is fixed before training. The cosine schedule always has a 24000-step
horizon, including checkpoints selected earlier.

Data uses the repository's original generator with four disjoint episode
splits of 4096 each: representation training, probe training, validation, and
test. Probe sampling uses one window per episode. Test tensors are kept in a
separate file and evaluation requires a written selection record. This is new
four-way data; the old three-way baseline is rerun on the same four-way data
for comparison. Repeated coefficient values from distinct iid episodes are
allowed, as required by the generative distribution.

After validation selection, the selected training recipe and original
6000-step baseline are evaluated at W=3,4,5,6,10 for seeds 0,1,2. W=3 is a
negative control. Additional controls are raw-observation linear probes,
label-shuffled W=10 probes, and symbolic and observation interpolation
oracles. All candidates, checkpoint metrics, model/probe weights, split IDs,
and exact source hashes persist. No test result is used to revise selection.

The original Adam linear probe is unchanged (1500 steps, learning rate 0.01,
weight decay 0.0001, batch size 256). Model training uses batches of 128,
decoder unit normalization, decoder-parallel gradient removal, and gradient
clipping at 1. Checkpoint probing preserves the model-training RNG state.

The detached supervisor has a one-hour wall budget on simplex2 GPU5 only,
with 16 workers and a private CUDA MPS instance. It terminates only its own
worker process groups if the budget expires. GPU5 UUID:
`GPU-0216af65-2b0e-3798-454e-1208593efcc3`.

Remote run directory:
`/data/users/dmitry/shamir_h3_txc_training_search_20261107T0651/`.

Status is in `full/progress.json` and `full/jobs/*/status.json`; frozen
validation selection is `full/selection.json`; final results are
`full/confirmation_summary.json`. `full/completion.json` exists only after
the full confirmation completes. `launcher_exit.json` records process exit.

```bash
ssh simplex2 'cat /data/users/dmitry/shamir_h3_txc_training_search_20261107T0651/full/progress.json'
```

The study changes training choices, not the generator, nominal sparsity,
dictionary width, evaluation metric, or labels. It does not establish a
monotonic benefit until the held-out window sweep supports that conclusion.
