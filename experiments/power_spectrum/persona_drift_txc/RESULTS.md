## Initial 10k-step Step-7 result

The initial persona-drift gate is negative. There is weak short-horizon
temporal information in the raw activation history, but the 10k-step TXCs do
not preserve it better than a current-turn SAE. Steering should therefore wait
for a smaller-dictionary or early-stopped representation ablation.

## Data and reference checks

- Subject model: Qwen3-32B, post-MLP residual at layer 32.
- Corpus: 400 fixed-script conversations, 100 in each of coding, philosophy,
  therapy, and writing.
- Each conversation has 15 user/assistant turns.
- Persona-held-out split: 240 train, 80 validation, 80 test conversations.
- Activation tensor: \(400\times15\times5120\).
- No conversation was truncated; final context lengths were 1,742–2,008
  tokens.
- Maximum discrepancy from the Assistant-Axis reference projection was
  \(6.68\times10^{-5}\).
- Mean Assistant-Axis score moved from -9.87 on the first assistant turn to
  -25.29 on the last, a mean change of -15.42.
- The released full-transcript reference check recovered the expected
  capped-versus-unsteered ordering in all three published domains.

The generated corpus and compact provenance files are retained under
`results/corpus/` and `results/reference/`. The 59 MB activation tensor and
multi-GB learned weights remain in the stopped RunPod volume rather than the
Git repository. The generated corpus and row-level probe predictions are
stored as lossless `.xz` archives to respect the repository's 1 MB
per-file limit; their uncompressed hashes are recorded in the run manifest.

## Representation health

All models used 8,192 latents, nominal BatchTopK \(k=20\), 10,000 optimizer
steps, and seed 42.

| Model | Parameters | Best val NMSE (step) | Final val NMSE | Test NMSE | Realized \(L_0\) | Never fired, full corpus | Never fired, train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAE | 83.9M | 0.2445 (2,851) | 0.2656 | 0.2732 | 27.04 tokens | 5,324 | 5,765 |
| T-SAE | 83.9M | 0.2524 (2,501) | 0.2657 | 0.2754 | 42.83 tokens | 1,072 | 1,498 |
| TXC W=4 | 335.6M | 0.3933 (3,701) | 0.4300 | 0.4748 | 29.82 windows | 828 | 5,078 |
| TXC W=8 | 671.1M | 0.4420 (1,801) | 0.5329 | 0.5725 | 29.31 windows | 765 | 6,468 |

The explicit one-percent tail criterion marks all four curves as plateaued.
Their tail relative improvements are negative: -2.01% for SAE, -0.99% for
T-SAE, -2.99% for TXC W=4, and -2.40% for TXC W=8. Thus the loss is not merely
still descending slowly; validation reconstruction has worsened.

The TXC dead counts use one shared code per window, whereas SAE and T-SAE use
one code per token. They should not be read as directly matched sparsity
statistics. The large difference between TXC train-only and full-corpus dead
counts also shows substantial held-out activation of latents that training
examples rarely or never selected.

At step 2,001, W=8's first AuxK update caused a transient validation-NMSE spike
to about 1.95. It recovered quickly, but never recovered its pre-AuxK best of
0.442. This and the much earlier validation optimum make the final W=8
checkpoint a deliberately harsh test of the stipulated 10k schedule.

The complete curves and dead-latent bars are in
[training_diagnostics.png](results/training/training_diagnostics.png). Exact
per-model health JSON and training logs are in `results/training/`.

## Primary Step-7 result

The predeclared target is the minimum future Assistant-Axis score over the next
four assistant turns, using an eight-turn history. Every nontrivial probe is
conditional on current axis score, normalized turn, domain, and latest user
message. Ridge regularization is selected only on validation personas.

| Predictor | Held-out test \(R^2\) | RMSE |
| --- | ---: | ---: |
| Axis + turn + domain | 0.8899 | 2.793 |
| Axis + turn + domain + latest user | 0.8250 | 3.521 |
| Raw current activation | 0.9024 | 2.629 |
| Raw activation history | 0.9039 | 2.608 |
| SAE | 0.8465 | 3.298 |
| T-SAE | 0.8513 | 3.245 |
| TXC W=8 | 0.7025 | 4.590 |
| SAE + TXC W=8 | 0.7684 | 4.050 |
| T-SAE + TXC W=8 | 0.7664 | 4.067 |

The conversation-bootstrap nested comparisons are:

| Comparison | Mean \(\Delta R^2\) | 95% CI | Gate |
| --- | ---: | ---: | --- |
| Raw history - raw current | +0.0018 | [-0.0131, +0.0167] | Not supported |
| SAE + TXC - SAE | -0.0773 | [-0.1320, -0.0360] | Opposite direction |
| T-SAE + TXC - T-SAE | -0.0857 | [-0.1384, -0.0407] | Opposite direction |
| TXC - SAE | -0.1454 | [-0.2274, -0.0751] | Opposite direction |

Adding the complete causal user-message history does not rescue the result:
raw-history gain is +0.0076 [-0.0068, +0.0228], while SAE+TXC gain is -0.0162
[-0.0369, -0.0009].

The published safe threshold is not calibrated to this new corpus: every test
row breaches it, giving breach prevalence 1.0. AUPRC is consequently undefined.
This does not affect the continuous \(R^2\) gate, but binary breach claims
cannot be made from this run.

## Exploratory temporal signal

Raw history does contain a small signal at shorter horizons:

| Window | Horizon | Raw-history \(\Delta R^2\) over raw current | 95% CI |
| ---: | ---: | ---: | ---: |
| 4 | 1 | +0.0095 | [+0.0005, +0.0186] |
| 4 | 2 | +0.0175 | [+0.0086, +0.0270] |
| 4 | 4 | +0.0133 | [+0.0042, +0.0236] |
| 8 | 1 | +0.0183 | [+0.0052, +0.0326] |
| 8 | 2 | +0.0193 | [+0.0048, +0.0360] |
| 8 | 4 | +0.0018 | [-0.0131, +0.0167] |

None of the corresponding SAE+TXC comparisons has a positive lower confidence
bound. At W=8, horizons 2 and 4 are significantly negative. The benchmark
therefore has some local temporal predictability, but the current learned TXC
is the failure point.

## Interpretation and reassessment

The cleanest reading is:

1. Assistant-Axis position and the current residual stream already predict
   future position extremely well.
2. Raw history adds a small amount of information over one to two turns, and
   under W=4 even at the four-turn target.
3. The oversized TXCs overfit early and compress away rather than expose that
   incremental signal at the requested 10k endpoint.
4. This run does not justify proceeding to steering.

The smallest useful next experiment is to train only SAE, TXC W=4, and
possibly TXC W=8 with 1,024–2,048 latents, select checkpoints by fixed
validation NMSE, and rerun the frozen nested probe. The binary target should
either be calibrated from the training distribution before looking at test
labels or the corpus should be regenerated to straddle the published safe
threshold. More held-out personas and counterbalanced user-message schedules
are required before making a broad persona-generalization claim.

## Artifacts

- [Step-7 plot](results/future_drift_probe.png)
- [Training diagnostics](results/training/training_diagnostics.png)
- [Primary decision](results/step7_gate.json)
- [Probe summary](results/probe_summary.csv)
- [Bootstrap comparisons](results/probe_bootstrap_deltas.csv)
- [Row-level predictions](results/probe_predictions.csv.xz)
- [Run manifest](results/run_manifest.json)

All uncertainty intervals are conditional on one generated corpus and one
representation seed. The test set holds out one persona style in each domain,
so its bootstrap supports generalization across topics/conversations within
those four styles, not across a broad population of personas.
