## Matched 20k steering baselines

This folder measures fresh C7 backtracking steering curves for the two controls
needed before testing pooled SAE latents:

- `topk_sae`: the final-token SAE baseline, train key `f437e623fabc37ec`.
- `txc_base`: the matched TXC-base baseline, train key `08fe3af07682fab4`.

Both are seed 42, trained for 20,000 optimizer steps on the same Llama-3.1-8B
layer-10 activation source. Evaluation uses the frozen 31 truly-wrong plus 30
originally-correct MATH-500 cohort, cut25 continuation protocol, greedy
DeepSeek-R1-Distill-Llama-8B generation, the canonical 25-point magnitude grid,
and the canonical Sonnet 4.6 backtracking judge.

The runner imports and calls the reference branch's
`run_arch_evaluation`. It does not reimplement prompt templating, generation,
steering, judging, or delta-gc. Before spending on the sweep it verifies:

- exact checkpoint metadata and 20k train key;
- exact 31 plus 30 cohort size;
- byte-identical zero-magnitude hook behavior;
- one successful live judge call;
- successful judge coverage for all 1,525 arm-panel keys before accepting a
  result.

The final-token SAE is intentionally measured before any pooled-latent steering.
The pooled experiment can then change only feature aggregation while retaining
this evaluation protocol.

`run_runpod.sh` executes the two arms in that order under a four-hour hard
timeout. Its heartbeat, logs, partial judge rows, and completed arm results are
all on the persistent volume. Its exit trap flushes the volume and requests pod
shutdown, including after an error or timeout.
