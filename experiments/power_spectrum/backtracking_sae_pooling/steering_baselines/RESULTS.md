## Verdict

**TXC-base clearly beats the final-token SAE in this fresh, matched-20k C7
steering comparison.** Its peak backtracking inducement is **0.4590**, versus
**0.0984** for the SAE. The broader negative steering lobe also replicates:
averaged over magnitudes `{-12, -10, -8, -7, -6, -5}`, TXC-base scores
**0.1967** and the SAE scores **0.0656**.

This is the clearest causal sign of life for the TXC found in the current
investigation. It is not yet evidence that TXCs are generally useful: the
comparison is one checkpoint seed, one mined feature per architecture, and one
behavioral task. The peak is selected from 25 magnitudes and should not be
treated as a preregistered point estimate.

![Fresh and historical steering curves](results/fresh_25mag_seed42/steering_comparison.png)

## Fresh results

| Architecture | Feature | Selectivity | Peak delta-gc | Peak magnitude | Per-token residual L2 | Negative-lobe mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final-token SAE | 10668 | 0.0553 | 0.0984 | -10 | 10.000 | 0.0656 |
| **TXC-base** | **25630** | **0.1328** | **0.4590** | **-12** | **4.973** | **0.1967** |

The residual-L2 column is the norm added at each steered token,
`abs(magnitude) * decoder_norm`. The canonical hook does not normalize decoder
directions. TXC-base therefore achieves its larger effect with roughly half
the per-token residual perturbation norm of the SAE peak, not with a larger
intervention.

Both arms completed all **1,525/1,525** expected judge keys. The zero-magnitude
hook took the exact no-op path and both curves have delta-gc zero at magnitude
zero.

## Comparison with the May reference

| Architecture | Fresh peak | May peak | Fresh lobe mean | May lobe mean |
| --- | ---: | ---: | ---: | ---: |
| Final-token SAE | 0.0984 at -10 | 0.2295 at -16 | 0.0656 | 0.0355 |
| TXC-base | 0.4590 at -12 | 0.4262 at -8 | 0.1967 | 0.2432 |

The isolated SAE peak at magnitude -16 does not reproduce, so comparing only
the two maximized values overstates historical stability. The more meaningful
pattern is the negative lobe from -12 through -5: TXC beats the SAE at all six
grid points in the fresh run, and its lobe mean is directionally consistent
with May. The exact TXC optimum moves from -8 to -12, which is unsurprising for
a 61-question judged cohort but means the peak magnitude is not stable.

## Protocol and checks

- Frozen canonical C7 implementation at commit `1c213513f`.
- Seed 42 checkpoints trained for exactly 20,000 steps: TopK SAE train key
  `f437e623fabc37ec`; TXC-base train key `08fe3af07682fab4`.
- Same 61-question Phase-1 cohort for both arms: 31 truly wrong and 30
  originally correct examples.
- Cut-and-continue at 25% of the original reasoning trace, with at most 1,024
  new tokens and the full canonical 25-magnitude grid.
- Feature selection uses the canonical positive-minus-negative activation
  statistic. Generation, judging, and delta-gc computation delegate directly
  to the frozen reference implementation.
- The exact zero-hook no-op, checkpoint metadata, cohort composition, and live
  judge call were all gated before generation.
- No failed judge labels or API errors were observed. Raw judge output remains
  on the stopped persistent RunPod volume; compact result and preflight files
  are stored here.

## Interpretation and next decision

The result supports a narrow story: a window-derived TXC feature is a much
better *causal steering direction* for backtracking than the top final-token
SAE feature, even though pooled SAE latents recover much of TXC-base's
detection performance. Representation detection and causal control are not
interchangeable here.

The immediate follow-up should be small and adversarial rather than another
architecture sweep:

1. Repeat only the negative lobe on two fresh seeds and mine features without
   looking at their steering outcomes.
2. Add the pooled-SAE feature selected by the completed detection experiment,
   using residual-L2-matched magnitudes.
3. Test whether TXC feature 25630 changes backtracking specifically, rather
   than generic answer correctness, verbosity, or degeneration.

If the lobe survives those controls, backtracking becomes a legitimate
task-specific sign of life. If it does not, this run should remain a promising
single-seed result rather than rescue the general TXC hypothesis.

## Compute and artifacts

- Primary H100 run: 4,476 seconds at $2.99/hour, estimated **$3.72**.
- Brief artifact-recovery restart: 130 seconds, estimated **$0.11**.
- Total steering GPU cost: estimated **$3.83**.
- Pooling screen plus steering: estimated **$4.59** RunPod compute.
- API judge cost is not available from the runner and is excluded.
- Compact machine-readable results: `results/fresh_25mag_seed42/summary.json`.
- Checkpoint and runner provenance: `results/fresh_25mag_seed42/provenance.json`.
- Spend record: `results/fresh_25mag_seed42/spend.json`.
