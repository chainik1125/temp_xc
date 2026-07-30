# Corrected C7 300K training cells

This directory runs the remaining paper-faithful C7 dictionary cells against
an isolated archive of historical commit
`284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3`. It does not switch branches or
reuse the current v2 architecture implementations.

The historical C7 runner seeded its NumPy window sampler but did not seed
Python or PyTorch before model initialization and `torch.randint` calls. This
runner fixes that omission, records the correction as protocol
`c7-300k-seeded-v1`, and otherwise uses the historical architecture,
`TrainingConfig`, trainer, cache-key computation, and checkpoint format.

Production cells:

- `txc_base`, width 32,768, seeds 1, 2, and 42 on three H100s;
- `tsae_paper`, width 16,384, seed 42 as a width-sensitivity control.

The already-submitted T-SAE width-32,768 seed-42 cell
(`32f27809cdf34da9`) is not rerun.

Run `setup_runtime.sh` once per pod, then `launch_grid.sh h100` or
`launch_grid.sh a40`. Both scripts fail closed unless the checkout is on
`neurips-aniket`.

After a checkpoint is complete, `detect.py` reproduces the paper's grouped
five-fold sparse-probe evaluation at
\(S\in\{1,2,4,8,16,32\}\). It refuses partial or mismatched checkpoints and
writes one compact, provenance-stamped JSON result. Steering is intentionally
kept separate: the historical full evaluator loses generated panels if the
Anthropic judge is unavailable, so it must not be launched until generation
persistence and the judge credential have both been verified.

Once the three corrected TXC detection JSONs and the T-SAE-16K sensitivity
JSON are present, `plot_detection.py` builds the reviewer package:

```bash
python purified/experiments/backtracking_300k_seeded/plot_detection.py \
  --root purified/results/neurips_rebuttal/backtracking_300k_seeded \
  --tsae16-json purified/results/neurips_rebuttal/backtracking_300k_seeded/cells/tsae_paper_d16384_seed42/detection.json \
  --output-dir purified/results/neurips_rebuttal/backtracking_300k_seeded/publication
```

The script fails closed on the training and detection protocols, historical
source commit, source-activation artifact hash, cohort counts, train keys,
widths, seeds, step count, and complete \(S\)-grid. It reports the corrected
TXC-base seeds as mean \( \pm \) sample SD, with the submitted \(S=8\) probe
budget as the headline comparison. Submitted seed-42 SAE, T-SAE-32K, and
TXC-base values are rounded table-transcribed contextual references and are
never pooled with the new replication. This package is a TXC-base detection
replication; it neither reruns the submitted TXC-pro detection winner nor
supplies the still-missing multi-seed steering evaluation.
