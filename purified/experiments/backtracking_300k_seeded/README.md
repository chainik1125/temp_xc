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
