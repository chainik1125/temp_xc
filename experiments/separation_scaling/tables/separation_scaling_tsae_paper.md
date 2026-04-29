# Paper-faithful T-SAE in separation_scaling

Same transformer setup as `config.yaml` (σ=1e-3, d=64, ctx=128, 20k steps), same generator, same probe. Only difference: T-SAE arch with paper-faithful matryoshka split + 25k training steps instead of the original `Temporal BatchTopK SAE` entry which used 50/50 split + 2k steps.

## Best single-feature R² per arch (across δ sweep)

| δ | TopK | TXC | MatTXC (main cfg) | MLxC | TFA | TFA-pos | T-SAE old (50/50, 2k) | **T-SAE paper (20/80, 25k)** |
|---|---|---|---|---|---|---|---|---|
| 0    | -0.001 | -0.001 | -0.010 | -0.000 | -0.001 | -0.001 | -0.001 | **+0.001** |
| 0.05 | -0.001 | -0.000 | -0.012 | -0.000 | -0.000 | +0.000 | -0.000 | **+0.000** |
| 0.10 | +0.002 | +0.030 | -0.000 | +0.022 | +0.001 | +0.002 | +0.007 | **+0.005** |
| 0.15 | +0.050 | +0.244 | +0.263 | +0.092 | +0.052 | +0.078 | +0.109 | **+0.285** |
| 0.20 | +0.069 | +0.209 | +0.421 | +0.148 | +0.190 | +0.187 | +0.079 | **+0.611** |

## Bayes-optimal R² ceiling at δ=0.20

The Bayes-optimal R² ceiling is **per-component asymmetric** because at δ=0.20 the components are deliberately separated: comp-0 is sticky (x=0.05, a=0.8), comp-1 moderate, comp-2 diffuse. From `r2_ceiling.json`:

| component | ceiling (mean over t) | ceiling (final t) |
|---|---|---|
| 0 (sticky) | **0.866** | **0.998** |
| 1 (moderate) | 0.226 | 0.309 |
| 2 (diffuse) | 0.247 | 0.310 |

Reported `best_single_r2` = max across components, so it tracks comp-0 in practice. Gap recovered = `best_single_r2 / 0.866`.

## Important caveat: this is NOT a fair vs-MatTXC comparison

The `MatTXC` numbers above are from the **main-config** `config.yaml` (`fixed_k_total=10`, `matryoshka_widths=[8,16,32,64,128]`, `inner_weight=10`, `temporal_steps=1500`) which is *not* the configuration that maximises MatTXC's single-feature R². The companion **protocol sweep** (`tables/protocol_sweep.md`) hits substantially higher single-feature R² for MatTXC under different sparsity settings:

| MatTXC config (protocol sweep) | δ=0.15 single | δ=0.20 single | gap recovered (δ=0.20) |
|---|---|---|---|
| batchtopk k=10 | 0.43 | **0.81** | 93.5% |
| batchtopk k=20 | 0.51 | 0.80 | 92.4% |
| topk baseline | 0.57 | 0.81 | 93.5% |
| batchtopk k=4 | 0.59 | 0.73 | 84.3% |
| Main-config MatTXC | — | 0.421 | 48.6% |
| **T-SAE paper (this run)** | 0.285 | **0.611** | **70.6%** |

So the honest reading is:

- **Paper-faithful T-SAE substantially beats main-config MatTXC** (0.611 vs 0.421 at δ=0.20). This is the comparison that lives inside a single `config.yaml` run with both archs at their default settings — and at default settings, the matryoshka prefix in T-SAE pulls more single-feature signal than MatTXC's particular config does.
- **Paper-faithful T-SAE is below protocol-tuned MatTXC** (0.611 vs 0.81). With `batchtopk k=10` or `topk baseline`, MatTXC reaches ~94% of the comp-0 ceiling, vs T-SAE Paper's ~71%.

So the "T-SAE wins" story really only holds against the *under-tuned* MatTXC entry that lives in the headline `tables/separation_scaling.md` table. The right next experiment, if we want a clean architectural verdict, is to run T-SAE Paper across the same protocol-sweep grid (varying sparsity method + k) so both archs are at *their* best.

## What this tells us about the existing T-SAE benchmark entry

The existing "Temporal BatchTopK SAE" entry in the original `config.yaml` used `group_fractions=[0.5, 0.5]` + only 2000 training steps. The 50/50 split makes the matryoshka pressure trivial (both groups same size and weight, so prefix loss = full-width loss), and 2k steps was too short to converge. Effectively the existing entry was BatchTopK SAE with a temporal contrastive loss — not the paper architecture.

Switching to the paper-default `[0.2, 0.8]` split + 25k steps recovers most of the architecture's compression. The 20% prefix is where ergodic-component identity gets concentrated. So the existing benchmark entry was *substantially* under-representing T-SAE — by 0.611 − 0.079 = +0.532 in single-feature R² at δ=0.20.

This also matches the [coupled-features T-SAE result](../../../docs/dmitry/case_studies/coupled_features/summary.md): under paper-faithful settings, T-SAE compresses global structure into a small set of features more reliably than any window arch tested at the same default config.

## Caveats

1. **Single seed (42).** The original benchmark used the same single seed; we matched it. No replication-based variance.
2. **σ=1e-3 transformer init**, as in the original. Per the σ sweep finding (`tables/mup_study.md`), TXC's single-feature R² benefits from σ=2e-2; MatTXC and presumably T-SAE paper have less to gain since they already concentrate well at σ=1e-3.
3. **Only the T-SAE arch was rerun.** The other 6 archs' numbers in the table above are reused from the original `config.yaml` run — i.e. the *under-tuned* MatTXC. See protocol sweep for tuned MatTXC.
4. **No protocol sweep on T-SAE Paper.** We have one config; MatTXC has a 30+-cell sparsity sweep. A fair architectural comparison requires sweeping T-SAE Paper too.

## Files

- `config_tsae_paper.yaml` — minimal config with only T-SAE Paper arch
- `results_tsae_paper/cell_delta_*/results.json` — per-cell raw output
- `results_tsae_paper/combined.json` — combined per-arch×δ summary
