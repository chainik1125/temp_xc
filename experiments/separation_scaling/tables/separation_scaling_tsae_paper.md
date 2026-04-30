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

## Update: head-to-head re-run on a40_txc_1 (protocol-tuned MatTXC vs T-SAE Paper)

Re-ran both archs on identical transformer activations (same σ=1e-3 transformer ckpt, same seed, same probe) at the **protocol-winning MatTXC config** plus paper-faithful T-SAE. This reproduces the original protocol_sweep MatTXC result (within seed noise) and gives a head-to-head verdict.

Best single-feature R² per δ (results from `results_protocol_compare/`):

| δ | MatTXC bk10 | MatTXC topk-baseline | **T-SAE Paper** | comp-0 ceiling | gap recovered (best of bk/tk) | T-SAE gap |
|---|---|---|---|---|---|---|
| 0    | +0.001 | +0.001 | +0.001 | 0.000 | — | — |
| 0.05 | +0.003 | +0.003 | +0.000 | 0.292 | 1% | 0% |
| 0.10 | +0.012 | +0.012 | +0.005 | 0.554 | 2% | 1% |
| 0.15 | **+0.564** | **+0.564** | +0.285 | 0.730 | **77%** | 39% |
| 0.20 | **+0.851** | **+0.851** | +0.611 | 0.866 | **98%** | **71%** |

Two clean takeaways:

1. **MatTXC reproduces (and slightly exceeds) the original 0.81 protocol_sweep result** — 0.851 at δ=0.20, **98% of comp-0 ceiling**. The two sparsity methods (batchtopk k=10, topk baseline) converged to identical numbers to 4 decimal places — which means at this seed + transformer they're discovering the same solution.
2. **MatTXC strictly beats T-SAE Paper head-to-head, at every δ where signal exists.** At the most-separated cell (δ=0.20), MatTXC gets +0.24 over T-SAE Paper (0.851 vs 0.611). At δ=0.15 the gap is +0.28.

So the corrected story is:

- **Protocol-tuned MatTXC > paper-faithful T-SAE on this benchmark** under fair head-to-head conditions.
- **Paper-faithful T-SAE > main-config MatTXC** (the entry that was in the headline `tables/separation_scaling.md`) — but that's a comparison against an under-tuned MatTXC.
- The original `Temporal BatchTopK SAE` benchmark entry (50/50 split, 2k steps) was severely under-tuning T-SAE — our paper-faithful 0.611 vs 0.079 shows the existing benchmark entry undersold T-SAE by 0.53 in single-feature R².

So at this single seed, T-SAE Paper reaches 71% of ceiling and MatTXC reaches 98%. **But this is a single-seed measurement.** See the multi-seed update below — the picture changes substantially.

Files: `results_protocol_compare/cell_delta_*_results.json` + `results_protocol_compare/combined.json`. Configs: `config_protocol_compare.yaml`, `config_protocol_compare_d02.yaml` (latter re-runs only δ=0.2 since the δ=0.2 cell of the first run died when the local Mac OOM'd while the SSH wrapper was holding the remote process).

## Multi-seed transformer-variance study (TXC + T-SAE Paper, seeds {42, 1, 2})

The single-seed verdicts above hide a much larger source of variance: **the transformer init itself**. A small study at δ ∈ {0.15, 0.20} with 3 transformer seeds (each fully retrained from scratch) shows seed-to-seed variance dwarfs the architectural gap.

Best single-feature R² per (arch, δ, transformer seed):

| arch | δ | seed=42 | seed=1 | seed=2 | **mean ± std** | % of comp-0 ceiling |
|---|---|---|---|---|---|---|
| TXC          | 0.15 | 0.549 | 0.111 | 0.435 | **0.365 ± 0.227** | 50% |
| TXC          | 0.20 | 0.453 | 0.270 | 0.135 | **0.286 ± 0.160** | 33% |
| T-SAE Paper  | 0.15 | 0.285 | 0.049 | 0.280 | **0.205 ± 0.135** | 28% |
| T-SAE Paper  | 0.20 | 0.611 | 0.159 | 0.083 | **0.284 ± 0.285** | 33% |

Reference (seed=42 only): MatTXC bk10 = 0.564 (δ=0.15), 0.851 (δ=0.20).

What this changes:

1. **At δ=0.20, TXC and T-SAE Paper are statistically tied** — 0.286 vs 0.284 across 3 seeds. The "T-SAE Paper wins at 0.611" line was a seed=42 fluke. The "MatTXC wins at 0.851" line is also a single-seed measurement — the gap to TXC/T-SAE-mean is large enough to plausibly be real, but we can't tell from one seed.
2. **Standard deviations are 0.13–0.29 — i.e. ≈ 50–100% of the mean.** TXC at δ=0.20 ranges 0.135 → 0.453 across 3 transformer seeds. T-SAE Paper at δ=0.20 ranges 0.083 → 0.611. The single-seed numbers in the headline benchmark table are within-noise of any other arch.
3. **The earlier "Pareto-dominance" claims (T-SAE Paper > MatTXC, MatTXC > T-SAE Paper) are both unreliable.** With this much seed variance, head-to-head verdicts on this benchmark require ≥ 3 seeds before they mean anything.

Honest take:

- We do not have enough seeds to call a winner between TXC, MatTXC, and T-SAE Paper on the symmetric MESS3 setup.
- **MatTXC seed=42 = 0.851 at δ=0.20 is the highest single observation** but until we run MatTXC at seeds 1 and 2, we can't tell if that's its mean or its high-variance tail.
- The δ=0.20 cell appears more discriminating than δ=0.15 — but only marginally, given std ≥ 0.16 for everything.
- **Variance source decomposition would be valuable**: separately bound (a) transformer-init variance, (b) SAE-init variance, (c) probe-init variance. Currently all three move together with the main `seed:` field.

Recommended next step: run MatTXC at seeds {1, 2} on the existing seedvar transformers (config_seedvar_s1.yaml + s2.yaml already cached transformer.pt for seeds 1, 2), so all three archs have 3-seed coverage at δ=0.20.

Files: `results_seedvar_s1/cell_delta_{0.15,0.2}/results.json`, `results_seedvar_s2/cell_delta_{0.15,0.2}/results.json`, `results_txc_seed42/cell_delta_*/results.json`. Configs: `config_seedvar_s1.yaml`, `config_seedvar_s2.yaml`, `config_txc_seed42.yaml`.

## Files

- `config_tsae_paper.yaml` — minimal config with only T-SAE Paper arch
- `results_tsae_paper/cell_delta_*/results.json` — per-cell raw output
- `results_tsae_paper/combined.json` — combined per-arch×δ summary
