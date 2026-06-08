# Backtracking — self-exciting (Hawkes) AC benchmark

**Autoresearch investigation #1. Verdict: POSITIVE** (the loop's first positive
architecture result). Real CoT backtracking is **self-exciting**; its validated
synthetic mirror is an order-sensitive (AC) benchmark on which **window/temporal
dictionaries recover the hidden intensity `λ` (≈0.95 at T≥4) while per-token SAEs
sit at their provable DPI floor (≈0.41)** — robustly into the scarce regime
(`d_sae < F = 20`). The window archs trade away local-feature recovery (eAUC) to
do it: a clean global-vs-local specialization.

**Reading order:** [`prereg.md`](prereg.md) (measurement preregistration, frozen)
→ [`measurement.md`](measurement.md) (measure → mirror record) →
[`bench_spec.md`](bench_spec.md) (architecture-test spec, frozen) →
[`bench_record.md`](bench_record.md) (**the results**).

**Scripts** (`-m autoresearch.backtracking.<x>` from `purified/`): `measure`
(stages 2-3), `mirror` (fit+validate), `gating` (§8 ceilings), `kernel_order`
(held-out K selection → K=2), `run_grid` (120-cell grid), `render_figs`.
**`figs/`** — frontier, λ-vs-T, untrained control, eAUC/NMSE, signature, mirror.
**`results/`** — derived stats JSON (grid_results, bench_stats, gating,
kernel_order, mirror, kpos_robustness, measurement). Real input labels:
`../../results/c7_backtracking/stage_a/` (Ward Stage-A).
