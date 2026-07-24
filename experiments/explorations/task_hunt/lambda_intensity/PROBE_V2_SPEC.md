# λ-readout v2 — freeze-candidate spec (probe-capacity knobs)

**Status: FREEZE CANDIDATE (runpod-b, `briefings/probe-adequacy.md`).
Nothing here is adopted. The λ-readout methods decision is mac-local's,
taken jointly with the variance-machinery owner at the methods review;
adoption = freezing this file as-is (or amending it there). Until then
the canonical readout remains v1 (`temp_bench/evals/lambda_recovery.py`,
frozen, bit-identical forever) and every committed panel number stands
unchanged.**

**Why this exists.** runpod-d (λ̂ panel) and runpod-e (hedging panel)
report — independently measured, **under review, not yet reviewed or
adopted** (their 2026-07-24 LOG entries; RECORD_B § 1d) — that the v1
probe is capacity-limited for dense codes: unregularized OLS on
p = d_sae = 2048 features with n = 1024·(32/T) rows sits at n ≈ p at
T = 16 (negative held-out r² on dense cells), and ridge and/or more
windows lift dense-code cells by +0.18…+0.23 while their diagnostics
reproduce the committed panel numbers to 1e-4 at the old settings. This
spec is the contingency: it makes the decision *executable* either way.

## 1. The frozen v2 convention

Implementation: `src/temp_bench/evals/lambda_recovery_v2.py`, dispatched
by `SyntheticRecovery` **only** when `eval_cfg` sets
`lambda_probe_v2: true` (flag absent → byte-identical rows, evaluator
protocol stays 1.3.0). v2 **imports** v1's window sampler and tile
readout, so everything below the probe is identical by construction:

**Unchanged from v1** (readout convention):
- per-tile code readout; λ target at the tile's leading edge (position
  T−1); per-token archs read single positions; feature dim = ONE tile's
  code, never the concatenation over tiles;
- window sampling: `_sample_windows`, train pool seed 0 / eval pool
  seed 1, x and λ windows seed-aligned; NaN leading-edge targets
  dropped with v1's guard semantics;
- chance floor: the same probe procedure on shuffled train targets
  (permutation seed 7); degenerate-input zero returns;
- training, checkpoints, train_keys: untouched — a v2 re-run is
  **eval-only**.

**The three capacity knobs** (each explicit in `eval_cfg`; code defaults
match, but panel re-runs set all of them so eval_key pins the values):

1. **Probe** — `lambda_v2_probe: ridge`. Ridge regression with the
   penalty selected **inside the train half only**:
   `sklearn.linear_model.RidgeCV(alphas=np.logspace(-2, 4, 13))` — the
   efficient leave-one-out selection on the train tile-rows;
   deterministic; the eval half is never touched until the single final
   prediction. The selected α ships on the row as `lambda_alpha_v2`.
   The grid is runpod-d's committed diagnostic grid; runpod-e's
   `logspace(-1, 4, 12)` spans one decade less at the low end. All 72
   α selections recorded in runpod-d's diagnostic results are interior
   (1.0…3162, never an endpoint); runpod-e's diagnostic does not
   record the selected α.
   `lambda_v2_probe: ols` is the exact α → 0 limit (the same
   `LinearRegression` fit as v1) — contract-tested to reproduce v1.
2. **Windows** — `lambda_v2_n_windows: 8192` per half. Adequacy
   arithmetic at the panel anchor (p = d_sae = 2048, L = 32): at the
   largest panel tile T = 16, n_rows = 8192·(32/16) = 16384 = **8·p**
   — the briefing's n_rows ≥ 8·p line, and the setting both
   diagnostics used. Disclosed exception: **Stacked** reads at
   T·d_sae features (32768 at T = 16), so it stays p > n even at
   nw = 8192 — there the ridge penalty, not the row count, is what
   makes the cell readable (consistent with it taking the largest
   reported lift). Any claim comparing Stacked cells should say so.
3. **Split** — `lambda_v2_split: trace`: advance the v1 split index
   n//2 forward to the next TRACE boundary (sequence → trace map =
   `data.extra["trace_ids"]`, now exposed by both Ward datasources; a
   datasource that declares none means every sequence is its own trace,
   so synthetic benches keep exactly v1's n//2). Forensics receipt
   (`results/split_forensics.json`): the stream is trace-contiguous;
   exactly one trace (152) straddles n//2 (14 train / 1 eval window);
   at the committed panel settings ZERO eval draws touch it, so **no
   committed number is affected by split leakage**; at nw = 8192 the
   raw half-split would leak 2/8192 eval draws (|Δr| ≤ ~5e-4) and the
   boundary snap leaks none. `lambda_v2_split: half` remains available
   for exact-v1 comparisons.

**Row contract.** v2 rows are NEW leaderboard rows (the flags hash into
eval_key) carrying both generations: the unchanged v1 columns
(`lambda_recovery`/`lambda_r2`/`lambda_chance`, still computed at v1's
own settings) and `lambda_recovery_v2` / `lambda_r2_v2` /
`lambda_chance_v2` / `lambda_alpha_v2` / `lambda_v2_n_{train,eval}_rows`
— every v2 row is its own paired v1 readout on the same data. No
existing row is rewritten or invalidated.

Contract tests: `tests/test_lambda_recovery_v2.py` — (a) ols + nw 1024 +
half split reproduces v1 to 1e-10 (all-finite and NaN grids); (b)
determinism; (c) the trace split never places two windows of one trace
in different halves, snaps forward only, rejects non-contiguous ids;
(d) `python run.py validate` green + the smoke sweep YAML
(`configs/sweeps/lambda_probe_v2_smoke.yaml`) resolves and pins the
frozen knob values.

## 2. What adopting implies — panel re-runs

Eval-only re-runs (all checkpoints reused; train_keys unchanged):

| panel | datasource | leaderboard rows to re-eval |
|---|---|---|
| λ̂ Stage 2 (+ post-matched amendment) | `ward_real_lambda_base_l12` | 108 (84 at k_pos = 8, 24 at k_pos = 8·T) |
| hedging-trend Stage 2 | `ward_real_slope8_distill_l14` | 84 (60 + 24) |
| **total** | | **192 cells** |

Mechanics: the cells gain `eval_extra` (the `grid.run_cell`
pass-through added for this purpose) — e.g. in the panel runners after
`cells = _cells(ds)`:

```python
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}
for c in cells:
    c["eval_extra"] = V2
```

Training is cache-hit per cell; only the eval re-runs.

**Cost estimate** (stated arithmetic, not measured): per cell the v2
eval encodes 2 × 8192 windows × 32 positions ≈ 0.5 M token-positions
(seconds on GPU; the 4044×128×4096 fp16 cache materialization
dominates I/O) and fits RidgeCV twice (headline + chance floor) on up
to 131k × 2048 rows (T = 2; ~0.5–1 CPU-min via one SVD amortized over
the 13 α values) or 16k × 32768 dual-form for Stacked T16. Call it
**~2–4 min/cell wall, ≈ 190 cells ⇒ roughly 8–12 machine-hours
sequential, ~3–4 h at the panel's usual 3 workers; GPU-minutes proper
(encode only) < 3 GPU-hours total.** The smoke sweep
(`lambda_probe_v2_smoke.yaml`, synthetic selfexcite, 1 cell) is the
cheap end-to-end check to run first — note any sweep writes
leaderboard rows, which is why this task committed the YAML without
running it.

## 3. What re-bases in the variance receipts

`support_stats/stage2_variance.py` is now probe-agnostic (CLI
parameters; defaults reproduce the committed v1 receipts
byte-identically — verified by empty diff). After the λ̂ panel re-run
lands its results JSON, the v2 re-base is one command:

```
.venv/bin/python -m experiments.explorations.task_hunt.support_stats.stage2_variance \
    --probe v2 --crosscheck-json <path to the v2 re-run results JSON> \
    --out-prefix stage2_variance_v2
```

That re-derives, on v2 numbers: every per-seed cell value and 95% CI;
the paired-by-seed TXC-pre − T-SAE and TXC-pre − per-token differences
(sign-flip p, BCa/t CIs); the T = 2→8 trend permutation p (the panel's
one significant headline at n = 3, permutation p = 0.0093 on v1
numbers — it must be restated from the v2 receipts, not carried over);
the trained−untrained margins; and the seed power calc. The committed
v1 receipts (`stage2_variance.{json,md}`) are never overwritten — the
v2 receipts land beside them. The hedging panel has no committed
variance receipts from this machinery; if the review wants them
re-based there too, the same command with `--ds
ward_real_slope8_distill_l14` and that panel's results JSON applies
(its post cells run at k_pos = 8·T; pass `--k-pos` accordingly per
sub-population).

## 4. What this spec does NOT decide

Whether the canonical λ readout changes. The probe-capacity findings
remain **reported, under review** (runpod-d and runpod-e's 2026-07-24
LOG entries); the committed panels' verdicts and their leaderboard
numbers stand as written until the methods review says otherwise. If
the review adopts v2, the § 3b "peaks rather than saturates" reading
and the hedging panel's T-decline reading are re-examined on the v2
columns — with the v1 columns still present on every row for the
paired comparison. If the review declines, v1 remains canonical, the
flag stays available for diagnostics, and nothing on the leaderboard
has moved either way.
