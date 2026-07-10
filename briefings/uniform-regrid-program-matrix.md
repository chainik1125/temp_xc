---
status: active
created: 2026-07-10
for: runpod
venue: runpod
---

# Uniform re-grid → fill the program-level B×A matrix

## Goal (the acceptance gate, read first)

After this job, `-m experiments.explorations.synthetic.render_report` must render
**both** matrices in `experiments/explorations/synthetic/REPORT.md` with **every
in-scope cell filled** (no `—`, no loose-match `*`) at the canonical operating
point, and `results/program_stats.json` must show each cell's realized L0 within
tolerance of `B*`. Concretely, for **matrix A (per-position)** and **matrix B
(per-window)**, each `(bench, latent-axis) × arch` cell present in
`registry.py` resolves to a real grid group.

Plus: every per-bench `bench_record.md` regenerates with **zero numeric drift**
on its existing metrics (only the new realized-L0 fields appear) — the same
zero-drift gate as the record-pipeline refactor.

## Why this job exists

The program report substrate is built and committed (`registry.py`,
`src/explorations/synthetic/report.py`, `render_report.py`, `REPORT.md`,
`test_program_report.py`). It renders today but the matrix is **all `—`**: the
architectures are matched on **realized L0** (`l0_per_token` / `l0_per_window`,
added to the shared evaluator), and **no historical row carries it**. This job
produces a uniform grid that does.

Two facts from the current coverage block (see `REPORT.md` → Grid coverage):
- **signed_motion has zero fair-backbone rows** — it was only ever run on the
  TopK-legacy backbone (`topk_sae`, `stacked_sae`, `txc_base`). It needs a fresh
  fair-backbone grid to enter the matrix at all.
- backtracking / changepoint / frequency have good fair-backbone coverage but no
  realized L0.

## Step 0 — protocol bump (the mechanism that guarantees L0 everywhere)

Realized L0 was added additively at protocol **1.2.0**, so *new* cells get it but
a re-run of an *existing* cell hits the eval cache and does **not**. To capture
L0 on every cell deterministically:

1. In `src/temp_bench/evals/synthetic_recovery.py`, bump
   `SyntheticRecovery.protocol_version` to **`"1.3.0"`** with a one-line comment:
   realized L0 (`l0_per_token` / `l0_per_window`) added to the metric set;
   recovery metrics byte-identical; bump invalidates the eval cache so every cell
   re-evaluates and records L0 (checkpoints reused — `train_key` unchanged).
2. Switch the protocol filter to `"1.3.0"` in the four per-bench renderers
   (`backtracking/`, `changepoint/`, `frequency/`, `signed_motion/render_figs.py`)
   and in `registry.py` (`Bench.protocol` → `"1.3.0"` for all four). After the
   comprehensive re-grid every cell has a 1.3.0 row, so 1.3.0-only loses nothing
   and avoids double-counting old 1.2.0 rows.
3. Update the protocol assertion in `tests/` if one pins `"1.2.0"`.

(Train checkpoints are cache-reused; this is an eval-only re-run for existing
cells + full runs for new ones. Much cheaper than a from-scratch grid.)

## Step 1 — the grid

Scope: the **fair-backbone family** only (`batchtopk_sae`, `tsae` [token];
`stacked_batchtopk`, `txc_batchtopk_pre`, `txc_batchtopk_post` [window];
`spectral_txc` [see note]). Benches: backtracking, changepoint, frequency,
signed_motion. Reuse the shared grid driver (`src/explorations/synthetic/grid.py`
`run_pool`) — one cell list, one canonical runner, never a bespoke append.

Per bench, sweep:
- **d_sae** anchored on `F` — `{⌈F/2⌉, F, 2F}` (mark `F`); frequency keeps its
  `{32, 64, 101, 256}`.
- **T** ∈ `{1 (token archs), 2, 4, 8}`; frequency also `16`.
- **k_pos** ∈ `{1, 2, 4, 8, 16}` — but only values satisfying each arch's
  dictionary constraint: pre/stacked need `d_sae ≥ k_pos·T`; post's budget is
  `k_pos` per window so it needs `d_sae ≥ k_pos`. Skip (and `log`) clipped
  corners.
- **seeds** `{1, 2, 42}`; plus the **untrained control** (`n_steps=0`) per arch/T
  as the existing benches do.

The wide k_pos sweep is what lets the renderer find, for each arch, the cell
whose realized L0 hits `B*` under *each* convention (they need different k_pos —
see `test_program_report.py`).

**spectral_txc note:** it is the frequency-specialized column (DCT-band prior).
Run it on **frequency** for sure; running it on the other three benches is a
genuine question (does band-limiting help the equality / self-exciting latents?)
— include it if budget allows, else frequency-only and leave those cells `—`.
Keep `spectral_txc_dcac` / `_full` to the frequency band-partition addendum only.

## Step 2 — converge the canonical cells

The canonical operating point is `d_sae = F`, `T_can = 4`, `B* = 4`
(`registry.OP`). After the grid, run `render_report` and read
`program_stats.json`:
- **per-position** cell for an arch = its group at `(F, T=4 [or 1 for token])`
  with realized `l0_per_token ≈ 4`;
- **per-window** cell = realized `l0_per_window ≈ 4`.

If any in-scope canonical cell is missing or its `loose` flag is true, add the
one k_pos that realizes `B*` for that arch at `T=4` and re-run just those cells
(cache makes this cheap). Expected anchors: token `k_pos=4`; pre/stacked
per-position `k_pos=4` & per-window `k_pos=1`; post per-position `k_pos=16` &
per-window `k_pos=4`.

## Step 3 — regenerate + verify

1. Regenerate every per-bench record (`…<bench>.render_figs`) and the program
   report (`render_report`). Confirm **zero numeric drift** on existing metrics
   vs the committed records (diff the AUTO blocks + stats JSON; only L0 fields
   are new).
2. Confirm both matrices are fully filled at the canonical point.
3. Commit: the protocol bump + renderer switches, the new leaderboard rows, the
   regenerated records + figures + `program_stats.json`. Push to `origin/arxiv`.
4. Update `experiments/explorations/synthetic/STATUS.md` §0 (program report DONE,
   matrix filled) and **delete this briefing** (done → gone, per `briefings/README.md`).

## Constraints (hard rules)

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**
(the protocol bump is in the eval plugin, not core); everything through the
canonical runner (code-version stamped); paper-section names. Prime directive: a
sound verdict, never a win — if a matrix cell is a genuine hole (arch can't reach
`B*` at `d_sae=F`), leave it `—` and note why; do not shop knobs to fill it.

## Size / sharding

Rough order: 4 benches × ~5 archs × ~4 T × ~4 feasible k_pos × 3 d_sae × 3 seeds
≈ a few thousand cells, but train is cache-reused and eval is cheap. Shard by
bench if needed (`run_pool` writes incrementally). `log()` any clipped/dropped
corners so silent truncation never reads as full coverage.
