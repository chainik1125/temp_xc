---
status: active
created: 2026-07-10
for: runpod
venue: runpod
---

# Uniform re-grid → fill the program-level per-token B×A matrix

## Goal (the acceptance gate, read first)

After this job, `-m experiments.explorations.synthetic.render_report` must render
the **per-token matched** matrix in `experiments/explorations/synthetic/REPORT.md`
with **every in-scope cell filled at both capacities** `{F, F/2}` (no `—`, no
loose `*`), and `results/program_stats.json` must show each cell's realized
`l0_per_token` within tolerance of `B* = 2`. A cell is `(bench, latent-axis) ×
arch` per `registry.py`, shown as `recovery@F / recovery@(F/2)`.

Plus: every per-bench `bench_record.md` regenerates with **zero numeric drift**
on its existing metrics (only the new realized-L0 fields appear) — the same
zero-drift gate as the record-pipeline refactor.

## Locked design (from the mac-local design pass — do not re-litigate)

- **Fairness:** match **per-token sparsity** (`l0_per_token = B*`); this is the
  controlled comparison. Per-window is NOT a program matrix (its `B*≥T` need
  collides with deep-scarce `d_sae`); it is a per-bench T-frontier overlay
  (§ Step 3, secondary).
- **Realized L0, not the knob:** match on measured `l0_per_token` /
  `l0_per_window` (added to the shared evaluator). Nominal `k_pos` diverges from
  realized density across archs.
- **Canonical cell:** `T_can = 4` (token archs T=1), `B* = 2` atoms/token, at each
  capacity `{F, F//2}`. Feasible everywhere: `k_win = B*·T_can = 8 ≤ min(F//2) = 9`.
- **Windows swept:** `T ∈ {2, 4, 8}` (powers of two, `T ≤ L/2 = 16`); token archs
  give T=1. **No T=16** (drops out of the scarce regime under per-token matching).
- **Capacities:** `{F//2, F, 2F}` per bench (matrix uses `{F, F//2}`; `2F` is the
  over-complete frontier point). `F` per `registry.py`: backtracking 20,
  signed_motion 19, changepoint 20, frequency 101 (**alphabet M**, circle is
  rank-2 — alphabet-scaled, not a direction count).

## Why this job exists

The program substrate is built + committed (`registry.py`,
`src/explorations/synthetic/report.py`, `render_report.py`, `REPORT.md`,
`test_program_report.py`). The matrix renders **all `—`**: archs match on realized
L0 and **no historical row carries it**. Two facts from the coverage block:
- **signed_motion has zero fair-backbone rows** (TopK-legacy only:
  `topk_sae`/`stacked_sae`/`txc_base`) — needs a fresh fair-backbone grid.
- backtracking / changepoint / frequency have fair-backbone coverage but no L0.

## Step 0 — protocol bump (guarantees L0 on every cell)

Realized L0 was added additively at protocol **1.2.0**, so a re-run of an
*existing* cell hits the eval cache and skips it. To force L0 everywhere:

1. In `src/temp_bench/evals/synthetic_recovery.py` bump
   `SyntheticRecovery.protocol_version` → **`"1.3.0"`** (comment: realized L0
   added; recovery metrics byte-identical; bump invalidates the eval cache so
   every cell re-evaluates — checkpoints reused, `train_key` unchanged).
2. Switch the protocol filter to `"1.3.0"` in the four per-bench renderers and in
   `registry.py` (`Bench.protocol`). Post-re-grid every cell has a 1.3.0 row, so
   1.3.0-only avoids double-counting old 1.2.0 rows.
3. Update any test that pins `"1.2.0"`.

## Step 1 — the grid

Scope: **fair-backbone family** (`batchtopk_sae`, `tsae` [token];
`stacked_batchtopk`, `txc_batchtopk_pre`, `txc_batchtopk_post` [window];
`spectral_txc` [see note]). Benches: all four. Reuse `grid.run_pool` (one cell
list, canonical runner, never a bespoke append).

Per bench sweep:
- **d_sae** ∈ `{F//2, F, 2F}` — backtracking/changepoint `{10,20,40}`,
  signed_motion `{9,19,38}`, frequency `{50,101,202}`.
- **T** ∈ `{1 (token archs), 2, 4, 8}`.
- **k_pos** ∈ `{1, 2, 4, 8, 16}`, only values meeting the arch's dict constraint
  (pre/stacked: `d_sae ≥ k_pos·T`; post: `d_sae ≥ k_pos`). `log()` clipped drops.
- **seeds** `{1, 2, 42}` + the **untrained control** (`n_steps=0`) per arch/T.

The wide k_pos sweep lets the renderer find, per arch, the cell whose realized
`l0_per_token` hits `B*=2` at each capacity.

**spectral_txc note:** frequency-specialized (DCT-band). Run on **frequency** for
sure; on the other three it's a genuine question (does band-limiting help the
equality / self-exciting latents?) — include if budget allows, else leave `—`.
Keep `spectral_txc_dcac`/`_full` to the frequency band addendum only.

## Step 2 — converge the canonical cells

After the grid, `render_report` + read `program_stats.json`. Each in-scope cell
must resolve at `(T=4 [or 1 for token], d_sae ∈ {F, F//2})` with realized
`l0_per_token ≈ 2`. Expected anchors: token `k_pos=2`; pre/stacked `k_pos=2`
(`l0_t≈2`, `k_win=8`); post `k_pos=8` (`l0_t=k_pos/T≈2`). If a cell is missing or
`loose`, add the k_pos that realizes `B*=2` for that arch and re-run just those
(cache makes it cheap).

## Step 3 — regenerate + verify

1. Regenerate every per-bench record (`…<bench>.render_figs`) + the program report
   (`render_report`). Confirm **zero numeric drift** on existing metrics vs the
   committed records (diff AUTO blocks + stats JSON; only L0 fields are new).
2. Confirm the per-token matrix is fully filled at both capacities.
3. **(Secondary)** add the per-token-vs-per-window **overlay across `T∈{2,4,8}`**
   to each bench's frontier figure (per-token line = `l0_per_token=B*`; per-window
   line = window archs held to `l0_per_window=B*`, i.e. their minimum density) —
   the skeptic check, where its T-scaling is visible. Not gated; do if time allows.
4. Commit (protocol bump + renderer switches + new rows + regenerated records +
   figs + `program_stats.json`), push to `origin/arxiv`, update
   `experiments/explorations/synthetic/STATUS.md` §0, and **delete this briefing**
   (done → gone, per `briefings/README.md`).

## Constraints (hard rules)

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**
(the protocol bump is in the eval plugin, not core); everything through the
canonical runner; paper-section names. Prime directive: a sound verdict, never a
win — if a cell is a genuine hole (arch can't reach `B*` at that `d_sae`), leave
it `—` and note why; do not shop knobs to fill it.

## Size / sharding

~4 benches × ~5 archs × 4 T × ~4 feasible k_pos × 3 d_sae × 3 seeds, but train is
cache-reused and eval is cheap. Shard by bench if needed (`run_pool` writes
incrementally). `log()` clipped/dropped corners so truncation never reads as full.
