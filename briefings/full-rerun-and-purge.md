---
status: active
created: 2026-07-10
for: runpod
venue: runpod
supersedes: uniform-regrid-program-matrix.md
---

# Full clean-room rerun of the synthetic result set + purge of stale traces

**User order (2026-07-10):** rebuild the *entire* synthetic-benchmark result set
from scratch under the new uniform design, then delete stale/old result traces so
the canonical store holds only the fresh set.

## ⚠️ Time + safety (read first)

- **Time.** A full from-scratch frontier is ~2,000 cells ≈ **~20 h** at the
  reference rate (198 cells / 108 min, 6 workers). This **likely exceeds a 12 h
  window.** Mitigations, in order: bump `max_workers` to the box's core count;
  **shard by benchmark** (finish one bench's full grid before the next, so every
  completed bench is coherent); if the window closes mid-run, **stop after the
  last fully-completed bench** and do NOT purge.
- **Purge is the LAST step, never the first.** Run → regenerate → verify → *then*
  delete. A partial run must leave the old rows intact (safe fallback).
- **Purge is SCOPED to the synthetic datasources only.** NEVER touch rows for
  other experiments (probing / em / rlhf / § 4 coupling·denoising·markov·coupled).
  Synthetic datasources = `{toy_backtracking_selfexcite_d64,
  toy_signed_motion_M19_d40, toy_changepoint_modes_d64, toy_cyclic_circle_M101_d128,
  toy_cyclic_random_M101_d128}`. git history is the backup; commit before purging.

## Acceptance gate

1. `results/leaderboard.jsonl` for the five synthetic datasources contains **only**
   fresh rows: protocol **1.3.0**, fair-backbone archs, the new design, current
   code_version. Every non-synthetic row is byte-unchanged.
2. The program report (`render_report`) renders the **per-token matrix AND both
   companion panels** (`panel_nmse`, `panel_eauc`) fully filled at `{F, F/2}`.
3. Every per-bench `bench_record.md` + figs regenerates from the fresh rows.
4. `git status` clean after commit; pushed to `origin/arxiv`.

## Locked design (from the mac-local design pass — do not re-litigate)

- **Match per-token sparsity** on *realized* `l0_per_token` (evaluator already
  records it; `l0_per_window` is a diagnostic only). No per-window matrix.
- **Canonical matrix cell:** `T_can=4`, `B*=2`, capacities `{F, F//2}` (uniform).
- **Windows** `T ∈ {2,4,8}` (token archs give T=1). **L=32, seq_len=64** (uniform,
  unchanged).
- **Capacities/d_sae sweep** `{F//2, F, 2F}`: backtracking/changepoint `{10,20,40}`,
  signed_motion `{9,19,38}`, frequency `{50,101,202}`. `F` per `registry.py`
  (frequency F=101 = alphabet M; circle rank-2).
- **Archs (fair-backbone only):** `batchtopk_sae`, `tsae` (token);
  `stacked_batchtopk`, `txc_batchtopk_pre`, `txc_batchtopk_post`, `spectral_txc`
  (window). Deprecated (`topk_sae`, `stacked_sae`, `txc_base`) are OUT — their old
  synthetic rows are exactly what the purge removes.

## Step 0 — protocol bump

In `src/temp_bench/evals/synthetic_recovery.py` bump `protocol_version` →
**`"1.3.0"`** (realized L0 added; recovery metrics byte-identical). Switch the
protocol filter to `"1.3.0"` in **all** renderers — the four per-bench
`render_figs.py` **and** `registry.py` (`Bench.protocol`). After the full rerun,
every synthetic cell has a 1.3.0 row, so 1.3.0-only is complete.

## Step 1 — clean-room rebuild (from scratch)

Clear the synthetic training cache so the rerun is genuinely from scratch, not a
cache replay (identify synthetic checkpoints via `checkpoints/manifest.jsonl`;
if they can't be cleanly separated from other experiments' checkpoints, skip the
clear — the new design's d_sae/k_pos differ so most cells miss cache anyway).

Then, **sharded by bench**, run the full uniform grid via `grid.run_pool` (one
cell list per bench, canonical runner, incremental writes):
- **archs** = the 6 fair-backbone archs (spectral on frequency for sure; on the
  other three if the window allows — else leave `—`);
- **T** ∈ `{1 (token), 2, 4, 8}`; **d_sae** ∈ `{F//2, F, 2F}`;
- **k_pos** ∈ `{1, 2, 4, 8, 16}` meeting each arch's dict constraint (pre/stacked
  `d_sae ≥ k_pos·T`; post `d_sae ≥ k_pos`) — `log()` clipped drops;
- **seeds** `{1,2,42}` + the untrained control (`n_steps=0`) per (arch,T).

The wide k_pos sweep lets the matcher hit `B*=2` at `T_can=4` for each arch
(expected: token `k_pos=2`, pre/stacked `k_pos=2`, post `k_pos=8`) AND gives the
per-bench frontiers their k_pos axis.

## Step 2 — regenerate everything

From the fresh 1.3.0 rows, regenerate: every per-bench record
(`…<bench>.render_figs`) + figs, and the program report (`render_report` → matrix
+ `panel_nmse` + `panel_eauc` + `program_stats.json`).

**Sanity, not a gate:** the from-scratch recovery numbers should *reproduce* the
old committed per-bench records closely (same seeds/params). Small drift from
BatchTopK/GPU nondeterminism is fine; **large drift is a finding — investigate and
report, do not silently overwrite.** The old design's d_sae axis ({8,16,20,40})
changes to {F//2,F,2F}, so the frontier *plots* legitimately change.

## Step 3 — purge stale traces (LAST, scoped)

Only after Steps 1–2 are complete and verified for **all four benches**:

1. Rewrite `results/leaderboard.jsonl` keeping a row iff:
   `datasource ∉ SYNTH` **OR** (`datasource ∈ SYNTH` **AND** `protocol == "1.3.0"`
   **AND** `code_version.commit_sha == <this rerun's commit>`). This drops every
   stale synthetic row (old protocol, deprecated archs, old code_version) and
   preserves all non-synthetic rows exactly.
2. Prune orphaned synthetic checkpoints via the manifest **only if** cleanly
   identifiable; otherwise leave (content-addressed cache is harmless).
3. Delete regenerated-and-superseded synthetic artifacts that are no longer
   referenced (old stray results JSONs / figs the renderers replaced).
4. Re-run every renderer once more on the purged leaderboard → confirm the matrix,
   panels, and all per-bench records still fully populate (proves nothing depended
   on a purged row).

## Step 4 — commit + wrap

Commit (protocol bump + renderer switches + fresh rows + regenerated records/figs +
program_stats + the purge), push to `origin/arxiv`, update
`experiments/explorations/synthetic/STATUS.md` §0, and **delete this briefing**.

## Constraints (hard rules)

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**;
everything through the canonical runner (code-version stamped); paper-section
names. Prime directive: a sound verdict, never a win — genuine holes stay `—`.
