# STATUS — synthetic-benchmark redo (living briefing / pre-compact handoff)

**This is the one-stop briefing.** Update *this file only* before a compact; it
is the canonical current-state of the synthetic-benchmark fairness redo. Read it
top-to-bottom, then the linked per-benchmark docs as needed.

Last updated: 2026-06-09.

---

## 0. TL;DR — what's active right now

**The backtracking BatchTopK fair-backbone redo is DONE** (2026-06-09). All 4 new
BatchTopK archs are built + registered, the 198-cell grid ran (0 gaps), and the
single-source record + paper figures are regenerated and committed. **Verdict
holds:** per-token pinned at the DPI floor 0.41; all three window families
(TXC-pre, TXC-post, Stacked) recover λ 0.87→0.95; the win survives the uniform
backbone. New findings: **TXC-pre > TXC-post** (post slips at large T — sparser
per-window code), and the **shared-code crosscoder ≫ Stacked on eAUC** at matched
λ. See [`backtracking/bench_record.md`](backtracking/bench_record.md).

Why it was done: T-SAE already used **BatchTopK** (Bussmann et al. — the strong
backbone) while TopK-SAE / TXC / Stacked-SAE used plain per-sample **TopK**, an
uncontrolled confound favouring T-SAE. Fix: BatchTopK everywhere, so the *only*
variable is decode/code structure. (Design that was locked + executed: § 4–5.)

**Next** (no longer this redo): the change-point / EM roadmap (§ 6, gated on a
labeler) or any new bench, reusing the same fair-backbone grid as the template.

---

## 1. Where everything lives (post-reorg)

The program is a self-contained tree: `purified/synthetic/`.

- Single governing doc at the root: [`README.md`](README.md) — prime directive
  ("a sound verdict, never a win"), the measure→mirror→bench loop + § 3 validity
  gates, the capacity/windowing/probe conventions, and the benchmark index. The
  DC/AC frequency lens lives at [`../docs/ideas/frequency_lens.md`](../docs/ideas/frequency_lens.md).
- One self-contained subdir per benchmark, each with docs + scripts + `figs/` +
  `results/`. Scripts run as `.venv/bin/python -m synthetic.<bench>.<script>`.
- **Canonical results store (single source of truth):**
  `purified/results/leaderboard.jsonl` — every cell, code-version-stamped, via
  the runner. Real-label inputs (Ward backtracking) stay at
  `purified/results/c7_backtracking/stage_a/`.
- **Archs** are framework plugins in `src/temp_bench/archs/`; data generator in
  `src/temp_bench/data/synthetic.py`; evaluators in `src/temp_bench/evals/`.
  Never edit `src/temp_bench/core/` (hard rule).

### Single-source record pipeline (already built — keep using it)
`results/leaderboard.jsonl` → `-m synthetic.backtracking.render_figs` →
paper-quality `figs/backtracking_{main,specialization,untrained_control,local_tradeoff}.{pdf,png}`
+ `results/backtracking_bench_stats.json` + **auto-filled** `<!-- AUTO:* -->`
blocks in `bench_record.md` (headline + every table). `render_figs` reads the
leaderboard directly and is **idempotent** — no hand-typed numbers anywhere.
Figures are embedded (`![...]`) so they render in VS Code preview.

---

## 2. Benchmark status

| benchmark | dynamics class | verdict | backbone | state |
|---|---|---|---|---|
| **backtracking** | self-exciting (AC) | **POSITIVE** | **BatchTopK (redo DONE)** | measured+mirror+bench done; 198-cell BatchTopK re-grid landed + record/figs regenerated (§ 4) |
| **signed_motion** | order-sensitive (AC) | NEGATIVE | TopK | done; leave as published |
| **topic_switching** | change-point/sticky | **ABORT** (composition-dominated, labeler inadequate) | — | measured; no bench |
| **changepoint** | change-point/absorbing | — | — | spec only, **gated** (no real anchor: topic aborted, EM needs paid judge) |

Backtracking headline (current TopK result, to be reproduced under BatchTopK):
per-token SAEs pinned at the **DPI floor λ≈0.41**, window crosscoders **λ≈0.95**
(T≥4), robust into the scarce regime `d_sae<F=20`; clean local(eAUC)-vs-temporal(λ)
specialization. See [`backtracking/bench_record.md`](backtracking/bench_record.md).

---

## 3. Key facts the redo rests on

- **`lambda_recovery`** (headline metric): held-out Pearson corr of a *linear*
  regression probe on the arch's code → the hidden self-exciting intensity `λ`,
  per-tile at the tile's leading edge. Per-token DPI floor `√(Varλ/Varb)≈0.41`
  (provable); window ≈0.95. `eAUC` = local feature-direction recovery (no `gAUC`
  here — `λ` is a *latent*, not a direction; `λ` is the "global/temporal" axis).
- **Batch-size finding (the trigger for the throughput fix):** the trainer passes
  the same `batch_size=1024` to every arch but it counts *native units*
  (`trainer.py:124`). So per-token archs reconstruct **1024 tokens/step**; window
  archs (TXC/Stacked) reconstruct **1024·T**. At fixed `n_steps` the window archs
  see up to 8× more data — an uncontrolled, result-aligned confound. Fix in § 4.
- **TXC mechanics (load-bearing for the design):** `encode` *sums* the
  per-position pre-acts into one `(d_sae)` code (`einsum("btd,tds->bs")`,
  `txc_base.py:133`), TopK on that squashed code, then decodes **all T positions
  from the same shared code** via `W_dec[:,t,:]`. So a TXC's sparse code is
  *shared across positions* — that's the crosscoder hypothesis, not a bug.

---

## 4. THE BatchTopK fairness redo (design LOCKED — ✅ EXECUTED 2026-06-09)

> **Status: DONE.** All 4 archs below are built (`src/temp_bench/archs/{batchtopk_sae,
> stacked_batchtopk,txc_batchtopk}.py`), registered, smoke-tested, and the 198-cell
> grid ran with 0 gaps. Record + figures regenerated from the leaderboard and
> committed. The design notes below are kept as the rationale of record.

Convert every arch to a **BatchTopK backbone** (BatchTopK during training →
fixed **JumpReLU threshold** at inference, matching T-SAE's recipe / Bussmann
et al.), so "backbone" is controlled and the only variable is decode structure.
Build these as **NEW arch variants** (leave the published TopK archs `topk_sae`,
`txc_base`, `stacked_sae` untouched so § 4 coupling/denoising + signed_motion
results stand). Parameterize by `k_pos`.

### The four archs (all BatchTopK-train → JumpReLU-eval, AuxK for dead features)

| arch | unit / selection | `k` rate | pool |
|---|---|---|---|
| `batchtopk_sae` (T=1) | per-token encode+decode | `k_pos`/token | B tokens |
| `tsae` (T=1) | per-token + temporal-contrastive (**already this**) | `k_pos`/token | B tokens |
| `stacked_batchtopk` (T∈{2,4,8}) | per-position, **independent** dicts/decode | `k_pos`/position | B·T (pos,batch) |
| `txc_batchtopk_pre` (T∈{2,4,8}) | **pre-squash**: BatchTopK on per-position pre-acts → sum survivors → shared code → shared decode | `k_pos`/token | B·T tokens |
| `txc_batchtopk_post` (T∈{2,4,8}) | **post-squash**: sum pre-acts → BatchTopK on the squashed code → shared decode | `k_pos`/**window** | B windows |

We run **both** TXC squash variants — whether a crosscoder should select
per-position (pre) or on the aggregate (post) is an open architectural question,
measure it.

### Budget accounting (the subtle part — get this right)

- **pre-squash / per-token / per-position**: `k_pos` actives **per token**
  (pool over the `B·T` tokens). Pre-squash code support = union of per-position
  selections (≤ `k_pos·T`).
- **post-squash**: `k_pos` actives **per window code** = **`k_win // T`** (the old
  TopK used `k_win = k_pos·T`). Rationale: each squashed atom is **reused at all
  T positions**, so `k_pos` shared atoms ≈ `k_pos·T` token-activations = parity
  with the per-token archs. ⇒ post-squash BatchTopK **corrects the `k_win=k_pos·T`
  over-count** (that convention double-counted the shared atoms). Keep `k_pos·T`
  a clean multiple of `T` (automatic since we set `k_pos` integer).

Consequence to expect (not a bug): post-squash code is far sparser (support
`k_pos`) than pre-squash (support up to `k_pos·T`) at the same budget — that
density gap *is* the pre-vs-post effect; NMSE/eAUC are reported outcomes, not to
be equalized.

### Throughput normalization (the batch-size fix)

**Equal tokens/step.** Window/sequence archs get `batch_size = base // T` (per-token
archs keep `base`), so every arch reconstructs ~`base` token-positions/step and
sees identical total data over `n_steps`. Bonus: it also makes the **BatchTopK
pool identical** (`B·T = base` tokens for every arch) so the global threshold is
computed over the same granularity. (Default `base = 1024`.)

### Decision log (why these choices)
- New variants, not in-place edits → protect published § 4 + signed_motion.
- Both pre & post squash → it's an unresolved architectural question.
- post-squash `k_win//T` → shared atoms are reused `T×`.
- equal tokens/step → removes the data-exposure confound + equalizes the
  BatchTopK pool.
- eval = JumpReLU threshold for all → consistent (per-token codes become
  variable-sparse at eval, as T-SAE already is; the λ/eAUC probes read `encode()`).

### Grid
11 (arch,T) configs [`batchtopk_sae`·T1, `tsae`·T1, `stacked_batchtopk`·T{2,4,8},
`txc_batchtopk_pre`·T{2,4,8}, `txc_batchtopk_post`·T{2,4,8}] × `d_sae`{8,16,20,40}
× seeds{1,2,42} = **132 trained** + **~33 untrained control** (`n_steps=0`,
`d_sae=20`) ≈ **165 cells**. `k_pos=1`, `L=32`, `n_steps=30000`, equal tokens/step.

---

## 5. Implementation checklist for the next agent

1. **Add the 4 archs** in `src/temp_bench/archs/` (plugin drop + `configs/archs.yaml`
   entries with `per_section_hparams.synthetic`). Reuse patterns already in-repo:
   - BatchTopK train encode + JumpReLU-threshold eval encode + threshold EMA
     update: copy from `tsae.py` (`_encode_per_token`, the `threshold` logic,
     `tsae.py:187-214, 333-349`).
   - AuxK dead-feature revival + decoder-norm: copy from `txc_base.py:174-194`
     / `tsae.py:352-363`.
   - `txc_batchtopk_*` reuse `txc_base`'s `W_enc (T,d_in,d_sae)` / `W_dec
     (d_sae,T,d_in)` and the `einsum` squash; the only change is *where* BatchTopK
     is applied (pre vs post squash) and the budget (`k_pos`/token vs
     `k_pos`/window).
   - `consumes`: `batchtopk_sae`="token"; `stacked_batchtopk`/`txc_batchtopk_*`
     follow their TopK analogues ("sequence"/"window").
2. **Throughput**: in the grid driver, set per-cell `batch_size = 1024 if T==1
   else 1024//T`. (Simplest; alternatively normalize in the trainer — but driver
   is cleaner and avoids `core/` edits.)
3. **Grid driver** `synthetic/backtracking/run_grid.py`: update `ARCH_T` to the
   11 new configs + the per-T batch_size. Keep the parallel ProcessPool + flock
   leaderboard. Untrained control = `n_steps=0` at `d_sae=20`.
4. **Smoke test** one cell per new arch (`--smoke`) before the full grid;
   especially verify `encode()` at eval returns a sane threshold-gated code for
   each (the λ/eAUC probes depend on it).
5. **render_figs** `synthetic/backtracking/render_figs.py`: update `ARCH_T`,
   `COLORS`, `LABEL`, `PER_TOKEN`, `MARK` for the new arch names; re-run → it
   auto-regenerates the figures + `bench_stats.json` + the `AUTO:*` tables in
   `bench_record.md`. (Aggregation already keys on arch/T/d_sae/kind/k_pos from
   the leaderboard — no logic change needed.)
6. **Amend `bench_spec.md`** with a dated pre-run note: backbone TopK→BatchTopK
   (+ throughput fix, + post-squash `k_win//T` correction), analogous to the
   existing K=8→2 amendment. Update the § 5 grid arch list. Then update
   `bench_record.md` prose (§1 Setup + the narrative) to the new arch set;
   numbers/tables/figs auto-fill.
7. **Validate + tests** (`run.py validate`, `pytest tests/ -q`), confirm 0 broken
   links, idempotent record, commit.

Estimated: ~2–3 hr re-grid on the RTX 5090 (~67s/cell solo; run ~6 parallel).

---

## 6. Roadmap beyond the BatchTopK redo

- **Change-point / EM bench** ([`changepoint/bench_spec.md`](changepoint/bench_spec.md)):
  the shared change-point/absorbing generator (dual latent: mode = DC/per-token,
  change-point = AC/window). **Gated** — topic-switching (its cheap anchor)
  ABORTED, so it needs either a stronger topic labeler (LLM segment tagging /
  validated topic model) or the **EM** anchor (emergent misalignment), which
  needs a *paid per-span judge labeler* that does not yet exist (`evals/em.py` is
  a stub). If the goal is an EM synthetic bench, the first real work is building
  that labeler + measuring EM's within-sequence temporal signature (is onset
  predictable / sticky?) per the autoresearch loop — *then* mirror + bench.
- Once the BatchTopK backtracking result lands, the same fair-backbone grid is
  the template for any future bench (topic-with-better-labeler, EM, …).

---

## 7. Hard rules + run reference + git

- `TEMP_BENCH_ALLOW_DIRTY=1`, `.venv/bin/python`, never edit `temp_bench/core/`,
  plugin-only, everything through the canonical runner (code-version stamped),
  paper-section names. Prime directive: a sound verdict, never a "win" — don't
  tune labeler/statistic/capacity/probe/metric to manufacture one.
- Run: `cd purified && TEMP_BENCH_ALLOW_DIRTY=1 .venv/bin/python -m
  synthetic.backtracking.<gating|kernel_order|measure|mirror|run_grid|render_figs>`.
  Canonical leaderboard: `results/leaderboard.jsonl`.
- **Git:** branch `arxiv`, **pushed to `origin/arxiv`**. The BatchTopK redo
  shipped in two commits: `6d406e19` (the 4 archs + grid/renderer/spec wiring,
  pushed mid-run) and the final record+figures+leaderboard commit (pushed on
  completion). Two empty dirs (`docs/synthetic/`,
  `experiments/synthetic/__pycache__`) can be `rmdir`'d (cosmetic, untracked).
