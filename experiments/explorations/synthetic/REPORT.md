# Synthetic-benchmark program — B×A comparison report

**Auto-generated.** Every number below is rebuilt from the canonical leaderboard
(`results/leaderboard.jsonl`) by
`-m experiments.explorations.synthetic.render_report`; nothing is hand-typed. The
matrix spec (which benches, which architectures, the canonical operating point)
lives in [`registry.py`](registry.py). Per-benchmark detail — the full `d_sae`
and `T` frontiers, controls, and prose verdict — stays in each bench's own
`bench_record.md`; this file is the *cross-benchmark summary*, not a replacement.

---

## How the grid is made comparable (frozen)

The program compares **B benchmarks × A architectures**. Two things are not
natively comparable; both are resolved so a cell is apples-to-apples.

**1. Cross-benchmark metric.** Each benchmark scores latent recovery with a
held-out **linear** probe on the identical `L`-tiling, normalized to
`[chance = 0, oracle = 1]`. That single normalized scalar is comparable across
benchmarks. A **dual-latent** benchmark contributes **one matrix row per
latent-axis** (e.g. changepoint → a DC `mode` row and an AC `time-since-switch`
row), so the split is shown, not averaged away.

**2. Cross-architecture budget.** Architectures are matched on **realized L0** —
the code density actually fired, measured on the shared eval windows
(`l0_per_token` / `l0_per_window`), **not** the nominal `k_pos` knob. They
diverge sharply: at `T=8`, nominal `k_pos=4`, a per-token SAE fires 4/token,
TXC-pre ~2/token, TXC-post 0.5/token. Matching on the knob would compare unequal
densities; matching on realized L0 does not.

There are **two legitimate fairness conventions**, and both are *slices of one
grid*, differing only in the per-token budget held as the window `T` grows:

| convention | held fixed | per-token budget across T | question |
|---|---|---|---|
| **per-position** | `l0_per_token = B*` | constant | at equal per-token budget, does joint temporal allocation help? |
| **per-window** | `l0_per_window = B*` | `B*/T` (shrinks) | at equal window-description budget, does one joint code beat T independent per-token codes? |

A token architecture is the `T=1` base case, so the two conventions coincide for
it. Under **per-position** a window arch gets `k_win = B*·T` (budget grows with
`T`); under **per-window** it is held to `B*` per window (`B*/T` per token —
starved, but free to allocate jointly). We report **both** matrices; a verdict
that flips between them is convention-dependent and flagged as such.

<!-- BEGIN AUTO:operating_point -->
Canonical cell: **d_sae = F** (per bench), window **T = 4** (token archs T=1), matched to **B\* = 4** atoms (nearest realized L0; loose match >1 marked `*`). Cells are normalized recovery `mean` over seeds, `[chance=0, oracle=1]`.
<!-- END AUTO:operating_point -->

---

## Matrix A — per-position matched (equal atoms per token)

Window archs get `k_win = B*·T`. This is the convention the individual bench
records used.

<!-- BEGIN AUTO:matrix_per_position -->
| bench · latent (DC/AC) | Per-token SAE | T-SAE (contrastive) | Stacked (per-position dicts) | TXC-pre (additive) | TXC-post (coincidence) | Spectral-TXC (DCT bands) |
|---|---|---|---|---|---|---|
| **backtracking** · λ — self-exciting intensity (linear-in-history) (AC) | — | — | — | — | — | — |
| **signed_motion** · sign — ±1 order-sensitive step (AC) | — | — | — | — | — | — |
| **changepoint** · mode m_t — global hidden state (DC) | — | — | — | — | — | — |
| **changepoint** · time-since-switch (primary AC latent) (AC) | — | — | — | — | — | — |
| **changepoint** · change-point c_t — adjacency floor (AC) | — | — | — | — | — | — |
| **frequency** · velocity Y — cyclic tone f = Y/M (AC) | — | — | — | — | — | — |
<!-- END AUTO:matrix_per_position -->

## Matrix B — per-window matched (equal atoms per window)

Window archs are held to `B*` atoms per window (`B*/T` per token); token archs
keep `B*` per token. The stringent "can a starved-but-joint code still win?"
test.

<!-- BEGIN AUTO:matrix_per_window -->
| bench · latent (DC/AC) | Per-token SAE | T-SAE (contrastive) | Stacked (per-position dicts) | TXC-pre (additive) | TXC-post (coincidence) | Spectral-TXC (DCT bands) |
|---|---|---|---|---|---|---|
| **backtracking** · λ — self-exciting intensity (linear-in-history) (AC) | — | — | — | — | — | — |
| **signed_motion** · sign — ±1 order-sensitive step (AC) | — | — | — | — | — | — |
| **changepoint** · mode m_t — global hidden state (DC) | — | — | — | — | — | — |
| **changepoint** · time-since-switch (primary AC latent) (AC) | — | — | — | — | — | — |
| **changepoint** · change-point c_t — adjacency floor (AC) | — | — | — | — | — | — |
| **frequency** · velocity Y — cyclic tone f = Y/M (AC) | — | — | — | — | — | — |
<!-- END AUTO:matrix_per_window -->

`—` = no grid cell at that (arch, T, d_sae) under this convention yet;
`*` = realized-L0 match looser than the tolerance. Run the uniform re-grid to
fill holes.

---

## Grid coverage

What (arch, T, d_sae) groups currently exist per bench on the leaderboard — the
holes the uniform re-grid will fill.

<!-- BEGIN AUTO:coverage -->
- **backtracking** (F=20): 72 (arch,T,d_sae) groups · archs: batchtopk_sae, stacked_batchtopk, stacked_sae, topk_sae, tsae, txc_base, txc_batchtopk_post, txc_batchtopk_pre
- **signed_motion** (F=19): 40 (arch,T,d_sae) groups · archs: stacked_sae, topk_sae, tsae, txc_base
- **changepoint** (F=20): 44 (arch,T,d_sae) groups · archs: batchtopk_sae, stacked_batchtopk, tsae, txc_batchtopk_post, txc_batchtopk_pre
- **frequency** (F=101): 90 (arch,T,d_sae) groups · archs: batchtopk_sae, spectral_txc, spectral_txc_dcac, spectral_txc_full, tsae, txc_batchtopk_post, txc_batchtopk_pre
<!-- END AUTO:coverage -->

---

## Per-benchmark records

- [`backtracking/bench_record.md`](backtracking/bench_record.md) — POSITIVE
- [`changepoint/bench_record.md`](changepoint/bench_record.md) — SPLIT (two-way)
- [`frequency/bench_record.md`](frequency/bench_record.md) — POSITIVE
- [`signed_motion/bench.md`](signed_motion/bench.md) — NEGATIVE

See [`STATUS.md`](STATUS.md) for the living program state and
[`README.md`](README.md) for the governing methodology.
