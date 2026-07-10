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

**Per-token matched is the primary comparison.** Holding `l0_per_token = B*`
equal is the *controlled* experiment: it equalizes the total atom budget spent
describing any span, so the only remaining variable is decode structure (one
joint window code vs `T` independent per-token codes). A token arch is the `T=1`
base case; a window arch gets `k_win = B*·T`. This is the convention the
individual bench records used.

The **per-window** convention (`l0_per_window = B*`, so per-token = `B*/T`
shrinks with `T`) is a *skeptic's check*, not a second fairness baseline: it asks
whether a window arch's advantage survives losing the budget growth that longer
windows otherwise grant. It cannot be a clean program matrix (its `B* ≥ T`
requirement — pre/stacked fire ≥1 atom/position — collides with deep-scarce
`d_sae` for `T>2`), and its signal *scales with T*. So it lives as a **per-token
vs per-window overlay across `T`** in each bench's own frontier, where it's
legible — not here.

**3. Capacity.** Each bench's canonical cells sit at **`{F, F/2}`** (boundary +
deep-scarce), a uniform rule for every bench. `F` is per-bench (usually the
feature-direction count; for frequency it's the alphabet `M=101`, footnoted
below) — cross-bench comparability rides on the normalization above, not on an
identical absolute `d_sae`.

<!-- BEGIN AUTO:operating_point -->
Per-token matched: window **T = 4** (token archs T=1), matched to **B\* = 2** atoms/token (nearest realized `l0_per_token`; loose match >1 marked `*`). Each cell is normalized recovery `mean` over seeds, `[chance=0, oracle=1]`, shown at **F, F/2** (`boundary / deep-scarce`):

- **backtracking**: d_sae ∈ {20, 10} (F=20; feature-direction count (1 backtrack + 19 content).)
- **signed_motion**: d_sae ∈ {19, 9} (F=19; feature-direction count (19 step directions).)
- **changepoint**: d_sae ∈ {20, 10} (F=20; feature-direction count (8 mode-signature + 12 content).)
- **frequency**: d_sae ∈ {101, 50} (F=101; alphabet M=101 (NOT a direction count): the circle embedding is rank-2 (all M symbols in a 2-D plane), so {101, 50} are alphabet-scaled capacities, both < the memorization budget |Ω|·M=1010.)
<!-- END AUTO:operating_point -->

---

## Matrix — per-token matched

Each cell is normalized recovery at `d_sae = F / F/2` (boundary / deep-scarce);
`—` = no grid cell there yet; `*` = loose realized-L0 match. Per-bench `d_sae`
values and the operating point are printed above. Run the uniform re-grid to fill
holes.

<!-- BEGIN AUTO:matrix_pertoken -->
| bench · latent (DC/AC) | Per-token SAE | T-SAE (contrastive) | Stacked (per-position dicts) | TXC-pre (additive) | TXC-post (coincidence) | Spectral-TXC (DCT bands) |
|---|---|---|---|---|---|---|
| **backtracking** · λ — self-exciting intensity (linear-in-history) (AC) | — / — | — / — | — / — | — / — | — / — | — / — |
| **signed_motion** · sign — ±1 order-sensitive step (AC) | — / — | — / — | — / — | — / — | — / — | — / — |
| **changepoint** · mode m_t — global hidden state (DC) | — / — | — / — | — / — | — / — | — / — | — / — |
| **changepoint** · time-since-switch (primary AC latent) (AC) | — / — | — / — | — / — | — / — | — / — | — / — |
| **changepoint** · change-point c_t — adjacency floor (AC) | — / — | — / — | — / — | — / — | — / — | — / — |
| **frequency** · velocity Y — cyclic tone f = Y/M (AC) | — / — | — / — | — / — | — / — | — / — | — / — |
<!-- END AUTO:matrix_pertoken -->

Per-window matching is **not** shown here — it is a per-`T` overlay in each
bench's own frontier (see `bench_record.md`); a verdict that survives per-window
too is flagged there as convention-independent.

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
