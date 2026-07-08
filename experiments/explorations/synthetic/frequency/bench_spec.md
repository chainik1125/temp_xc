# Synthetic benchmark spec — cyclic-tone frequency response (periodic axis)

**Status:** **FROZEN 2026-07-08** — § 8 gating due-diligence **PASS**
(`results/frequency_gating_stats.json`, commit `9d074219`). The three open
design decisions are settled and recorded as dated **amendments A1–A6** (§ 10
below; A1–A4 pre-run/gating, A5–A6 post-run validity refinements); the § 3/§ 5/§ 8
body is otherwise unchanged. From here the spec is frozen: further changes are
dated amendments only (like changepoint A1–A4).

**Provenance + honest framing.** This is a **synthetic-first architectural
discriminator** for the **periodic / frequency** axis — the one axis the suite
does not cover (Appendix B lists "rhythmic/periodic → period+SNR" with no
bench; the DC/AC `frequency_lens` idea is its home). It is **not** a
measure→mirror bench: there is no measured real-language periodicity anchoring
it. It is motivated by (i) Engels et al. 2024 (real LLMs embed cyclic concepts
— weekdays, months — as circles) and (ii) the FrequencyBench sprint on
`origin/dmitry-spectral-sprint2`
(`docs/dmitry/sprints/2026-06-10_freqbench_sprint/`), whose circle-tone task +
DCT-band spectral crosscoder this ports onto our fair backbone and conventions.
Like signed_motion, it is a discriminator (Part II conventions: § 8 gating +
architecture grid), and a real "measure" step on LM periodicity is **deferred**.
Ported/re-derived here through our conventions because the sprint result is
sprint-grade (1–3 seeds, plain-TopK backbone, stacked-concat probe); this bench
re-runs it on the BatchTopK fair backbone with the memorization-free per-tile
probe and multi-seed grid.

---

## 1. What it tests

A symbol walks a cyclic alphabet `Z_M` at a **hidden velocity `Y`**
(`Q_{t+1} = Q_t + Y mod M`); activations `x_t = u_{Q_t} + σ·ε_t`. Under a
**circle embedding** (`u_a = R·[cos 2πa/M, sin 2πa/M]`, `R` a random isometry),
each velocity `Y` becomes a genuine **temporal tone** at frequency `f = Y/M`
cycles/token, and recovering `Y` is classical single-tone spectral estimation
(ML decoder = periodogram peak-pick). The bench measures the **frequency
response `S(f)`** — at which temporal frequencies each architecture's code
makes the velocity linearly decodable — plus whether a **frequency-split
(spectral) crosscoder** decomposes its dictionary by band.

The velocity is invisible per-token (single token uniform over symbols → zero
info) and, for a linear reader of additive-over-time per-token codes, provably
at chance. Only a window code that mixes positions nonlinearly can expose it —
the same conversion axis as backtracking/signed_motion, now with a **frequency
structure** on the latent.

## 2. Generative process

`cyclic_tones()` (to add to `src/temp_bench/data/synthetic.py`), two embedding
modes:

- **circle (headline):** `u_a = R·[cos 2πa/M, sin 2πa/M]`, `R` a random `d_in×2`
  isometry. Velocity `Y` → tone `f = Y/M`. ML oracle = periodogram.
- **random (symmetry null):** `(u_a)` = random orthonormal frame in `d_in ≥ M`.
  Proven **flat** response: for prime `M` and exchangeable embeddings, relabel
  `a ↦ c·a` maps velocity `Y ↦ cY` bijectively, so all velocities are
  statistically equivalent — **no frequency axis** (the ratio-invariance
  theorem). This is the built-in negative control.

Per sequence: `Y ~ Unif(Ω)`, phase `B ~ Unif(Z_M)`, `Q_t = (B + Y·t) mod M`,
`x_t = u_{Q_t} + σ·ε_t`. Expose in `extra`: `velocity_labels` (Y),
`frequency` (Y/M), plus the embedding mode.

## 3. Ground truth — **OPEN, settle in gating**

- **Feature directions / reconstruction codebook:** the `M` symbol atoms
  `{u_a}`. For the **circle** mode these are `M` points on a 2D circle (NOT
  orthonormal) → `eAUC` (cosine-AUC) is replaced/supplemented by codebook
  recovery + NMSE; anchor `d_sae` on **`M`** (the meaningful capacity axis),
  NOT on 2. For the **random** mode they are `M` orthonormal directions (F = M,
  like signed_motion). **Memorization threshold = `|Ω|·M`** distinct clean
  windows (not `2F`) — mark it on the `d_sae` axis; the memorization-free probe
  keeps features `< |Ω|·M`.
- **Dynamical latent — velocity `Y`** (categorical, `|Ω|`): the frequency.
  Chance = `1/|Ω|` (uniform). Oracle = periodogram/DFT matched-filter accuracy
  on the noisy windows.

## 4. Task + metrics

- **`velocity_recovery` (headline):** multinomial-logistic probe on the code →
  `Y`, per-tile leading edge, **memorization-free** (features = one tile's
  `d_sae` code, never concatenated — the signed_motion fix), sequence-split,
  normalized to [chance = `1/|Ω|`, oracle = periodogram acc].
- **`S(f)` frequency response:** the SAME recovery reported **per Ω-class**
  (per frequency) — the deliverable curve. Interpret the low-`f` behaviour
  against the **Rayleigh cell** `|Δf| < 1/W` (window resolution).
- **Band decomposition (spectral arch):** per-branch probes — each tone should
  be decoded by the branch owning its DCT band (boundary tones shared).
- **`eAUC`/codebook recovery + `NMSE`:** capability-vs-artifact (the spectral
  winner must also reconstruct, not just recover the latent).

Linear probes mandatory (conventions § 5). Chance/oracle computable (we own the
generator; oracle is the periodogram).

## 5. Grid (per conventions)

- **archs (all BatchTopK fair backbone):** per-token `{batchtopk_sae, tsae}`
  (T=1); window `{txc_batchtopk_pre, txc_batchtopk_post, stacked_batchtopk,
  spectral_txc}` (T ∈ {2,4,8}). Throughput normalised (`batch = 1024/T`, equal
  `B·T` pool), `k_win = k_pos·T`.
- **`spectral_txc` (NEW arch):** `SpectralTXCBatchTopK` — window crosscoder with
  encoder/decoder kernels constrained to **DCT bands** (multiband: 4 bands
  DC/low/mid/high, `H/4` atoms + `k_win/4` budget each; dcac: 2 bands). Vanilla
  TXC = the single-full-band special case (DCT = orthonormal rotation of the
  time axis). **Re-implement** on `_TXCBatchTopKBase` (the sprint's `SpectralTXC`
  is plain TopK — port the band parameterisation onto BatchTopK→JumpReLU + AuxK
  + decoder unit-norm + grad-orth + per-branch budgets).
- **`d_sae`:** anchored on the ground-truth codebook count (§ 3) — scarce +
  overcomplete; mark `|Ω|·M` (memorization) on the axis.
- **`k_pos`:** 1 (+ a `k_pos = 2` robustness anchor).
- **window:** the emission window `W` (sprint used 16) tiled into a common
  `L = 2^k`; `T ∈ {2,4,8}`. Reconcile `W`/`L`/`T` in gating.
- **datasources:** `toy_cyclic_circle_*` (headline) **and** `toy_cyclic_random_*`
  (null). seeds {1,2,42}. + untrained-encoder control.

## 6. Validity controls

- **Provable per-token floor:** single token uniform → `I(Y; x_t) = 0`; and any
  additive-over-time readout of per-token codes has velocity-independent means →
  linear per-token probe at chance regardless of `d_sae`.
- **Memorization-free per-tile probe** (features < `|Ω|·M`); split by sequence.
- **Untrained-encoder control:** a claimed spectral/window advantage must vanish
  at random init.
- **Symmetry null:** the random-embedding response must be flat (verifies the
  theorem; separates "frequency" from "symbol overlap").
- **Capability-vs-artifact:** spectral winner also reconstructs (codebook/NMSE).
- **Realistic regime:** any win must hold at `d_sae ≤ codebook count`, well below
  `|Ω|·M`.

## 7. Preregistered predictions

- **P1:** per-token `velocity_recovery` ≈ chance (provable), flat across `d_sae`.
- **P2:** window archs recover `Y` for `f ≳ 1/W`, with a possible low-`f` dip
  (Rayleigh); the `S(f)` curve is the deliverable. (Sprint found vanilla TXC
  mildly *high*-pass — a preregistered guess to confirm/falsify, not assume.)
- **P3:** `spectral_txc` (multiband) ≥ vanilla TXC AND shows **band-attributed
  atoms** (per-branch decomposition verified). **Genuine uncertainty** whether
  it strictly *beats* vanilla on the single-tone task (sprint: ≈ tie at mid
  capacity, multiband better only at capacity extremes / seed-stability; the
  decisive multiband win needed *superposition*, out of this scope). A tie with
  a cleaner decomposition is a complete, reportable outcome.
- **P4 (null):** random-embedding response flat; per-frequency recovery shows no
  frequency ordering (confusion tracks symbol overlap, not `Δf`).
- **Possible negatives (all reportable):** (a) no spectral advantage on
  single-tone → the multiband value is superposition-only; (b) memorization
  above `|Ω|·M` inflates recovery → caught by the per-tile probe, flagged.

## 8. Gating due-diligence — **RUN FIRST, then freeze this spec**

Write `frequency/gating.py` → `results/frequency_gating_stats.json`. Confirm:
1. **Ground truth settled** (§ 3): codebook/`F` definition, `d_sae` anchor, the
   `|Ω|·M` memorization threshold, and the final `M / d_in / Ω / W / L` (open
   decisions — see below).
2. **Per-token velocity ceiling ≈ chance** (provable + empirical).
3. **Window oracle = periodogram accuracy** (near-1 for `f ≳ 1/W` at the chosen
   SNR); per-frequency oracle spread shows the Rayleigh structure.
4. **Random-embedding response flat** (the ratio-invariance null).
5. **Separation gate:** circle window ceiling − per-token ≥ the usual bar on the
   resolvable band.

### Open design decisions (settle in gating, then freeze)

1. **What `F` / the codebook is for the circle mode** (§ 3) — biggest question;
   anchor `d_sae` on `M`, not 2.
2. **`M`, `d_in`, `Ω`.** Null needs `d_in ≥ M`. Candidates: `M=101` (prime, for
   the ratio theorem) with `d_in=128`; or `M=61` with `d_in=64` (matches the
   family). Prime `M` matters for the null. `Ω` should span DC→high DCT bands of
   `W` (sprint: `{0,1,2,4,8,16,24,32,40,50}` for M=101, W=16).
3. **`W` / `L` / `T` reconciliation** — the emission window `W` vs the common
   tiled eval `L` vs the arch windows `T ∈ {2,4,8}` (powers of two dividing L).

## 10. Pre-run amendments (dated; settle §§ 3/8 open decisions)

All four are the direct output of the § 8 gating run (`gating.py` →
`results/frequency_gating_stats.json`, commit `9d074219`), which **passed every
gate**. They freeze the settled numbers; nothing here changes the preregistered
predictions (§ 7).

**A1 (2026-07-08) — the settled `M / d_in / Ω / σ / seq_len / L / T`.**
`M = 101` (prime → the ratio-invariance null is a clean 2-orbit structure
`{0} ∪ {1..M-1}`); `d_in = 128` (≥ `M` for the random null, **unified across
both modes** — the periodogram oracle projects onto the 2-D circle plane via the
true `R`, so in-plane oracle SNR is `1/σ`, *independent of `d_in`*; only the SAE
reconstruction task feels `σ√d_in`); `Ω = {0,1,2,4,8,16,24,32,40,50}` (|Ω| = 10,
chance = 0.1; the sprint ladder, DC→Nyquist `f = Y/M ∈ [0, 0.495]`); `σ = 0.10`
(σ-sweep {0.05,0.1,0.2,0.25}: the circle oracle on the resolvable band is 1.00
across all — σ is not load-bearing; 0.10 is the middle choice); `seq_len = 64`
(family convention; the tone spans the whole sequence, velocity constant/seq);
`L = 32`; **`T ∈ {2,4,8,16}`** — extended from the spec's `{2,4,8}` because the
gating oracle showed the resolvable band at `T ≤ 8` is too narrow to exhibit the
full high-pass `S(f)` (the sprint's single fixed window `W = 16` maps to our
`T = 16` cell): per-Ω-class periodogram oracle at T=2 is
`[0.58,0.17,0.25,0.48,0.77,0.92,0.92,0.92,0.94,0.98]` (only high-Y resolved),
widening to all-1.00 at T=16. `T = 16` divides `L = 32` (2 tiles) — still within
the power-of-two tiling convention.

**A2 (2026-07-08) — ground truth: codebook / `F` / `d_sae` anchor /
memorization.** Circle mode: the reconstruction codebook is the **`M` circle
atoms** `{u_a}` — a **2-D object** (they lie on one circle in a 2-plane), so
`eauc` (cosine-AUC) is ill-defined (adjacent atoms are nearly parallel,
`cos 2π/101 ≈ 0.998`) and capability is read off **`NMSE`** + codebook/circle
recovery instead. `d_sae` is anchored on **`M = 101`**, NOT on 2. Random mode:
`F = M` orthonormal directions (like signed_motion) → `eauc` is meaningful.
**Memorization threshold = `|Ω|·M = 1010`** distinct clean `(B,Y)` windows (same
for every `T ≥ 2`); the memorization-free per-tile probe keeps `d_sae < 1010`.
Settled `d_sae` sweep: **{32, 64, 101, 256}** — scarce `{32,64}` + at-`F` `{101}`
+ over-complete `{256}`, all `< |Ω|·M`; mark both `M` and `|Ω|·M` on the axis.

**A3 (2026-07-08) — the `spectral_txc` per-band budget (fairness).** The DCT
bands are built **adaptively per `T`** (multiband: DC `{0}` + up to 3 contiguous
AC groups — `{1-5},{6-10},{11-15}` at `T=16`, degenerating to DC/AC at `T=2`).
BatchTopK is applied **per band** (equal-L0-per-band guarantee), with total
window budget `k_win = k_pos·T` split across bands. This puts `spectral_txc` at
the **same total-per-window support as `txc_batchtopk_pre`** (whose shared code
also supports up to `k_pos·T` atoms), so **spectral-vs-pre is the clean
band-structure test** at matched budget; `txc_batchtopk_post` (budget `k_pos`
per window) is the density-corrected monolithic companion. The sprint's plain
per-branch TopK is ported to per-branch BatchTopK→JumpReLU + AuxK + Parseval
(coefficient-L2) decoder unit-norm + grad-orth.

**A4 (2026-07-08) — the raw-linear window control (a *new* reported ceiling).**
Gating adds a control beyond § 6: a **linear (multinomial-logistic) probe on the
raw concatenated `T`-tile** recovers velocity at **≈ chance** for the circle mode
(0.11 at every `T`, vs chance 0.10), because `E[x_t | Y] ≈ 0` (the circle is
centred, `B` uniform) so the velocity lives in the **2nd moment**, not the
class-conditional mean. Consequence (as in changepoint/signed-motion): a window
*win* on a trained code is nonlinear feature-learning, not linear access; the
untrained-encoder control checks the nonlinear-access residual. (Note: the
*random* null shows a mild rise of this raw-linear ceiling with `T` — 0.11→0.18
— a symbol-identity linear signal, not a frequency one; reported, not gated.)

**A5 (2026-07-08, post-run) — drop `stacked_batchtopk` from the grid
(validity, not scope).** The § 5 arch list included `stacked_batchtopk`, but its
per-tile code is the **concatenated per-position code** (feature dim `T·d_sae`),
so for the 2nd-moment velocity a linear probe recovers `Y` only by **memorizing**
the `≤ |Ω|·M = 1010` distinct clean windows once `T·d_sae ≥ 1010` — the exact
signed-motion memorization confound (the concatenation the conventions forbid).
The shared-code crosscoders (tile-code `= d_sae ≤ 256 < 1010`) are
memorization-free, and the per-token archs already instantiate the "cannot mix
positions" null, so `stacked` is a confounded comparator here and is omitted.
(A reduction of the arch set for validity, flagged transparently; the prime
directive favours it.)

**A6 (2026-07-08, post-run) — matched-budget band-partition comparators.** The
main grid compares the arch family at **equal `k_pos`**, where each arch
*allocates* its budget differently (Spectral fires `k_win = k_pos·T` atoms,
TXC-post fires `k_pos`) — so a Spectral-vs-post gap conflates band structure with
active-atom count. To answer P3 ("does the band *partition* help?") cleanly, add
`spectral_txc_full` (1 band = the vanilla DCT crosscoder) and `spectral_txc_dcac`
(2-band DC/AC) — same class / backbone / **total budget `k_win`**, differing only
in the DCT-band split. `multiband` vs `dcac` vs `full` at matched budget isolates
the partition effect (`run_grid_bands.py` addendum). Result: a **tie** at
windows that can saturate the tone (`T ≥ 4`), with `multiband`'s edge at the
scarcest window `T=2` and in the untrained access (band-limiting is the prior) —
the preregistered P3 outcome.

## 9. Reproduction (when built)

```bash
# (from repo root /workspace/temp_xc; TEMP_BENCH_ALLOW_DIRTY=1; NO HF token needed)
.venv/bin/python -m experiments.explorations.synthetic.frequency.gating
.venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid 24
.venv/bin/python -m experiments.explorations.synthetic.frequency.render_figs
```

All cells through the canonical runner (code-version stamped). No `core/` edits;
generator + arch + evaluator are plugin drops. Single-source record pipeline
(copy the changepoint `run_grid.py` / `render_figs.py` template): leaderboard →
`render_figs` → `figs/*` + `results/frequency_bench_stats.json` + AUTO blocks in
`bench_record.md`.
