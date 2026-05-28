# FreqBench — re-analysis + v2 port (§4 stress test)

**Author:** Aniket (building on Dmitry's 2026-05-06 run)
**Branch:** `arxiv-aniket`
**Status:** re-analysis complete; v2 port in progress.

FreqBench is the synthetic stress test for the paper's central claim —
*does a temporal architecture actually **use** temporal context, or does
it just **aggregate** samples?* It is the controlled counterpart to §4:
§4's synthetic settings (markov denoising, coupled-HMM) are **DC-type**
tasks where a slowly-varying latent is recovered from noisy emissions, so
**smoothing/aggregation alone wins** and the result cannot, on its own,
distinguish aggregation from genuine temporal filtering. FreqBench adds
the two benches that *can*:

| Bench | Latent structure | Oracle needs | Tests |
|---|---|---|---|
| **DC** | slowly-varying state | sample average over W | smoothing (≈ §4) |
| **AC** | signed velocity (Δphase) | order-sensitive comparison | genuine high-pass filtering |
| **Mixed** | M-velocity frequency ladder | per-frequency decode | frequency response shape |

Headline metric **NTPS** = (A − A_loc⋆)/(A_oracle − A_loc⋆): linear-probe
accuracy normalized between the one-token Bayes ceiling and the symbolic
temporal oracle.

## Provenance / reproducibility gap

Dmitry ran the full suite on 4 A40 pods (~1010 cells, 7 archs). **Only
the results JSON + plots + writeup + `make_plots.py` were ever committed**
(`origin/dmitry-synthetic`). The drivers (`run_freq_bench_*.py`), the
shared lib (`freq_bench_lib.py`), the archs (`freq_bench_archs.py`), and
the proposal `frequencybench_proposal.tex` ran on the pods and were
**never committed to any branch** — FreqBench is currently *not*
reproducible from the repo. His raw JSON is vendored here at
`results/freq_bench/dmitry_raw/` and his original writeup is preserved
alongside it (`dmitry_results.md`).

## Dmitry's bottom line (as written)

> DC: **yes** — windowed archs (txcdr/txc_base/tfa) climb to oracle;
> per-token archs sit at local ceiling. AC: **no — every arch fails**.
> Mixed: **no — flat frequency response**. *"They aggregate; they do not
> filter."*

The DC half reproduces exactly. The two negative results (AC, mixed) are
**too strong** — and his own committed data shows why.

## Correction (this re-analysis)

Recomputed by `experiments/freq_bench/reanalyze.py`; outputs in
`results/freq_bench/reanalysis/`.

### 1. The AC negative is a slice artifact — it peaks at `raw_k=1`, not 10

Dmitry's AC plot fixes `raw_k=10`. But the AC signal is strongly
sparsity-dependent and peaks at the **sparsest** code:

| arch | NTPS @raw_k=10 (his plot) | NTPS @raw_k=1 |
|---|---|---|
| txcdr_t2 | 0.13 | **0.42** |
| txcdr_t5 | 0.11 | **0.42** |
| tfa | 0.06 | **0.32** |
| tsae_attn | 0.03 | **0.27** |

At `raw_k=10` the code is too dense and the signal washes out — exactly
the slice the AC figure shows. (`ac_ntps_by_rawk.png`.)

### 2. The shuffle/reverse order-controls (never analysed) are decisive

Every row carries `A_shuffle` (tokens shuffled before probing) and
`A_reverse` (sequence reversed), never used in the writeup. At the strong
cell (W=16, raw_k=1, σ=0.1), chance = 0.5:

| arch | A (ordered) | A_shuffle | A_reverse |
|---|---|---|---|
| txcdr_t2 | **0.711** | 0.470 | **0.276** |
| txcdr_t5 | 0.710 | 0.495 | 0.283 |
| tfa | 0.662 | 0.497 | 0.350 |
| tsae_attn | 0.636 | 0.493 | 0.388 |
| regular_sae / tsae_bhalla | ~0.50 | ~0.50 | ~0.50 |

Shuffling collapses accuracy to chance; **reversing drives it *below*
chance** — the probe confidently predicts the *flipped* sign. That is the
textbook signature of a representation encoding **signed direction**, not
mere aggregation. The per-token archs stay flat at 0.5, exactly as they
must. (`ac_order_controls.png`.) `txc_base` shows the same dependence with
an inverted sign convention (ordered < chance, reverse > chance).

The Mixed bench shows the same, weaker, pattern: txcdr_t5's order gap
(A − A_shuffle) is +0.14 unsigned / +0.11 signed, again concentrated at
W=16, raw_k=1. (`order_gap_summary.png`.)

### Corrected conclusion

Sliding temporal crosscoders (txcdr) and the attention archs (tfa,
tsae_attn) **do** carry order-sensitive AC structure at long windows and
sparse codes. The linear-probe NTPS *understates* this because it can't
reach the multivote oracle (A_oracle=1.0), but the order controls are
unambiguous. Dmitry's "they aggregate, they do not filter" holds only for
the per-token archs and the dense-code regime; it is not a property of the
temporal architectures per se. This **strengthens** §4 rather than
undercutting it: the temporal-filtering capability is real, just weak and
sparsity-gated on this small-capacity (d_sae=40) synthetic data.

## v2 reproduction + capacity finding (fresh training)

The port (`experiments/freq_bench/`) was re-run from scratch on 3× A40 —
56 AC cells through the canonical leaderboard pathway (full `raw_k × W`
cross at d_sae=40 + a capacity slice). Driver:
`experiments/freq_bench/sweep.py`; analysis + plots:
`analyze_sweep.py` → `results/freq_bench/v2_sweep/`.

**1. The re-analysis reproduces under fresh v2 training.** At Dmitry's
capacity (d_sae=40), W=16, raw_k=1, the order-controls match his archived
numbers: per-token `regular_sae` flat at chance; `txcdr_t2/t5` ordered
≈0.64, shuffle→chance, **reverse≈0.37 (below chance)**.
(`order_controls.png`, `ntps_by_rawk.png`.)

**2. The AC negative is capacity-limited — a new result.** Widening the
dictionary recovers genuine signed temporal filtering (W=16, raw_k=1):

| arch | NTPS @d_sae=40 | @256 | @1024 | A_reverse @1024 |
|---|---|---|---|---|
| regular_sae (per-token) | 0.01 | 0.00 | 0.01 | 0.50 (flat) |
| txcdr_t2 | 0.37 | 0.44 | **0.51** | 0.23 |
| txcdr_t5 | 0.30 | 0.59 | **0.72** | **0.12** |

`txcdr_t5` climbs to NTPS=0.72 with A_reverse=0.12 (strongly
direction-encoding), while the per-token baseline stays pinned at 0 for
*every* capacity. So the gain is specific to the temporal architectures,
not a generic "bigger dictionary → easier probe" artifact. This answers
Dmitry's pre-registered capacity caveat: **the d_sae=40 AC negative
under-stated the temporal archs' ability; they DO filter, given capacity.**
(`capacity.png`.)

**3. The information is mostly linearly readable.** The nonlinear (MLP)
probe tracks the linear NTPS closely (e.g. txcdr_t5 @1024: linear 0.72 vs
MLP 0.78) — a modest lift, not a regime change. So the linear-probe NTPS
is not badly under-reading; the limiting factor was capacity, not probe
class. (`linear_vs_mlp.png`.)

**4. Weight-space confirmation (FreqFrac).** The encoder atoms carry real
AC energy: `freqfrac`≈0.50 for txcdr_t2 (T=2 ⇒ DC+Nyquist split) and
≈0.70–0.74 for txcdr_t5 — i.e. the trained atoms detect transitions, not
just averages. Per-token archs have no temporal axis (freqfrac undefined).
(`freqfrac_by_rawk.png`.)

## What's left to do

1. **[done]** Re-analysis from committed JSON.
2. **[done]** v2 plugin port (DC/AC/Mixed datasources + NTPS/order-control/
   MLP/FreqFrac evaluator + `freq_bench` experiment). 3 contract tests.
3. **[done]** GPU AC sweep with raw_k facet + capacity slice → capacity
   finding above.
4. **[done]** Nonlinear (MLP) probe + spectral FreqFrac, folded into the
   evaluator (protocol 1.2.0).
5. **Extend the capacity sweep to the Mixed bench** (and to `tfa`/`tsae`):
   does the frequency-response curve sharpen with capacity too? Datasource
   `fb_mixed_unsigned_W16_s10` + the mixed generator are already wired.
6. **σ sweep** (0, 0.05, 0.25) to map the noise–capacity trade-off; the
   generator takes `sigma`, only σ=0.1 was swept here.
7. **DC bench cells** through the v2 port (datasource `fb_dc_W8_p65_s10`
   exists) to reproduce the per-token-vs-windowed split as a sanity anchor.
8. **Cleanup / paper integration**: drop or clearly flag the `tsae_attn`
   mislabel; decide which figure (capacity vs order-controls) anchors the
   §4 stress-test narrative.
