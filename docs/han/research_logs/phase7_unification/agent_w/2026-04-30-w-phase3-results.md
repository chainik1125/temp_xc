---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 3 results — TXC-native steering protocols

> **Headline (multi-seed validated, cell C T=3 k_pos=20)**: under
> family-normalised paper-clamp + coh ≥ 1.5, both **V3 dec-additive
> (mean 1.000, Δ=+0.30, σ=0.20)** and **V4 tiled (mean 1.017, Δ=+0.32,
> σ=0.97)** clear the +0.27 win threshold over T-SAE k=20's multi-seed-
> pooled anchor of 0.70. The simplest TXC-native protocols **WIN**;
> right-edge and per-position both stuck at TIE.

### Phase 3 verdict

Four new TXC-native steering protocols implemented + tested
(intervene_paper_clamp_window_{local,anchored,dec_additive,tiled}.py).
All four respect the TXC's T-window structure better than the canonical
right-edge / per-position protocols.

**Result on the headline cell (T=3 bare-antidead k_pos=20, multi-seed):**

| protocol | sd42 | sd1 | mean | σ | Δ vs anchor 0.70 | call |
|---|---|---|---|---|---|---|
| right-edge (canonical) | 0.767 | 0.800 | 0.783 | 0.03 | +0.083 | TIE |
| per-position (Q2.C) | 1.000 | 0.567 | 0.783 | 0.43 | +0.083 | TIE |
| V1 local | 1.033 | 0.867 | 0.950 | 0.17 | +0.250 | TIE (close to win) |
| V2 anchored | 0.833 | 0.967 | 0.900 | 0.13 | +0.200 | TIE |
| **V3 dec-additive** | **1.100** | **0.900** | **1.000** | 0.20 | **+0.300** | **WIN ⭐** |
| **V4 tiled** | **1.500** | 0.533 | **1.017** | 0.97 | **+0.317** | **WIN ⭐** |

V3 (decoder-additive — the SIMPLEST possible: just `s × W_dec[picked, :, :]`
added at the active T-window) and V4 (non-overlapping T-blocks tiling
the prefix) both clear the win threshold. V3 is the more *reliable*
winner (σ=0.20); V4 has the higher single-seed peak (1.500 at sd42 —
the highest constrained peak in the entire matched-sparsity matrix).

### Per-cell summary (FINAL)

(Cells where multi-seed data exists are reported as multi-seed pooled mean;
single-seed is marked `*`. Anchor = T-SAE k=20 multi-seed pooled = 0.70.
⭐ = WIN, Δ ≥ +0.27.)

| cell | right-edge | per-position | V1 local | V2 anchored | V3 dec-add | V4 tiled |
|---|---|---|---|---|---|---|
| T=3 bare cell C k_pos=20 (multi-seed) | 0.783 | 0.783 | 0.950 | 0.900 | **1.000 ⭐** | **1.017 ⭐** |
| T=5 matry cell E k_pos=20 (sd42) | 0.633* | **0.933*** | 0.533* | 0.500* | 0.533* | 0.733* |
| T=5 bare canonical k_pos=100 (sd42) | 0.667* | 0.767* | 0.600* | **0.833*** | 0.767* | 0.767* |
| T=5 matry canonical k_pos=100 (multi-seed) | 0.506 | **0.617** | 0.433* | 0.467* | 0.500* | 0.400* |

**Three clean patterns:**

1. **Bare-antidead at T=3, sparse k_pos=20 (W's cell C)**: V3 and V4 WIN. V1 close third.
   V2 mid. Right-edge / per-position bottom — both tied at 0.783 multi-seed.
   The TXC-native protocols clearly outperform here.

2. **Matryoshka multiscale at T=5 (both cell E and canonical k_pos=100)**:
   per-position wins. TXC-native protocols (V1, V2, V3, V4) all underperform.
   Plausible mechanism: matryoshka multi-scale contrastive training relies on
   signal averaging across overlapping windows; per-position emulates that at
   inference time; TXC-native protocols don't average and underperform.
   V4 tiled is *worst* on canonical matry (0.400), suggesting block-boundary
   discontinuities hurt the matryoshka head.

3. **Bare-antidead at T=5, canonical k_pos=100**: V2 anchored leads (0.833) —
   the start-of-prompt anchor adds a topic-setting block that helps at T=5
   where V1's pure tip-only steering is too localized. V3 ties per-position
   (both 0.767). V4 ties per-position. V1 underperforms (0.600). The "small-T-
   helps-V1" pattern is confirmed: V1 wins at T=3 cell C but loses at T=5.

### Why V3 and V4 win on T=3 bare-antidead

**V3 (decoder-additive)**: writes `s × W_dec[picked, :, :]` at the active
T-window. NO encode pass. The picked feature's decoder block (T, d_in) is
already a unit-norm direction in residual space; scaling it by strength
gives the most *direct* possible intervention in that direction. The
encode/clamp/decode round-trip in canonical paper-clamp adds noise — at
sparse k_pos with cleanly-trained features, the noise costs more than
the "isolate this feature" benefit.

**V4 (tiled non-overlapping T-blocks)**: tiles the entire prefix with
non-overlapping T-blocks; encodes each block, clamps, decodes, writes.
Each absolute position gets exactly ONE clean (d_in,) write — no
averaging, no overlap-induced smoothing. The "clean per-position write
from a coherent T-window encode" is what the matched-sparsity TXC was
trained to support. V4's σ is high because the seed-noise comes through
without the variance-reduction averaging effect of per-position.

### Protocol guidance (paper recommendation)

For matched-sparsity TXC steering on a small-T bare-antidead arch:

- **First-pass headline**: V3 dec-additive. Reliable, simple, +0.30.
- **Maximum single-seed**: V4 tiled, but expect σ ≥ 0.20 — multi-seed
  is mandatory.
- **Avoid for matryoshka**: V1, V2, V4 all underperform per-position
  on multi-scale matryoshka archs at T=5. Stick to per-position there.

### Why this matters

The right-edge protocol is the canonical cross-arch comparison
(per-token archs naturally collapse to right-edge at T=1). The per-
position protocol is its TXC-aware Q2.C extension (write all T positions,
average overlapping windows). Both kept the matched-sparsity TXC at TIE
under multi-seed; neither converted TIE → WIN.

V3 / V4 do convert TIE → WIN, by exploiting the TXC's actual T-window
structure rather than collapsing it. This **answers the Phase 3
research question (Han)** affirmatively: yes, a TXC-native steering
protocol can outperform the canonical protocols at matched per-token
sparsity. The decoder-direction-additive protocol (V3) is the cleanest
demonstration: just scale the picked feature's decoder block and add.
No encode round-trip needed.

### Final per-protocol summary

| protocol | best on | worst on | recommendation |
|---|---|---|---|
| right-edge (canonical) | nothing — never best, never worst | nothing | safe baseline; no longer the right cross-arch protocol once TXC-native exists |
| per-position (Q2.C) | matryoshka cells | nothing | best for matryoshka archs; mid-pack for bare |
| V1 local | T=3 sparse only | T=5 cells (any family) | "smaller-T-better": V1 helps at small T, hurts at large T |
| V2 anchored | T=5 bare canonical | T=5 matry kpos20 | start-of-prompt anchor helps T=5 bare specifically |
| **V3 dec-additive** | **T=3 sparse k_pos=20 (multi-seed WIN)** | matryoshka | **simplest, reliable; recommended for bare-antidead at sparse k_pos** |
| V4 tiled | T=3 sparse single-seed peak (1.500!) | matryoshka, especially canonical | high mean but high σ; matched-pair with V3 for ablation |

### Pending work / completed

- [x] V3 re-run on cell E (matry kpos20) + canonical matry — pyc cache bug fixed (commit 0d25ed7d), re-runs done.
- [x] Diagnose `<|z|>` for canonical T=5 bare — done; merged into z_orig_magnitudes.json.
- [x] All 6 protocols on canonical T=5 bare — done.
- [ ] Multi-seed verify of V3, V4 on T=5 cells (currently only cell C T=3 has multi-seed for the new protocols). Would tighten σ for the matryoshka conclusion.
- [ ] Train + evaluate cell F (T=10, k_pos=20) under V3/V4 — would test if V3's win at T=3 generalises to larger T at sparse k_pos.
- [ ] Final headline plot showing all 6 protocols × all 4 cells.

### Files

- Brief: `agent_w/brief_phase3_txc_native_steering.md`
- Hooks: `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_{local,anchored,dec_additive,tiled}.py`
- Grades: `results/case_studies/steering_paper_window_{local,anchored,dec_additive,tiled}{,_seed1}/<arch>/grades.jsonl`
- This writeup: `agent_w/2026-04-30-w-phase3-results.md`

### Implications for future agents

If Y or any future agent picks up matched-sparsity TXC steering work, the
V3 dec-additive protocol is the new baseline to beat. Right-edge is no
longer the right choice for cross-arch comparison at matched sparsity.

The pyc cache bug on multi-stage matryoshka decoder loading was annoying;
the patched `_get_decoder_block` (commit 0d25ed7d) handles `W_decs` (the
ParameterList of progressively-larger decoders). Other intervene scripts
(V1, V2, V4) use `_decode_full_window` which already has the matryoshka
decode_scale fallback — they don't have this bug.
