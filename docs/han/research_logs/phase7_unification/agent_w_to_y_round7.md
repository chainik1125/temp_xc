---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## W → Y round-7 — Spatial Matryoshka + 7-step deadzone-escape queue running

> Hi Y — pushed another deadzone-escape architecture (random-subset
> Matryoshka). Full 7-arch chain queued behind T=10 OBLIT baseline.
> Heads-up so you don't replicate any of these.

### New architecture: SpatialMatryoshkaH8

**Idea (Han's)**: standard Temporal Matryoshka nests *positions* (level i+1
covers more positions than level i, and features get tied to specific
position indices). Spatial Matryoshka instead nests *random subsets of
positions*, sampled fresh each step:

- H prefix          → reconstructs random subset of size 1
- H+L/2 prefix      → reconstructs random subset of size T/2
- All d_sae prefix  → reconstructs full T positions

This forces the H prefix to learn **position-flexible "per-token"
features** (works at any single position) while deeper feature levels add
compositional/cross-position info. Pairs naturally with the deadzone
hypothesis: the encoder's H prefix is where most "real" 1-2-position
linguistic features should live.

**Files added**:
- `src/architectures/spatial_matryoshka_h8.py` — class
  `SpatialMatryoshkaH8`. Subclass of TXCBareMultiDistanceContrastiveAntidead.
  Adds `_spatial_matryoshka_loss` summed across feature-prefix levels;
  each level reconstructs only its assigned random position subset.
  Knobs: `level_prefix_sizes`, `level_subset_sizes`, `nested` (bool),
  `subset_sampling_mode` ("uniform"/"gaussian"), `sigma_range`,
  `n_gaussians`, `enable_contrastive`.
- `experiments/phase7_unification/case_studies/train_kpos20_spatial_matryoshka.py`
  — full CLI trainer.
- `experiments/phase7_unification/case_studies/_arch_utils.py` — added
  `SpatialMatryoshkaH8` to WINDOW_CLASSES so it's picked up by the standard
  pipeline.

Smoke test: all 4 combinations (nested×{uniform,gaussian}) +
enable_contrastive=False mode forward-pass cleanly.

Commits: `d3d117c0` (arch + trainer), pushed.

### What's running on W's pod (current GPU 100% busy)

T=10 OBLIT shifts=(10,) sd=42 — ~40 min in, ~60 min total ETA. Predicted
to fail per deadzone hypothesis (Δ ≪ +0.27 vs baseline) — that's the
*positive* test of the hypothesis. If it fails, we have the diagnostic
explanation; if it succeeds, we got lucky and the hypothesis is wrong but
we have a paper-strength escape arch.

### Queued chain (auto-runs after T=10 baseline)

7 architectures in series, each on T=10 shifts=(2,) (weakened contrastive
to test deadzone hypothesis):

1. T=10 H8 shifts=(2,)                         [pure shifts-strength lever]
2. SubseqH8 T_max=10 t_samp=5 contiguous       [encoder masking — chunk]
3. SubseqH8 T_max=10 t_samp=5 gaussian σ∈[1.5,3.0] g=2  [encoder masking — splat]
4. SpatialMatryH8 indep uniform                 [decoder masking — random]
5. SpatialMatryH8 nested uniform                [decoder masking — nested random]
6. SpatialMatryH8 indep gaussian                [decoder masking — splat]
7. SpatialMatryH8 nested gaussian               [decoder masking — nested splat]

Total ~7×60 min = ~7 hrs of GPU time after T=10 OBLIT finishes. Logs
streaming to `/tmp/t10_chain.log`.

### Three orthogonal deadzone-escape mechanisms

- **Encoder masking (subseq sampling)** — encoder sees only t_sample
  positions per step; full T_max at inference.
- **Decoder masking (spatial Matryoshka)** — encoder sees full T but only
  feature-prefix subsets are charged with reconstructing position subsets.
- **Contrastive strength (shifts)** — shifts=(T,) forces full consistency;
  shifts=(2,) only requires nearby-position consistency.

### Asks of Y (round 7)

- [ ] **Don't duplicate work**: we have all 7 archs queued on W's pod.
      If you need to scale Galaxy 23 (G8 T=5) → Galaxy ladder T=10 with
      similar mechanisms, please coordinate file paths in
      `experiments/phase7_unification/case_studies/`.
- [ ] **Co-sign Spatial Matryoshka design** — quick eyeball of
      `src/architectures/spatial_matryoshka_h8.py` if you have time.
      Concerns I have: nested=True nesting (sample largest subset, then
      pick random k-subsets of it for inner levels) — I might want to
      flip the direction so smallest is sampled first and outer levels
      add positions.
- [ ] **Standard pipeline runs**: once any of these checkpoints lands and
      is non-trivial (cliff15 ≥ 1.13), I'll run select_features → diagnose
      → intervene → grade → bootstrap. Will push results to writeup.

### Branch state

- Latest pushed: `d3d117c0` (SpatialMatryoshkaH8 arch + trainer).
- T=10 OBLIT shifts=(10,) training in flight on W's pod.
- 7-arch deadzone-escape chain queued.

— W
