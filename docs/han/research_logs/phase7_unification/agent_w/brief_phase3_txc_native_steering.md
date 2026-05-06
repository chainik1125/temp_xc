---
author: Han
date: 2026-04-30
tags:
  - design
  - in-progress
---

## Agent W Phase 3 brief — TXC-native steering protocols

> **Mission**: investigate whether a steering protocol that exploits the
> TXC's window structure can outperform both right-edge and per-position
> on the matched-sparsity benchmark. **Phase 3 starts here**; Phase 1
> sweep + Phase 2 axes 1, 3, 4 are wrapped in
> `agent_w/2026-04-30-w-final-summary.md`.

### Why this question

Phase 1+2 settled the matched-sparsity matrix: at k_pos=20, all 8 trained
TXC cells either TIE or WIN against the multi-seed-pooled T-SAE k=20
anchor (0.70). Y's T=2 cells WIN; T=3, T=5, matryoshka all TIE. **But
the comparison is weighed down by the σ_anchor = 0.80 instability**
under the constrained metric, and by the observation that current
protocols don't really exploit what makes TXC different from a
per-token SAE in the first place.

The current protocols viewed honestly:

| protocol | encoder T-integration | decoder T-output | residual write |
|---|---|---|---|
| right-edge | T tokens → 1 z | 1 of T slices kept | every position from T-1 to S-1 (single-slice delta per position) |
| per-position | T tokens → 1 z | all T slices used | every position 0..S-1 (averaged across overlapping windows) |

Right-edge wastes T-1 of the T decoder slices. Per-position uses all
slices but averages them across overlapping windows so the resulting
per-position delta is a smeared average rather than a coherent T-block
write. Neither protocol applies the full (T, d_in) decoder output as a
*coherent T-token concept-write* — exactly the operation the TXC was
trained to support.

Phase 3 hypothesis: **A protocol that writes the full (T, d_in) decoder
output coherently at the T positions the encoder integrated** should
outperform both. The matched-sparsity TIE band is already established;
this is the search for the extra +0.27 that converts TIE → WIN.

### Hypothesis space — protocol variants to test

Naming convention: `intervene_paper_clamp_window_<variant>.py`, output
`steering_paper_window_<variant>/<arch>/{generations,grades}.jsonl`.

#### V1 — Local-T-window (the priority)

At each generation step, encode the **most recent** T-token window
once, decode the full `(T, d_in)` reconstruction, clamp the picked
feature in z, decode the clamped version, take the (T, d_in) delta,
and write each of the T slices at its corresponding absolute position
in the most recent T-window. **Leave the rest of the prefix
un-modified.**

```text
hook fires once with residual of shape (B, S, d_in):
  window_last       = residual[:, S-T : S, :]                # (B, T, d_in)
  z                 = TXC.encode(window_last)                # (B, d_sae)
  z_clamped         = z.clone(); z_clamped[:, picked] = s_abs
  delta_TxD         = TXC.decode(z_clamped) - TXC.decode(z)  # (B, T, d_in)
  residual_steered  = residual.clone()
  residual_steered[:, S-T : S, :] += delta_TxD               # write ALL T slices
return residual_steered    # only positions S-T..S-1 modified
```

**TXC-native because**: encoder integrates over the same T tokens the
decoder writes back to. One encoder forward per generation step (vs. S-T+1
in current protocols → faster). Decoder output used in full. Steering
localized to the "active" window the encoder was designed to characterise.

**Expected behaviour**: more controllable; less prefix saturation; the
model's recent-context attention sees a coherent T-token concept span
while long-range attention sees natural prose.

**Predicted matrix shifts**:
- T=10 + V1 should *help* (more decoder slices used coherently). T=10
  was untested under existing protocols (W's cell F was skipped).
- T=5 matryoshka × V1 should be the strongest combination (matryoshka
  needs the full decoder output to use both H/L groups; V1 provides it
  coherently).
- T=2 + V1 may not help (T=2 too narrow to make coherent T-block
  writing meaningful — Y's T=2 win was driven by minimal encoder
  averaging at low T, which V1 doesn't change).

#### V2 — Local-T-window with start-of-prompt anchor

Same as V1, but *also* apply at the prompt's beginning (e.g., positions
0..T-1). Two T-blocks of steering: at the topic-setting opening and at
the active generation tip.

Predicted: more sustained concept anchoring without full-prefix
saturation. May help concepts that need topic-setting (knowledge,
domain) — the start-of-prompt write biases the topic without forcing
the model to drift back to default mid-generation.

#### V3 — Decoder-direction additive (no encode)

Skip the encode + clamp protocol entirely. Just additively scale the
picked feature's decoder direction:

```text
delta_TxD = s_abs × W_dec[picked, :, :]    # (T, d_in), the unit-norm direction × strength
residual_steered[:, S-T : S, :] += delta_TxD
```

**TXC-native because**: directly uses the decoder's per-position
direction without any encode / clamp / decode round-trip. The "natural"
direction of the picked feature, scaled.

Predicted: cleaner than encode/clamp/decode at the cost of losing
paper-clamp's "isolate this feature only" property. Should be a useful
ablation: if V3 ≈ V1, the encode/clamp ceremony is doing nothing useful;
if V3 < V1, the SAE's per-position cross-feature interactions matter.

#### V4 — Tiled non-overlapping (cheap full-prefix variant)

Tile the prefix into non-overlapping T-blocks. Encode each block once,
decode, clamp, get T-position delta, write to its T positions. No
averaging. Each position has exactly one T-block contribution.

```text
for each non-overlapping T-block in residual:
  z_block          = TXC.encode(block)
  z_block_clamped  = z_block.clone(); z_block_clamped[picked] = s_abs
  delta_block      = TXC.decode(z_block_clamped) - TXC.decode(z_block)  # (T, d_in)
  residual_steered[block_pos: block_pos+T, :] += delta_block
```

Predicted: between right-edge and per-position. Faster than
per-position (no overlap = ⌈S/T⌉ encodes vs. S-T+1). Coherent T-block
writes (no averaging across windows) but creates discontinuities at
block boundaries.

### Test plan

#### Stage 1 — V1 on existing ckpts (priority)

Use existing trained ckpts (cells from Phase 1+2). No new training needed.

| arch | T | k_pos | seed | already trained? |
|---|---|---|---|---|
| txc_bare_antidead_t2_kpos20 (Y) | 2 | 20 | 42, 1 | ✓ both seeds |
| txc_bare_antidead_t3_kpos20 (W cell C) | 3 | 20 | 42, 1 | ✓ both seeds |
| txc_bare_antidead_t5_kpos20 (Y) | 5 | 20 | 42, 1 | ✓ both seeds |
| agentic_txc_02_kpos20 (W cell E) | 5 | 20 | 42 | ✓ seed=42 |
| txc_bare_antidead_t5 (canonical k_pos=100) | 5 | 100 | 42 | ✓ |
| agentic_txc_02 (canonical k_pos=100) | 5 | 100 | 42 | ✓ |

Per cell: select_features (already done) + diagnose_z_magnitudes
(already done) + V1 intervene (~10–15 min) + grade (~10–15 min). **Total
~25–30 min/cell.** 6 cells × 30 min = ~3 hours total.

Compute the V1 matrix and compare against right-edge and per-position
matrices already in `steering_paper_normalised{,_seed1}/` and
`steering_paper_window_perposition{,_seed1}/`.

#### Stage 2 — V3 (decoder-additive ablation) on V1 winners

If V1 lands a +0.27 win on any cell, also run V3 on that cell to
ablation-test whether the encode/clamp ceremony matters. ~25 min/cell.

#### Stage 3 — V2 (start-of-prompt anchor) on V1 winners

If V1 lands a knowledge-class win, V2 might amplify it. ~25 min/cell.

#### Stage 4 — Cell F (T=10, k_pos=20) trained under V1

If V1 looks promising at T=5, train T=10 cell (~60 min train + 30 min
V1 pipeline = ~90 min). T=10 is where V1 should shine if the "use full
decoder output" hypothesis is right. Until V1 proves itself, skip cell
F.

### Pre-registered outcomes

Threshold rule remains ±0.27 vs **multi-seed-pooled T-SAE k=20
anchor (0.70)** under coh ≥ 1.5. Where seed=1 ckpts exist, run V1 on
both seeds and pool. Single-seed cells (cell E) get a single-seed V1
result.

**Win** (any cell V1 mean ≥ 0.97): V1 unlocks TXC-native steering. Run
V2 + V3 to characterise; train cell F to test T-extension; report as
Phase 3 headline. **Paper-grade outcome.**

**Tie band** (V1 means in [0.43, 0.97], no individual cell winning):
TXC-native protocol is *another* plausible protocol but doesn't
decisively win either. Report as: "the structure of the TXC's T-output
matters but isn't sufficient to convert TIE → WIN at matched sparsity";
explore V2 + V3 for variance reduction.

**Loss** (V1 means below 0.43): V1 hurts compared to right-edge /
per-position. Report as: "the encoder's T-integration is best used as
an averaged signal across overlapping windows (per-position) rather than
applied coherently — counter to the architectural intuition". Skip V2,
V3, F. Phase 3 winds down.

### Coordination

- **Y has wrapped Phase 2.** Y's matched-sparsity matrix is the
  baseline V1 has to beat. Y is unlikely to run V1 in parallel
  (different mission), but if she's still active she might want to
  multi-seed verify V1's wins.

- **X is on the IT-side leaderboard mission**, irrelevant to V1.

- **Z is on the leaderboard hill-climb (probe-AUC)**, irrelevant to V1.
  Z's local 5090 has different ckpts; coordinate only if Z trains a
  new TXC at k_pos=20 that V1 should also evaluate.

- **W (me) owns Phase 3 V1 ↔ V4 testing.** Will commit Phase 3 results
  under `agent_w/phase3_txc_native/` if it grows into its own subdir,
  or otherwise as `2026-04-30-w-phase3-*.md` writeups in `agent_w/`.

### Pod spec

| field | value |
|---|---|
| Hardware | RunPod A40, 46 GB VRAM, 46 GB pod RAM, 900 GB volume |
| Branch | `han-phase7-unification` (commit directly) |
| Git identity | `hxuany0@gmail.com` / `Han` |
| Push pattern | `git push "https://oauth2:$GH@github.com/chainik1125/temp_xc.git" han-phase7-unification` |
| Existing ckpts | `experiments/phase7_unification/results/ckpts/{txc_bare_antidead_t{2,3,5}_kpos20, agentic_txc_02_kpos20}__seed{42,1}.pt` (where seed=1 exists) |
| Activation cache | `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (24k seqs × 128 ctx × 2304 d, 14.16 GB; built) |
| Anthropic API | `--n-workers 1`, shared 50 req/min; uncongested if Y/X/Z aren't grading concurrently |

### Files to produce

- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_local.py` (V1 hook)
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_dec_additive.py` (V3, conditional on V1 win)
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_anchored.py` (V2, conditional on V1 win)
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_tiled.py` (V4, optional)
- `experiments/phase7_unification/case_studies/steering/run_phase3_local.sh` (chained intervene + grade for V1)
- `experiments/phase7_unification/results/case_studies/steering_paper_window_local{,_seed1}/<arch>/{generations,grades}.jsonl` (V1 outputs)
- `docs/han/research_logs/phase7_unification/agent_w/2026-04-30-w-phase3-v1.md` (V1 results writeup)
- `docs/han/research_logs/phase7_unification/agent_w/phase3-summary.md` (synthesis after Stage 1)

### Reading list

1. `agent_w/steering-pipeline-mechanics.md` — protocol mechanics + the right-edge/per-position pseudocode that V1 is replacing
2. `agent_w/2026-04-30-w-final-summary.md` — Phase 1+2 matrix + anchor-σ discovery
3. `agent_y_phase2/2026-04-30-y-final-summary.md` — Y's matched-sparsity headline
4. `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py` — right-edge hook (V1 will mirror its structure)
5. `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_perposition.py` — per-position hook (V1 will use the same `_decode_full_window` helper but write differently)

### Open question for Han

Q. Should V1 apply only at the most recent T-window (the active
generation tip, leaving older context untouched) or at every T-position
group across the prefix? My V1-as-described above applies only to
positions [S-T..S-1]. The "every T-position group" alternative is
basically V4 (tiled). The first is more localized; the second is more
saturated.

Default I'll pursue: V1 (most recent T-window only), with V4 as
follow-up if V1 is encouraging.

### One-line thesis

The TXC was trained to compress and reconstruct T-token windows;
steering protocols that respect that structure (V1) should be more
effective than protocols that throw away T-1 of the T decoder slices
(right-edge) or smear the T-output by averaging across overlapping
windows (per-position).
