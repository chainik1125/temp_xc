---
author: Han
date: 2026-04-21
tags:
  - design
  - reference
  - in-progress
---

## Phase 5.7 architecture reference

Consolidated reference for the novel / autoresearch architectures
introduced in Phase 5.7 (the "brave archs" exploration launched off
the 25-arch canonical bench). One section per arch: the motivating
hypothesis, the math, code pointers, training dispatcher + probing
routing, param budget, and current result status.

Architectures in this doc (Tier 1 + Tier 2):

- [[#A2 — txcdr_contrastive_t5]]
- [[#A3 — matryoshka_txcdr_contrastive_t5]]
- [[#A1 — txcdr_rotational_t5]]
- [[#A5 — txcdr_basis_expansion_t5]]
- [[#A4 — mlc_temporal_t3]]
- [[#A10 — time_layer_contrastive_t5]]
- [[#A8 — txcdr_dynamics_t5]]

See [`2026-04-21-autoresearch-plan.md`](2026-04-21-autoresearch-plan.md)
for the plan, fairness rules, val/test split, and shelving decisions.

### Background: the Ye et al. 2025 T-SAE recipe

Most of the contrastive variants in this doc are ports of the Ye
et al. 2025 "Temporal SAE" (T-SAE) recipe onto window-based SAEs.
The T-SAE recipe (paper §3.2, reimplementation in
`src/architectures/temporal_contrastive_sae.py`):

1. **Matryoshka partition** of the SAE latent vector. The first `h`
   indices are "high-level" (semantic); the rest are "low-level"
   (syntactic). Paper uses h = 20 % of d_sae. Decoder columns are
   split the same way.
2. **Two reconstruction MSEs**. One uses all latents (`L_full`); one
   uses only the high prefix (`L_high`). The low slice is therefore
   forced to carry the residual the high slice cannot.
3. **Symmetric InfoNCE** on the high prefix between an adjacent token
   pair `(x_{t-1}, x_t)`: pull `z^H_t` toward `z^H_{t-1}` (diagonal
   of the B×B similarity matrix), push off-diagonal pairs apart.

Total loss `L = L_full + L_high + α · L_contr`. Paper uses α=1.0.

The key *principle* we carry forward: adjacent-same-sequence pairs
should produce similar high-level latents. When the base unit is a
single token, "adjacent" means `t−1`. When the base unit is a T-token
window (our case), we use shift-by-1 windows that overlap in T−1
positions.

Code reference for the ported single-token variant:
[`src/architectures/temporal_contrastive_sae.py`](../../../../src/architectures/temporal_contrastive_sae.py).
It's the 25-arch bench's `temporal_contrastive` row (last_position
AUC 0.7359 — underperforms, probably because a single-token SAE on
Gemma-2-2B discards most of the available window context).

### A2 — txcdr_contrastive_t5

**Hypothesis**: a window-based TXCDR with the T-SAE recipe should
combine TXCDR's strong mean_pool result (best at mean_pool: 0.8064)
with the paper's +0.8 pp contrastive lift (demonstrated by our
`mlc_contrastive` port on the MLC base).

**Architecture**:

- Base: `TemporalCrosscoder` (vanilla TXCDR, T=5). Encoder is
  per-position `(T, d_in, d_sae)`; decoder is per-position
  `(d_sae, T, d_in)`; TopK sparsity at the window level.
- **High/low split**: **flat 50/50**. First `h = d_sae // 2` latents
  are "high", the rest "low". **No sub-window scales** here — the
  split is purely by feature index, mirroring `mlc_contrastive`.
- **Matryoshka MSE**:
    - `L_full = ‖W_dec · z − W‖²` (all latents → T-window)
    - `L_high = ‖W_dec[:h, :, :] · z[:, :h] − W‖²` (high half only → T-window)
- **Contrastive InfoNCE** on adjacent windows
  `(W_{t-1}, W_t)` where `W_t = x[t-T+1 : t+1]`:
    - `z_prev_H = z_prev[:, :h]`, `z_cur_H = z_cur[:, :h]`
    - symmetric cross-entropy on the B×B cosine-similarity matrix.
- Total: `L_matr_prev + L_matr_cur + α · L_contr` with α = 0.1.
  (α=0.1 matches `mlc_contrastive`'s default that won +0.8 pp in the
  bench. The paper's α=1.0 is heavier; tunable in Part B.)

Why 50/50 and not 20/80? `mlc_contrastive` already uses 50/50 and won;
we keep the split consistent so A2's result is comparable to the
`mlc_contrastive` → `mlc` delta. Tuning the split is a Part-B knob.

**Code**:

- Class: [`src/architectures/txcdr_contrastive.py`](../../../../src/architectures/txcdr_contrastive.py)
  → `TXCDRContrastive(TemporalCrosscoder)`.
- Trainer: `train_txcdr_contrastive()` in `train_primary_archs.py`;
  uses `make_pair_window_gen_gpu(buf, T)` to sample shifted T-windows.
- Probe routing: treated like `txcdr_t5` (identical encode API); hits
  `_encode_txcdr` with `arch == "txcdr_contrastive_t5"`.

**Param budget** (d_in=2304, d_sae=18432, T=5): ≈ 425 M (same shape
as vanilla TXCDR).

**Result** (seed 42, last_position_val, k=5): **FINALIST**,
Δ_val = +0.0120 over txcdr_t5 (t=1.17, 24 wins / 10 losses).

### A3 — matryoshka_txcdr_contrastive_t5

**Hypothesis**: the existing position-nested Matryoshka TXCDR
(`matryoshka_t5` in the 25-arch bench, last_position 0.7494) was
under-delivering; adding contrastive pressure on the scale-1 prefix
should address the gap.

**Architecture**:

- Base: `PositionMatryoshkaTXCDR` (our existing `matryoshka_t5`).
  Code: [`src/architectures/matryoshka_txcdr.py`](../../../../src/architectures/matryoshka_txcdr.py).
  This is the generalisation of the paper's 2-scale Matryoshka to T
  nested *sub-window* scales:

  - Latent indices are split into T prefix groups of size `m_t`
    (default uniform: `m_t = d_sae // T`). Cumulative sizes
    `prefix_sum[t] = m_1 + … + m_t`.
  - Each scale `t ∈ {1..T}` has its own decoder
    `W_decs[t] : (prefix_sum[t], t, d_in)` and bias `b_decs[t] : (t, d_in)`.
  - **Scale-1** decoder uses only the first `m_1` latents to reconstruct
    **each single position** of the T-window.
  - **Scale-2** decoder uses the first `m_1 + m_2` latents to reconstruct
    central 2-position sub-windows.
  - **Scale-T** decoder uses all `d_sae` latents to reconstruct the
    full T-window.
  - Loss at base = mean over t of `MSE(decode_scale_t, center_sub_window_t)`.

  Terminology: **"scale-`t`"** refers to the sub-window SIZE `t`, and
  **"scale-`t` prefix"** refers to the first `prefix_sum[t]` latent
  indices. "Scale-1 prefix" = first `m_1 = d_sae // T` latents.

- **Contrastive head**: symmetric InfoNCE on the **scale-1 prefix**
  `z[:, :m_1]`. At T=5, d_sae=18432 → m_1 = 3686 ≈ 20 % of d_sae,
  matching the paper's high-level fraction.
- **Pair**: adjacent T-windows `(W_{t-1}, W_t)` shifted by 1 token.
- Total: `L_matr(W_prev) + L_matr(W_cur) + α · L_contr(scale-1 prefix)`
  with α = 0.1.

Why contrast on the scale-1 prefix specifically? Scale-1 latents are
the ONLY ones forced to reconstruct each individual position on their
own. Adjacent T-windows share T−1 of their T input positions, so the
scale-1 latents that fire for those overlapping positions should
already be largely aligned across the two windows — InfoNCE enforces
that alignment directly.

**Code**:

- Class: [`src/architectures/matryoshka_txcdr_contrastive.py`](../../../../src/architectures/matryoshka_txcdr_contrastive.py)
  → `MatryoshkaTXCDRContrastive(PositionMatryoshkaTXCDR)`.
- Default `contr_prefix = self.prefix_sum[0] = m_1 = d_sae // T`.
  Override-able via constructor.
- Trainer: `train_matryoshka_txcdr_contrastive()`.
- Probe routing: reuses `_encode_matryoshka` (encode API matches
  the position-nested base).

**Param budget**: ≈ 680 M (base ~660 M for the T nested decoder tensors
plus a shared encoder; contrastive head adds no params).

**Result** (seed 42, last_position_val, k=5): **FINALIST**,
Δ_val = +0.0155 over `matryoshka_t5` (t=1.46, 22 wins / 12 losses).

### A2 vs A3 — the partition structure is different

| axis | A2 `txcdr_contrastive_t5` | A3 `matryoshka_txcdr_contrastive_t5` |
|---|---|---|
| H/L partition | flat 50/50 on latent index | position-nested Matryoshka (T scales) |
| H prefix size | `d_sae // 2` (50 %) | `m_1 = d_sae // T` (20 % at T=5) |
| How H is trained | `L_high` MSE uses first h decoder cols | scale-1 decoder reconstructs each single position using first m_1 latents |
| Contrastive target | `z[:, :h]` | `z[:, :m_1]` (scale-1 prefix) |
| Decoder structure | single per-position `W_dec: (d_sae, T, d_in)` | T separate per-scale decoders `W_decs[t]: (prefix_sum[t], t, d_in)` |
| "Scale" concept | N/A | central to the architecture |

**One-line intuition**: A2 partitions latents *by index*, A3 partitions
them *by sub-window scale*. Both contrast on the first prefix of that
partition; both work at ~+1.2–1.5 pp Δ_val over their respective base.

### A1 — txcdr_rotational_t5

**Hypothesis** (brief.md §3.4): "feature direction rotates smoothly
across time." Parameterise per-position decoder as
`W_dec^(t) = W_base · exp(t · A)` with `A` skew-symmetric → a true
rotation of a shared base dictionary.

**Architecture**:

- Base direction per feature: `W_base ∈ R^{d_sae × d_in}`.
- Rank-K factorisation for `A`: `A = Q J Q^T` where
  `Q ∈ R^{d_in × K}` is orthonormal at init (via QR of a Gaussian) and
  `J ∈ R^{K × K}` is skew (= `0.5 (J_raw − J_raw^T)`). K=8 default.
- Effective per-position decoder:
  `W_dec^(t) = W_base · (I + Q (exp(t · J) − I) Q^T)`.
- Decode rewrites to avoid materialising `(d_sae, T, d_in)`:
  `x_hat[b, t] = (z @ W_base) + (z @ W_base @ Q) · (exp(t · J) − I) · Q^T + b_dec[t]`.
- Training: vanilla reconstruction MSE.

**Code**: [`src/architectures/txcdr_rotational.py`](../../../../src/architectures/txcdr_rotational.py).

**Param budget**: ≈ 255 M (vs vanilla TXCDR 425 M). Shared `W_base`
replaces the `T` per-position decoder stack; plus a tiny `Q` (d_in × K)
and `J_raw` (K × K).

**Result**: **DISCARD**, Δ_val = −0.0332 vs txcdr_t5 (t = −3.12, 12/23).
The rank-K constraint at K=8 was too tight to match vanilla's free
per-position decoder at 25 k steps. Could potentially be rescued with
K=16 or K=32 at more steps, but shelved (see autoresearch plan).

### A5 — txcdr_basis_expansion_t5

**Hypothesis** (brief.md §3.4): softer than rotational — per-position
decoders are a time-varying linear combination of K << T shared basis
matrices.

**Architecture**:

- `W_base: (K, d_sae, d_in)` — K=3 basis decoder matrices (shared
  across features).
- `α: (T, K)` — learnable time-coefficients. Init: `α[t, 0] = 1` for
  all t, `α[t, 1..K-1]` small sinusoidal perturbation so higher basis
  channels don't start at exact zero grad.
- Effective decoder: `W_dec^(t) = Σ_k α[t, k] · W_base[k]`.
- Decode via `z @ W_base[k]` per-k cached once: (B, K, d_in), then
  combine with `α` per-t.
- Vanilla reconstruction MSE.

**Code**: [`src/architectures/txcdr_basis.py`](../../../../src/architectures/txcdr_basis.py).

**Param budget**: ≈ 340 M (W_enc 212 M + W_base 127 M + tiny α).
Smaller than vanilla TXCDR.

**Result**: **DISCARD**, Δ_val = −0.0448 (t = −4.15, 9/26). Same
family as A1 — a different time-parameterization constraint — and
same failure mode: at 25 k steps, hand-designed structure hurts more
than it helps.

### A4 — mlc_temporal_t3

**Hypothesis**: crossing MLC's layer-axis sharing with TXCDR's
temporal axis. Input is `(B, T=3, L=5, d_in)`; encoder shared across
time; decoder per-(t, l).

**Architecture**:

- Encoder `W_enc: (L, d_in, d_sae)` shared across T. Pre-activation
  `pre[b, s] = Σ_t Σ_l x[b, t, l] · W_enc[l, :, s]` — *summed* over
  both time and layer axes (so features that "win" the TopK fire
  consistently across all T positions and all L layers).
- TopK over the flat d_sae latent.
- Decoder `W_dec: (d_sae, T, L, d_in)` — separate column per (t, l).

Input shape at inference: `(B, T=3, L, d_in)` — the three adjacent
tokens' multi-layer activations.

**Code**: [`src/architectures/mlc_temporal.py`](../../../../src/architectures/mlc_temporal.py).

**Param budget**: ≈ 849 M (W_enc 212 M shared + W_dec 637 M per-(t, l)).
Bigger than MLC (425 M) because the decoder now has T× more columns.

**Probe**: reads `acts_mlc_tail` and uses the last T positions at
last_position (`mlc_tail[:, -T:, :, :]`). Sliding window over the tail
for mean_pool.

**Result**: **DISCARD**, Δ_val = −0.0615 vs `mlc` (t = −3.54, 7/27).
Likely under-trained at 25 k steps given 2× the decoder params; also
the shared-across-time encoder removes MLC's per-input flexibility.

### A10 — time_layer_contrastive_t5

**Hypothesis**: does the InfoNCE-wins signal from A2/A3 generalise to
the joint (T, L) latent space? Base (`time_layer_crosscoder_t5`) is
already top-5 at last_position (0.7928).

**Architecture**:

- Base: `TimeLayerCrosscoder`. Encode produces `(B, T, L, d_sae)` with
  a *global* TopK over the flattened (T·L·d_sae) grid. Decoder is
  per-(t, l).
- **High prefix**: first `h = d_sae // 2 = 4096` features along the
  feature axis (same 50/50 convention as A2 and mlc_contrastive).
- **Contrastive summary**: `s = z[:, :, :, :h].mean(dim=(T, L))` —
  average over both time and layer axes to collapse to `(B, h)`. The
  average is the "what does this window represent, ignoring exact
  position and layer" summary.
- **Pair**: adjacent T-windows `(W_prev, W_cur)` shifted by 1 token,
  shape `(B, 2, T, L, d_in)`.
- Loss: `recon(W_prev) + recon(W_cur) + α · InfoNCE(s_cur, s_prev)`
  with α = 0.1.

Averaging over (T, L) is necessary because otherwise the contrastive
would be on a (B, T·L·h) tensor and the "shift by 1 token" would make
the two summaries structurally different (feature at (t, l) in W_prev
corresponds to feature at (t−1, l) in W_cur). Collapsing time and
layer resolves that.

**Code**: [`src/architectures/time_layer_contrastive.py`](../../../../src/architectures/time_layer_contrastive.py).

**Param budget**: 944 M — same shape as the `time_layer_crosscoder_t5`
base. Contrastive head adds no params (just a mean + infoNCE loss
over the existing latent). Uses d_sae = 8192 (not 18432) to fit on
48 GB A40 alongside the 18 GB multilayer buffer, inherited from base.

**Status** (2026-04-21): training.

### A8 — txcdr_dynamics_t5

**Hypothesis** (brief.md §4): features evolve across a window via a
learned sparse dynamical system rather than being independently
recomputed at each position. Latent-level analog of the contrastive
idea: contrastive *encourages* adjacent-latent similarity via a loss;
dynamics *enforces* it via architectural state.

**Architecture**:

- Initial: `z_0 = TopK(W_enc · x_0 + b_enc)`.
- Recurrence:
  `z_{t+1} = TopK(γ ⊙ z_t + W_enc · x_{t+1} + b_enc)` for t = 1..T−1.
- Per-feature gate: `γ = sigmoid(γ_raw) ∈ (0, 1)^{d_sae}`, learnable.
  `γ_raw` init = 0 → γ = 0.5 (mid-range decay at init). Each feature
  can learn its own retention rate across time.
- Encoder `W_enc: (d_in, d_sae)` and decoder `W_dec: (d_sae, d_in)`
  are **shared across time** (no per-position weights). The time-
  variation lives entirely in the z_t sequence.
- Reconstruction: `x_hat_t = z_t @ W_dec + b_dec`; loss sums MSE over
  all T positions.
- Probing: `encode(x) := z_{T−1}` (last-position state) for
  last_position aggregation; `encode_sequence(x)` returns all T
  states for mean_pool.

**Code**: [`src/architectures/txcdr_dynamics.py`](../../../../src/architectures/txcdr_dynamics.py).

**Param budget**: ≈ 85 M — drastically smaller than vanilla TXCDR
(425 M). Weights are shared across T, so `W_enc` and `W_dec` are
(d_in × d_sae) rather than (T × d_in × d_sae). This is an intentional
*capacity* vs *structural-prior* experiment; if it under-performs at
25 k steps, capacity is the confound — not the dynamics idea being
wrong per se.

**Status** (2026-04-21): queued behind A10.

### How to add a new candidate (for the next agent)

A candidate lives in four places; wiring all four is ~50 lines of
patches total:

1. **Arch class** in `src/architectures/<name>.py`. Inherit from the
   base the candidate extends (TemporalCrosscoder, MLCContrastive,
   PositionMatryoshkaTXCDR, TimeLayerCrosscoder, etc.) whenever
   possible — the class inheritance keeps the encode API identical so
   probing reuses the existing dispatch.
2. **Trainer entry** in `experiments/phase5_downstream_utility/train_primary_archs.py`:
   a `train_<name>(...)` function (use existing ones as templates —
   most are 15-20 lines) plus a dispatcher branch in `run_all`. Meta
   dict must encode enough to rebuild the model from the checkpoint.
3. **Probe routing** in `experiments/phase5_downstream_utility/probing/run_probing.py`:
   - `_load_model_for_run`: build model from meta.
   - `_encode_for_probe`: route the arch name to the right encode
     path. If the encode API matches an existing base (e.g. any TXCDR
     variant that takes `(B, T, d)` → `(B, d_sae)`), add the arch
     name to the corresponding `if arch in (...)` tuple.
4. **`run_autoresearch.sh`'s `BASE_OF`** map: set the baseline arch
   to compute Δ_val against. If omitted, defaults to txcdr_t5.

Then launch via:

```bash
bash experiments/phase5_downstream_utility/run_autoresearch.sh <name>
```

The orchestrator will train, probe baseline at val if missing, probe
the candidate at val, summarise, write a row to
`autoresearch_index.jsonl`, and commit+push per milestone.

### Current scoreboard (2026-04-21, seed 42, last_position_val, k=5)

| # | candidate | verdict | Δ_val | t | wins/losses |
|---|---|---|---|---|---|
| A3 | matryoshka_txcdr_contrastive_t5 | **FINALIST** | +0.0155 | +1.46 | 22/12 |
| A2 | txcdr_contrastive_t5 | **FINALIST** | +0.0120 | +1.17 | 24/10 |
| A1 | txcdr_rotational_t5 | DISCARD | −0.0332 | −3.12 | 12/23 |
| A5 | txcdr_basis_expansion_t5 | DISCARD | −0.0448 | −4.15 | 9/26 |
| A4 | mlc_temporal_t3 | DISCARD | −0.0615 | −3.54 | 7/27 |
| A10 | time_layer_contrastive_t5 | AMBIGUOUS | −0.0147 | −1.11 | 15/18 |
| A8 | txcdr_dynamics_t5 | DISCARD | −0.0658 | −5.64 | 5/29 |

Observed pattern: InfoNCE on adjacent latents wins (2/2); hand-designed
decoder / encoder constraints lose (3/3). See autoresearch plan's
*"Tier 1 results"* + *"Tier 2 shelving decision"* sections for the
full discussion.

### Agentic autoresearch cycles (2026-04-21 → 2026-04-22, 8 cycles + 4 variance runs)

Inspired by Karpathy's autoresearch: Claude-driven
hypothesis → code change → result → takeaway loop. Full per-cycle
log at [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md).
Seeded from the 8 TXC / 4 MLC hypotheses listed there.

**Cycle inventory (Δ_val vs family vanilla base, last_position_val, seed 42):**

| # | arch name | family | change vs Part-B A3/A2/MLC | Δ_val | t | verdict |
|---|---|---|---|---|---|---|
| 01 | `agentic_txc_01` | TXC | + orth penalty λ=1.0 on scale-1 decoder | +0.0078 | +0.63 | LOST |
| **02** | **`agentic_txc_02`** | TXC | **InfoNCE at scales 1,2,3 with γ=0.5 decay** | **+0.0354** | **+3.81** | **BEAT_REF** |
| 03 | `agentic_txc_03` | TXC | cycle 02 but γ=1.0 (equal weights) | +0.0072 | +0.58 | LOST |
| 04 | `agentic_txc_04` | TXC | n_scales=5 (all scales), γ=0.5 | +0.0054 | +0.41 | LOST |
| 05 | `agentic_txc_05` | TXC | n=3, γ=0.3 (sharper decay) | −0.0096 | −0.85 | LOST |
| 06 | `agentic_txc_06` | TXC | cycle 02 + K=4 same-sequence hard negatives | +0.0291 | +3.80 | TIE |
| 07 | `agentic_txc_07` | TXC | cycle 02 with InfoNCE → cosine consistency | +0.0174 | +1.81 | LOST |
| **08** | **`agentic_mlc_08`** | MLC | **InfoNCE at prefix lengths d_sae/4, d_sae/2, d_sae with γ=0.5** | **+0.0163** | **+2.45** | **BEAT_REF** |

**Implementation files for the two winners**:

- TXC: `src/architectures/matryoshka_txcdr_contrastive_multiscale.py`
  — subclass of `MatryoshkaTXCDRContrastive` overriding `forward` to
  compute InfoNCE at multiple scales.
- MLC: `src/architectures/mlc_contrastive_multiscale.py` — subclass of
  `MLCContrastive` applying InfoNCE at multiple prefix lengths.

**Losing variants also available for reference** (all in `src/architectures/`):
`matryoshka_txcdr_contrastive_orth.py` (cycle 01),
`matryoshka_txcdr_contrastive_hardneg.py` (cycle 06),
`matryoshka_txcdr_contrastive_consistency.py` (cycle 07).

### 3-seed test-set AUC for the two winners

Final test-set numbers (seed ∈ {42, 1, 2}; full-train 3040 / test 760
at `last_position` and `mean_pool`):

| arch | last_position (3-seed mean ± σ) | mean_pool (3-seed mean ± σ) |
|---|---|---|
| `agentic_txc_02` | 0.7749 ± 0.0038 | **0.7987 ± 0.0020** |
| `agentic_mlc_08` | **0.8046 ± 0.0009** | 0.7851 ± 0.0065 |

Each winner dominates at its family's "home" aggregation. Both top-3
at both aggregations. MLC at last_position has σ=0.0009 — tightest
seed variance on the bench.

Full ranking against all 25 previously-benched archs: see `summary.md`
Figure 1 (last_position) and Figure 2 (mean_pool).
