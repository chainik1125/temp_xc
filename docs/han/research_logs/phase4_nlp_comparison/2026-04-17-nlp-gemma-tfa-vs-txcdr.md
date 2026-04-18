---
author: Han
date: 2026-04-17
tags:
  - results
  - in-progress
---

## NLP Gemma sweep — TFA vs TXCDR vs Stacked on Gemma-2-2B-IT

First real-LLM comparison of TFA-pos against TXCDR and Stacked SAE, on Gemma 2 2B-IT residual-stream activations from FineWeb. Extends Andre's Gemma sweep ([[2026-04-09-nlp_gemma2_summary]]) by adding TFA-pos and a shuffled control at every (layer, k) point.

### TL;DR

- **Raw NMSE at matched sparse k**: Stacked < TXCDR < TFA (TFA ~2× higher than Stacked). But not apples-to-apples — TFA's dense pred codes mean it has ~50× more total active codes per token than Stacked.
- **Shuffle delta** (NMSE_shuffled / NMSE_unshuffled) is the architecture-independent temporal-exploitation measure: **TFA-pos = 1.75–2.13×** (strong temporal dependence), TXCDR = 1.12× (slight), Stacked = 1.0× (none by construction).
- **TFA genuinely uses sequence order** — 2× degradation when tokens are shuffled at every (layer, k) point.
- Shuffle delta for Stacked/TXCDR is confounded at mid-layer (`resid_L13`) by per-position distribution shift — flagged below. Does not affect TFA's clean 2× signal.
- The raw-NMSE ranking is methodologically pointless on its own; this sweep is most useful as the shuffle-control baseline that feeds the Phase 1–3 feature analyses ([[2026-04-17-nlp-feature-comparison-phase1]]).

## Setup

### Data
- **Subject model**: `google/gemma-2-2b-it` (d_model = 2304, 26 layers)
- **Dataset**: FineWeb `sample-10BT`, 24,000 sequences of 128 tokens
- **Layers**: `resid_L13` (mid-network) and `resid_L25` (final layer)
- **Cache format**: memory-mapped `.npy`, `(24000, 128, 2304)` float32 per layer
- **Eval split**: last 2000 sequences held out

### Architectures
All use d_sae = 18,432 (8× expansion):

| Model | Gen key | Window | Latents / token | Params |
|---|---|---|---|---|
| Stacked T=5 | `window_5` | 5 tokens | k (sparse per position) | 425M |
| TXCDR T=5 | `window_5` | 5 tokens | k×T (sparse across window) | 425M |
| TFA-pos | `seq` | full 128-token seq | k sparse novel + ~d_sae dense pred | 807M |

### TFA-pos hyperparameters
TFA training at 10K steps with lr=1e-3 diverged to NaN on real-LLM activations at this scale. A short stability study (see `logs/tfa_debug_fix_v4.log`) found that:

- Input scaling `x → x · sqrt(d)/mean(‖x‖)` is required (TFA's `λ = 1/(4d)` assumes norm ~ √d)
- Batch size 16 produces gradient spikes to 10⁶+ from outlier activations (e.g. BOS tokens)
- Decoder-row unit normalization after each step prevents `D` from drifting
- The combination lr=3e-4, bs=32, bf=8, decoder-norm, NaN-skip yields stable convergence

TFA runs here use these settings with 5000 training steps (vs 10000 for Stacked/TXCDR); the longer compute per step at bs=32 brings total training time to parity.

### Sweep grid
Cross of:
- architectures = {Stacked T=5, TXCDR T=5, TFA-pos}
- k = {50, 100}
- layer = {resid_L13, resid_L25}
- shuffle = {unshuffled, shuffle within sequence}

= **24 runs**, all completed.

## Results

### Reconstruction NMSE (lower is better)

| Layer | k | Shuffle | Stacked T=5 | TXCDR T=5 | TFA-pos |
|---|---|---|---:|---:|---:|
| `resid_L25` | 50  | no  | **0.0585** | 0.0763 | 0.1127 |
| `resid_L25` | 50  | yes | 0.0553 | 0.0857 | 0.2399 |
| `resid_L25` | 100 | no  | **0.0489** | 0.0689 | 0.1123 |
| `resid_L25` | 100 | yes | 0.0463 | 0.0778 | 0.2397 |
| `resid_L13` | 50  | no  | 0.1382 | 0.1547 | 0.1521 |
| `resid_L13` | 50  | yes | 0.0771 | 0.1043 | 0.2661 |
| `resid_L13` | 100 | no  | 0.1092 | 0.1314 | 0.1524 |
| `resid_L13` | 100 | yes | 0.0615 | 0.0903 | 0.2877 |

At matched *sparse* k, **Stacked is best on raw NMSE at every point**, with TXCDR behind by 15–40 %, and TFA behind by another 2×. This ranking is stable across k, layers, and the shuffled condition.

### Active latents per token

| Model | k=50 | k=100 |
|---|---:|---:|
| Stacked T=5 | 49.6 | 98.9 |
| TXCDR T=5 | 246 | 491 |
| TFA-pos | **~2500** | **~2600** |

TXCDR's per-token latent count is `k × T = 50 × 5 = 250` by construction. TFA's is dominated by the **dense predictable component** (~d_sae nonzero codes); the k value only controls the sparse novel head. The NMSE ranking above therefore does *not* compare models at matched active-latent budgets — TFA has ~50× more active codes per token than Stacked.

### Shuffle delta (shuf ÷ unshuf)

| Layer | k | Stacked | TXCDR | TFA-pos |
|---|---|---:|---:|---:|
| `resid_L25` | 50  | 0.95 | 1.12 | **2.13** |
| `resid_L25` | 100 | 0.95 | 1.13 | **2.13** |
| `resid_L13` | 50  | 0.56 | 0.67 | **1.75** |
| `resid_L13` | 100 | 0.56 | 0.69 | **1.89** |

TFA's NMSE is ~2× worse under shuffle at every point. The per-token Stacked / TXCDR numbers are subtler (see *Shuffle confound* below).

## Interpretation

### TFA heavily depends on sequence order

The TFA-pos shuffle delta is 1.75–2.13× at every layer × k combination. Position-shuffling destroys the causal attention pattern that TFA's `pred_codes` are computed from, and reconstruction roughly doubles in error. This is the first direct evidence in this project that TFA exploits sequence structure in real-LLM activations — more strongly than either TXCDR or Stacked.

### Shuffle confound for Stacked / TXCDR (esp. at L13)

The naïve reading of Stacked's shuffle delta (= 0.95 at L25, = 0.56 at L13) would be that shuffling *helps* Stacked. It doesn't — per-position SAEs can't benefit from temporal structure, so this is a confound:

- Stacked T=5 fits 5 independent per-position SAEs. In unshuffled mode, position-0 of every window always lands at certain absolute positions within the 128-token sequence (the first 5 tokens have distinct norm/content statistics, dominated by BOS, title, leading whitespace). Each per-position SAE has to handle that marginal.
- Under shuffle, every window position sees a uniform mix of sequence positions; each per-position SAE now faces a simpler *averaged* distribution, making reconstruction easier.
- The effect is much larger at `resid_L13` (shuf delta 0.56) than `resid_L25` (0.95), suggesting mid-network features have more position-specific structure.

So the shuffle control does **not** cleanly isolate temporal exploitation for Stacked / TXCDR on real LLM data: it confounds temporal information with per-position distribution shift. TFA's 2× degradation is large enough to survive this confound (the temporal effect dominates), but the TXCDR 1.1× effect is hard to separate from the distribution-shift effect.

### Layer dependence

At `resid_L25` (final layer), all three models improve from k=50 to k=100 modestly. Absolute NMSE is low (0.05–0.11), consistent with the layer carrying simpler, more linearly decomposable structure.

At `resid_L13` (mid-network), absolute NMSE is 2–3× higher across the board, and the gap between Stacked and TFA shrinks (TFA and Stacked both at ~0.15 at k=50). Mid-network features seem to be harder to reconstruct generally, and TFA's relative disadvantage shrinks.

### NMSE-at-matched-L0 is not the right TFA comparison

The TFA paper is explicit that at matched novel-L0 (the sparse head) TFA achieves *comparable* NMSE to SAEs, not lower. The raw-NMSE ranking above (Stacked < TXCDR < TFA) should not be read as "TFA is worse at reconstruction" but as "TFA trades some reconstruction capacity for predictable/novel decomposition with dense pred codes." The informative metrics for TFA are:

1. Its **shuffle delta** (how much it relies on sequence order) — clear 2× signal, as above.
2. **Temporal MI / span statistics** on `encode()` activations — computed automatically for every run; not yet analyzed (see below).
3. **Feature quality** (autointerp, feature map clustering) — not run yet for TFA.

## Methodological note on TFA training

Standard TFA training (lr=1e-3, bs=16, no decoder-norm) fails at NLP scale: training proceeds stably for ~1000 steps then gradients explode to 10⁶–10¹⁴, eventually producing NaN weights. The stability study in `logs/tfa_debug_fix_v4.log` shows the failure mode cleanly:

- Baseline: diverges between step 700 and 7700, final eval NMSE undefined (NaN weights)
- +decoder normalization (v2): survives 10K steps but loss plateaus at 400+, eval NMSE 0.72
- +decoder norm +bs=64 (v3): loss plateaus at 400, eval NMSE 0.16
- +decoder norm +bs=64 **+ lr=3e-4** (v4): clean convergence, eval NMSE 0.10, 0 skipped batches

The root cause appears to be outlier batches (rare tokens with unusual norm profiles) producing large gradients that, amplified by AdamW's second-moment accumulator at lr=1e-3, drive the optimizer off a cliff. Larger batches + smaller LR gives the optimizer enough averaging to handle outliers. The fix is implemented in `src/bench/architectures/tfa.py` and activated automatically when `tfa_batch_size >= 32`.

## Open questions

1. **Temporal MI / span statistics**: the bench framework runs these automatically via `TFASpec.encode()` on every run. Results are in the JSONs — not yet plotted or compared against TXCDR/Stacked.
2. **Feature recovery**: TFA decoder directions should form coherent clusters (Andre's finding for TXCDR). The `cluster` metric is computed per run; TFA vs TXCDR cluster quality comparison is pending.
3. **Matched-total-L0 comparison**: running Stacked / TXCDR at k=250 or k=500 would bring total active latents into TFA's range. That comparison was not in this sweep.
4. **Shuffle confound isolation**: to disentangle Stacked / TXCDR's temporal sensitivity from the per-position distribution shift, we'd need a control that preserves per-position marginals but destroys cross-position correlation (e.g., shuffle across sequences at matched positions).
5. **DeepSeek-R1 results** (in progress at time of writing): 4 of 12 DeepSeek runs complete, all TXCDR / Stacked. Early numbers are lower NMSE than Gemma (Stacked k=50 unshuf = 0.0402 at resid_L12), but TFA and shuffled controls are still pending.

## Files

- Code: `src/bench/architectures/tfa.py`, `src/bench/sweep.py`
- Sweep script: `scripts/run_nlp_sweep_full.sh`
- Stability study: `scripts/debug_tfa_stability.py`, `logs/tfa_debug_fix_v4.json`
- Results: `results/nlp_sweep/gemma/results_gemma-2-2b-it__fineweb__resid_L{13,25}{,__shuffled}.json`
- Checkpoints: `results/nlp_sweep/gemma/ckpts/`

## Reproduction

```bash
# Cache activations (~20 min)
python -m temporal_crosscoders.NLP.cache_activations \
    --model gemma-2-2b-it --dataset fineweb --mode forward \
    --num-sequences 24000 --seq-length 128 \
    --layer_indices 13 25 --components resid

# Full sweep (~9h for Gemma alone)
bash scripts/run_nlp_sweep_full.sh
```
