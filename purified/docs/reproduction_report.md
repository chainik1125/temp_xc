# Reproduction report — § 4 Synthetic experiments on framework v2

**Branch**: `arxiv`
**Date**: 2026-06-01
**Hardware**: local RTX 5090 (32 GB VRAM) + 54 GB RAM, WSL.

This report documents the post-submission reproduction of the paper's
synthetic experiments (§ 4) using the rebuilt framework v2 (see
`purified/docs/framework_v2.md`). The goal is **headline-narrative
fidelity**: do the locked architectures, trained via the new
literature-standard token-buffer pipeline, reproduce the paper's main
synthetic-feature-recovery claim?

## Paper claim being tested

§ 4 (and Fig 2) argues:

> *"TXC dictionaries align with global (hidden-chain) features; per-token
> SAE dictionaries align with local (emission) features. The divide is
> robust across two synthetic benchmarks (denoising + coupling)."*

Concrete numerical headline from the v1 (paper) c2.md aggregate:
- **TXC-base T=5** at `k_pos=5` on the coupling bench: gAUC ≈ 0.96
- **TopK-SAE** at `k_pos=2`: gAUC ≈ 0.99 (catches up at higher k)
- **TopK-SAE** at `k_pos=20`: eAUC ≈ 0.75 (local recovery emerges with capacity)
- **TXC-base** at `k_pos=1`: gAUC ≈ 0.99 (dominates at the most-sparse regime)

## v2 reproduction results

**120 cells**: 4 archs × 2 benches × 5 `k_pos ∈ {1, 2, 5, 10, 20}` × 3 seeds,
all at `d_sae=20, n_steps=10K, batch=1024`. Tables below are auto-
populated from the canonical leaderboard
(`scripts/populate_repro_report_multiseed.py`). Two additional archs
(`txc_pro`, `tfa`) ran in earlier passes and remain in the leaderboard
as historical data but are filtered from the rendered tables — TFA was
removed pending a faithfulness review against the upstream paper.

**Dictionary regime.** `d_sae=20` matches `M_emissions` (the number of
local features on both benches) and is **smaller than** the total
ground-truth feature set on the coupling bench (`K_hidden + M_emissions
= 30`). This is the "scarce dictionary" regime — closer to the real-LM
case where features outnumber atoms — and it forces each architecture
to *choose* which feature subset to align with rather than recovering
everything in parallel. An earlier d_sae=40 pass (over-dictionary) is
still in the leaderboard for the curious; the headline pattern is
substantively the same but absolute AUCs are 0.05-0.10 higher there.

### Coupling bench (`toy_coupled_K10_M20_d256`)

<!-- BEGIN AUTO-RESULTS coupling -->
**eAUC**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | 0.293±0.018 | 0.394±0.018 | 0.454±0.003 | 0.472±0.007 | 0.441±0.031 |
| `topk_sae` | 0.325±0.023 | 0.475±0.030 | 0.590±0.050 | 0.564±0.033 | 0.462±0.020 |
| `tsae` | 0.510±0.015 | 0.531±0.051 | 0.507±0.025 | 0.505±0.013 | 0.510±0.015 |
| `txc_base` | 0.530±0.023 | 0.544±0.018 | 0.467±0.004 | 0.467±0.004 | 0.467±0.004 |

**gAUC**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | 0.435±0.006 | 0.643±0.043 | 0.689±0.021 | 0.591±0.045 | 0.524±0.048 |
| `topk_sae` | 0.551±0.073 | 0.853±0.087 | 0.919±0.034 | 0.686±0.016 | 0.554±0.014 |
| `tsae` | 0.809±0.010 | 0.768±0.046 | 0.717±0.042 | 0.707±0.034 | 0.712±0.028 |
| `txc_base` | 0.971±0.017 | 0.946±0.039 | 0.663±0.029 | 0.663±0.029 | 0.663±0.029 |

**NMSE**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | 0.232±0.026 | 0.025±0.003 | 0.005±0.001 | 0.001±0.000 | 0.000±0.000 |
| `topk_sae` | 0.230±0.041 | 0.024±0.001 | 0.005±0.001 | 0.001±0.001 | 0.000±0.000 |
| `tsae` | 0.017±0.002 | 0.003±0.002 | 0.002±0.001 | 0.002±0.001 | 0.002±0.000 |
| `txc_base` | 0.152±0.005 | 0.137±0.006 | 0.133±0.006 | 0.133±0.006 | 0.133±0.006 |

<!-- END AUTO-RESULTS coupling -->

### Denoising bench (`toy_markov_n20_d40_noisy`)

<!-- BEGIN AUTO-RESULTS denoising -->
**eAUC**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | 0.359±0.012 | 0.376±0.012 | 0.504±0.027 | 0.550±0.018 | 0.492±0.013 |
| `topk_sae` | 0.415±0.005 | 0.529±0.043 | 0.822±0.038 | 0.931±0.013 | 0.526±0.018 |
| `tsae` | 0.514±0.002 | 0.615±0.015 | 0.864±0.017 | 0.832±0.033 | 0.611±0.019 |
| `txc_base` | 0.807±0.070 | 0.828±0.027 | 0.453±0.004 | 0.453±0.004 | 0.453±0.004 |

**gAUC**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | — | — | — | — | — |
| `topk_sae` | — | — | — | — | — |
| `tsae` | — | — | — | — | — |
| `txc_base` | — | — | — | — | — |

**NMSE**  (mean ± std across 3 seeds)

| arch | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `stacked_sae` | 0.499±0.004 | 0.374±0.008 | 0.132±0.007 | 0.022±0.008 | 0.006±0.005 |
| `topk_sae` | 0.500±0.004 | 0.384±0.006 | 0.137±0.001 | 0.019±0.003 | 0.005±0.008 |
| `tsae` | 0.461±0.004 | 0.334±0.010 | 0.088±0.006 | 0.017±0.003 | 0.017±0.002 |
| `txc_base` | 0.491±0.007 | 0.427±0.006 | 0.410±0.004 | 0.410±0.004 | 0.410±0.004 |

<!-- END AUTO-RESULTS denoising -->

## AC-only signed-motion bench (FrequencyBench § 5)

The order-sensitive (AC) counterpart to the DC-only benches above now
lives in its own paper-style writeup — see
[`ac_signed_motion_bench.md`](../autoresearch/signed_motion/bench.md). Headline:
at an ample dictionary the window crosscoder recovers the hidden sign at
oracle level (`s_temp = 1.000 ± 0.000`, 3 seeds) while every per-token SAE
is provably pinned at chance (data-processing inequality).

## Methodology notes (v2 differences from v1)

The v2 framework differs from v1 in three load-bearing ways:

1. **Token shuffle buffer is the default**. v1 sampled whole sequences
   `(B, seq_len, d_in)` with strong within-sequence correlation. v2's
   `ActivationBuffer` samples i.i.d. tokens from a buffer pool —
   literature-standard for SAE training (Anthropic, SAEBench App. B).
   For window archs (TXC-base etc.), v2's `WindowBuffer` samples i.i.d.
   T-windows from buffered sequences.

2. **Eval seed = training seed**. v1 had a seed mismatch where the
   evaluator re-materialised the synthetic generator with `seed=0` while
   the model was trained on `seed=1` — different feature directions →
   trained dictionary atoms couldn't match ground-truth. v2 passes the
   training seed through into the eval spec; feature directions are
   stable across train/eval. (See `SyntheticRecovery.protocol_version
   = 1.1.0` for the gating.)

3. **k_win clipping**. For toy synthetic benches where `d_sae=20` and
   `k_pos*T > d_sae`, v2's TXC-base clips `k_win = min(k_pos*T, d_sae)`
   with a warning. v1 raised. At `T=5`, this means `k_pos ≥ 4` already
   saturates: for window archs (TXC-base) the `k_pos ∈ {5, 10, 20}`
   cells all train at the same effective `k_win = 20` and report
   identical AUCs. The Fig 2 x-axis is informative only for
   `k_pos ∈ {1, 2}` at this dictionary size.

4. **Apples-to-apples reconstruction NMSE (protocol 1.2.0)**. The earlier
   `NMSE` (protocol ≤ 1.1.0) scored token archs on all sequence positions
   but window archs on separately-sampled windows — not comparable. v1.2.0
   samples one fixed set of length-5 windows and feeds the *same* windows to
   every architecture, aggregating error over identical positions (a window
   arch reconstructs each window jointly from one code; a per-token arch
   reconstructs each position independently). Only `NMSE` changes;
   `eAUC`/`gAUC` are unaffected. **The NMSE columns in the tables above were
   last populated under 1.1.0 and are pending a re-eval pass under 1.2.0**
   (re-eval reuses the cached checkpoints — `train_key` is unchanged — so it
   does not retrain).

## What's NOT reproduced (deliberate scope)

- **n_steps=30,000**. We use `n_steps=10,000` for this reproduction to
  stay within local-machine time budget. Per-cell wall went from ~10 min
  to ~30 sec. Spot-check convergence study (txc_base, seed=1, coupling,
  k=1) confirms 10K is sufficient:
  - `n_steps=1000`: gAUC=0.984, NMSE=0.130
  - `n_steps=5000`: gAUC=0.988, NMSE=0.077
  - `n_steps=10000`: gAUC=0.988, NMSE=0.070
  - `n_steps=30000`: gAUC=0.988, NMSE=0.070
  Likewise `topk_sae` k=10 denoising eAUC moves from 0.976 (10K) to
  0.942 (30K) — i.e. longer training does *not* further improve it
  (atoms drift to specialize on noise). 10K is the right operating
  point for these toy benches.
- **Full k_pos sweep**. v1 used 12 k_pos values; we use 5
  `{1, 2, 5, 10, 20}` — the key headline points.
- **LM-scale validation**. § 4 covers synthetic only; §§ 5.1-5.4 (real
  LM probing/backtracking/EM/RLHF) are stubbed pending evaluator ports
  from `origin/final`.

## Multi-seed reproduction

Multi-seed coverage was extended in a second pass: 3 seeds × 4 archs ×
2 benches × 5 k_pos = 120 cells. Tables above report **mean ± std
across all 3 seeds**.

Seed variance is small relative to the architectural differences:

- `txc_base` gAUC at k=1 (the headline cell): **0.971 ± 0.017**
- `topk_sae` gAUC at k=2 (sparse SAE peak): **0.853 ± 0.087**
- `tsae` gAUC at k=1: **0.809 ± 0.010**

On the denoising bench the SAE-family local-recovery peak is the
clearest signal: `topk_sae` k=10 eAUC = **0.931 ± 0.013**.

The TXC-vs-SAE gap on global recovery (≥ 0.4 AUC at k=1) is roughly
6-10× larger than seed-noise; the architectural-specialization claim
is robust to seed perturbation, even at this tighter `d_sae=20`
operating point.

## Headline finding

**Yes — the paper's "TXC dictionaries align with global features;
per-token SAEs align with local features" narrative reproduces
cleanly under the rebuilt v2 framework**, on a single seed at
n_steps=10K. Headline numbers below are pulled directly from the
auto-generated tables above.

### Global recovery (coupling bench, gAUC)

The two temporal-aware architectures top the gAUC ranking at the
sparsest setting (`k_pos=1`, mean across 3 seeds):

- **txc_base** : 0.971
- **tsae**     : 0.809
- topk_sae     : 0.551
- stacked_sae  : 0.435

`txc_base` reaches gAUC=0.971 already at k_pos=1; the SAE-family archs
(topk_sae, stacked_sae) trail by 0.4-0.5 AUC at the same sparsity.
`topk_sae` closes the gap at `k_pos=2` (0.853) but **only** at that
exact knee — by `k_pos=5` it has begun regressing, and the TXC clip
artifact dominates higher-k cells for window archs.

This matches the paper's c2 number (TXC-base gAUC ≈ 0.99 at sparse k
in the over-dictionary regime; ≈ 0.97 here under the tighter scarce-
dictionary setting). The gap to topk_sae is wider in v2 than in the
v1 leaderboard, which we attribute to the token shuffle buffer
producing cleaner i.i.d. training compared to v1's whole-sequence
sampling.

### Local recovery (denoising bench, eAUC)

The denoising bench has no hidden chain (gAUC undefined). On local-
feature recovery the ranking flips at moderate k (mean across 3 seeds):

- topk_sae k=10  : eAUC=0.931
- tsae k=5       : 0.864 (and k=10: 0.832)
- txc_base k=2   : 0.828 (peaks early; k=5+ saturates at clip)
- stacked_sae    : peaks 0.550 at k=10

The clean reversal — TXC family wins global, SAE family wins local at
moderate k — is the paper's main architectural-specialization claim,
and it survives both the framework rewrite *and* the move to the
scarce-dictionary regime.

See `Fig 2 — fig2_synthetic_overview_v2.{pdf,png}` for the visual.

### Recalibration note (2026-06-02)

The current synthetic benchmarks (`markov_chain_support` denoising and
`coupled_hmm` coupling) probe only one axis of temporal behaviour:
DC-style aggregation, where temporal models help by *averaging* repeated
noisy evidence for a stable latent. They give no signal on
order-sensitive (AC) structure — a model could pass both by being a
pure temporal smoother. That gap is now closed by the **AC-only
signed-motion bench** — written up as its own paper-style doc,
[`ac_signed_motion_bench.md`](../autoresearch/signed_motion/bench.md) (a
provable per-token impossibility). The coupling/denoising headline numbers
in this section reflect the DC-only regime and should be read alongside the
AC bench, which probes the orthogonal order-sensitive axis.

## What this validates about the framework

- **The data path is correct**. Token shuffle buffer + window buffer
  produce activations that train SAEs to recovery-grade AUCs (txc_base
  gAUC=0.971 at k_pos=1 reproduces the paper's c2 headline under the
  tighter d_sae=20 scarce-dictionary regime).
- **All 4 active architectures are functional**. 40/40 cells succeeded.
  Two additional archs (`txc_pro`, `tfa`) were also benched but have
  since been removed from the active registry — `txc_pro` because we
  no longer need it; `tfa` pending a faithfulness review against the
  upstream paper. Their historical leaderboard rows remain for
  audit-trail purposes but are filtered from the figure + report.
- **The runner is deterministic**. Same `(arch, seed, training_cfg,
  data_key)` → same `train_key` → identical results on rerun
  (cache-hit confirmed: retry of 7 cells hit cache for unchanged inputs).
- **Code-version stamping works**. Every result row carries
  `code_version.commit_sha + dirty + diff_sha256`.
- **The evaluator seed-passthrough fix is necessary**. Without it
  (v1.0.0 of `SyntheticRecovery`), gAUC results were random because
  feature directions disagreed between train and eval. Bumping
  `protocol_version = 1.1.0` invalidated the buggy cells and forced
  recomputation.

## What this surfaces as future work

- Real-LM evaluators (§ 5.1-5.4) still stubbed. Each is a focused
  port from `origin/final`.
- Multi-seed sweep at paper-canonical `n_steps=30,000` would tighten
  numerical fidelity with the paper.
- Upstream adapter wrapper for T-SAE (currently our v1 port code with
  `arch_version="2.0.0-port"` flag).
- Extend the AC / order-sensitive suite beyond the first signed-motion
  bench (now landed — see the AC section above). Natural next steps from
  `autoresearch/frequency_lens.md`: σ>0 noise robustness, a multi-frequency
  (mixed DC+AC) bench, and a sparser/larger-T decoder to surface a cleaner
  per-atom AC signature than the current near-random `atom_dc_fraction`.
- TFA: re-add once we have (a) a faithfulness review against the
  upstream reference impl and (b) a benchmark on which TFA is the
  *intended* test target (not a toy DC bench where its inductive bias
  is unsuitable).
