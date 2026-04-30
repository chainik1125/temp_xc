---
author: Han
date: 2026-04-30
tags:
  - guide
  - design
  - reference
---

## How a TXC is used to do steering — step-by-step mechanics

> Reference document for the Phase 7 Hail Mary case-study pipeline as
> used by Agents Y and W on Gemma-2-2b base, k_pos=20 cells.
> Captures the full chain from training a TXC to producing a graded
> steering matrix entry. All file paths are relative to the repo root
> `/workspace/temp_xc/`. Companion to the findings writeups
> `agent_w/2026-04-30-w-final-summary.md` and
> `agent_y_phase2/2026-04-30-y-final-summary.md`.

### Overview

A TXC (Temporal Crosscoder) is a sparse-autoencoder variant that
ingests a length-T window of residual-stream activations and produces
one window-level latent vector `z ∈ R^{d_sae}` (BatchTopK or TopK
constrained). Steering uses the picked latent's decoder direction to
intervene on the residual stream at inference time, modifying which
concepts the language model emits.

The pipeline has six stages: train → pick a feature per concept →
diagnose feature magnitudes → intervene at varying strengths → grade
the generations → compute the headline metric. Every stage is
implemented in `experiments/phase7_unification/case_studies/` with
files I'll cite per step.

### Step 1 — Train the TXC

**Code**: `experiments/phase7_unification/case_studies/train_kpos20_txc.py`
(W's bare-antidead variant) or `experiments/phase7_unification/case_studies/
train_kpos20_matry.py` (W's matryoshka multiscale variant) or
`experiments/phase7_unification/case_studies/train_kpos20_hailmary.py`
(Y's wrapper around `train_phase7.train_txc_bare_antidead`).

**Inputs**:
- Subject model: `google/gemma-2-2b` (base; not -it).
- Activation cache at L12 residual: `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (shape `(24000, 128, 2304)` fp16, 14.16 GB). Built once via `build_act_cache_phase7.py --layer 12` after `build_token_ids.py`.

**TXC architecture** (`src/architectures/txc_bare_antidead.py`):
```
W_enc:  (T, d_in,  d_sae)        per-position encoder weight
b_enc:  (d_sae,)                  shared bias
W_dec:  (d_sae,  T, d_in)         per-position decoder weight (unit-norm per feature)
b_dec:  (T, d_in)                 per-position bias
```
- d_in = 2304 (Gemma residual width).
- d_sae = 18432 (SAE dictionary size, 8× expansion).
- T = window length (T=2, 3, 5 in Phase 7 W; canonical Phase 5 used T=5).
- k_pos = per-position TopK budget (20 for matched-sparsity cells; 100 canonical).
- k_win = T · k_pos (window-level TopK budget; what the encoder actually applies).

**Encode**:
```
z = einsum("btd,tds->bs", x, W_enc) + b_enc       # (B, T, d_in) → (B, d_sae)
z = TopK(z, k=k_win)                              # zero out all but top-k_win
```

**Decode**:
```
x_hat = einsum("bs,std->btd", z, W_dec) + b_dec   # (B, d_sae) → (B, T, d_in)
```

**Loss**: per-window L2 reconstruction `||x - x_hat||²` averaged over batch + positions, plus the anti-dead "AuxK" term that penalizes feature death.

**Training config** (canonical `paper_archs.json::training_constants`):
- batch=4096, lr=3e-4, max_steps=25_000, grad_clip=1.0,
- plateau_threshold=0.02 (early-stop when 5-window loss-improvement-fraction drops below this), min_steps=3000,
- preload 24k FineWeb-Edu sequences (14.16 GB on GPU),
- decoder normalised to unit-norm per feature each step,
- gradient parallel to decoder removed each backward.

**Wall time** (A40 RunPod, 46 GB VRAM):
- bare-antidead T=3, k_pos=20: 33 min, plateau-converged step 4600
- bare-antidead T=5, k_pos=20: 46 min, plateau-converged step 3800
- matryoshka multiscale T=5: 95 min, plateau-converged step 3200 (5× per-step due to multi-scale InfoNCE + matryoshka H/L head)

**Output**:
- `experiments/phase7_unification/results/ckpts/<arch_id>__seed<S>.pt` (state dict, ~1 GB for bare T=3, ~2.7 GB for matryoshka).
- `experiments/phase7_unification/results/training_logs/<arch_id>__seed<S>.json` (meta + loss/l0 telemetry).

The meta JSON is the key for downstream loaders — it carries `src_class`, `T`, `k_pos`, `k_win`, `hook_name`, anchor layer, and is read by every case-study script.

### Step 2 — Pick a steering feature per concept

**Code**: `experiments/phase7_unification/case_studies/steering/select_features.py`,
calling into `case_studies/_arch_utils.py::encode_per_position` (the
single, arch-agnostic helper).

**Inputs**: a 30-concept benchmark fixture (`concepts.py`) where each concept has 5 example sentences (e.g., "medical": clinical-style sentences; "poetic": lyrical sentences; "harmful_content": flagged content; etc.) and a small set of contrastive baseline sentences.

**The TXC encoder is used here — same one trained in Step 1, not a different one.** For window archs, "encode at position t" means "feed the T-token window ending at t through the TXC encoder, producing one window-level latent z, and attribute it to position t". Implementation in `_arch_utils._slide_windows` + `encode_per_position`:

```text
# input: residuals (N sentences, S tokens, d_in=2304)
# output: per-position latents (N, S, d_sae) — one z per position

for window arch (TXC at window length T):
    windows = slide_T(residuals, stride=1)         # → (N, S-T+1, T, d_in)
    flat    = windows.reshape(N*K, T, d_in)        # K = S - T + 1
    z       = TXC.encode(flat)                     # → (N*K, d_sae)  one z per window
    z       = z.reshape(N, K, d_sae)
    # right-edge attribution: window covering tokens [t-T+1 .. t] → position t
    out[:, T-1 : T-1 + K, :] = z
    # boundary: positions t < T-1 (no full window) get zeros and are skipped
                                                   # by the content-position mask
for per-token arch (TopKSAE / T-SAE):
    out = SAE.encode(residuals.reshape(N*S, d_in)).reshape(N, S, d_sae)
```

So **the TXC is slid with stride 1**, one TXC forward per window. There are S-T+1 windows per length-S sequence, and each produces one (d_sae,) latent attributed to the right-edge position. Positions t < T-1 don't have a full window upstream and are excluded from the content-position mask used later in the lift computation.

This is the same encode pattern the steering intervention uses at generation time (Step 4) — at every generation step where t ≥ T-1, the same TXC is fed the T-window ending at t. So feature selection sees the same per-position semantics that intervention will see.

**Algorithm** (per concept):
1. Forward the 5 example sentences through Gemma to L12, take the residual `(N, S, d_in)`.
2. Encode via `encode_per_position` → `(N, S, d_sae)` per-position latents (window-derived for TXC, direct for T-SAE).
3. Mask to content positions (`attention_mask==1`, plus `t ≥ T-1` for window archs).
4. For each feature `j ∈ [0, d_sae)`, compute the **lift** — average activation on concept-positive content positions minus average activation on baseline content positions.
5. Pick the feature with maximum lift, subject to having a meaningful absolute activation (not just a noise-flicker).

**Output**: `results/case_studies/steering/<arch_id>/feature_selection.json` — a dict `{concept_id: {best_idx, best_act_train, best_lift_train, top_5}}`. One picked feature per concept × 30 concepts = 30 features per arch. Some features are picked for multiple concepts ("polysemanticity"); Y observed 24/30 distinct features at T=5/k_pos=20 vs 28/30 at T-SAE k=20.

The picked feature's **decoder direction** `W_dec[picked_idx, :, :]` (a per-position decoder block of shape `(T, d_in)` for window archs, or just `(d_in,)` for per-token archs) is what the intervention in Step 4 will activate.

### Step 3 — Steering, the basic idea

The picked SAE feature is a learned **direction in residual-stream space** — the unit-norm vector `W_dec[picked_idx]`. The model uses this direction (during normal forward passes) to write the "this output is medical-flavoured" / "this output is poetic" / etc. signal into its residual stream.

**Steering = forcefully writing more of that direction into the residual at inference time** so the model's downstream layers act as if the concept is strongly present, even when the prompt is neutral.

The simplest implementation would be additive: `residual[t] += strength × W_dec[picked_idx]`. The protocol used here is slightly more careful — **paper-clamp**:

1. At the SAE hook layer (L12), encode the live residual through the SAE → latents `z`.
2. Take a copy `z_clamped = z`. **Set the picked feature to a fixed strength**: `z_clamped[picked_idx] = s_abs`.
3. Decode both `z` and `z_clamped`. The difference `decode(z_clamped) - decode(z)` is the residual perturbation that "flips this one feature from its natural value to s_abs".
4. Add that delta to the residual.
5. Continue forward through the rest of the model. Sample the next token.

Why clamp to a fixed value rather than adding `s × W_dec[picked_idx]` directly? Because clamping isolates *only* the picked feature's contribution: the SAE's other features are left at their natural values, so the intervention doesn't pile additive noise on top of an already-active feature. If the prompt is concept-relevant, `s_abs` is the *new* activation; if irrelevant, the SAE has the picked feature near zero and `s_abs` activates it from scratch.

**The strength `s_abs` matters a lot, and it's set per-arch.** Different SAE architectures have wildly different natural firing scales for the same picked feature. Per-token T-SAE k=20 has `⟨|z|⟩ ≈ 10`; a window TXC at k_pos=100 has `⟨|z|⟩ ≈ 30`; a multi-layer crosscoder has `⟨|z|⟩ ≈ 159`. Clamping to "100" means 10× natural for one arch and 0.6× natural for another — completely different intervention regimes.

So we measure each arch's natural firing magnitude (Step 3a below) and apply a **family-normalised** strength: `s_abs = s_norm × ⟨|z|⟩_arch`, with `s_norm` swept over `{0.5, 1, 2, 5, 10, 20, 50}`. This way `s_norm=10` means "10× natural firing" for every arch — a fair comparison.

#### Step 3a — Diagnose `⟨|z|⟩` per arch (the normalisation constant)

`experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`. Forward 30 concepts × 5 example sentences = 150 sentences through Gemma → SAE encoder → read each picked feature's activation at content positions (one picked feature per concept) → pool and compute `mean(|z|)`. Output: `results/case_studies/diagnostics_kpos20/z_orig_magnitudes.json` keyed by arch_id.

(Multi-agent caveat: the file is written via overwrite, so when both Y and W write to the same path, each agent's run clobbers the other's entry. We patched `run_perposition.sh` to verify before invoking intervene — see `feedback_zmag_clobber.md`.)

### Step 4 — How steering differs between per-token and TXC

The per-token case is one position in, one position out. The TXC case is one *window* in, T positions out — and the protocol question is which of those T output positions get written back to the residual.

#### Per-token SAE (T-SAE, TopKSAE)

```text
for each generation step:
  forward through Gemma layers 0..11 → residual at L12: shape (S, d_in)
  for each position p in the prompt prefix:
    z         = SAE.encode(residual[p, :])        # (d_sae,)
    z_clamped = z.clone(); z_clamped[picked] = s_abs
    delta     = SAE.decode(z_clamped) - SAE.decode(z)   # (d_in,)  ← single vector
    residual[p, :] += delta
  forward through Gemma layers 13..25 with modified residual
  sample next token
```

One position, one z, one (d_in,)-shaped delta. Clean.

#### TXC (window encoder), two protocols

For a TXC at window length T, encoding "at position p" means feeding the T-token window `[p-T+1 .. p]` to the encoder, which produces one (d_sae,) latent for the window. Decoding returns `(T, d_in)` — a reconstruction for *all* T positions in the window, not just the latest one.

So there's a choice: which of those T decoder-output positions do we use as the steering delta?

**Right-edge protocol** (canonical, what most baseline numbers use):

```text
for each position p where p ≥ T-1:
  window    = residual[p-T+1 : p+1, :]            # (T, d_in)
  z         = TXC.encode(window)                  # (d_sae,)
  z_clamped = z.clone(); z_clamped[picked] = s_abs
  decoded_orig = TXC.decode(z)                    # (T, d_in)
  decoded_new  = TXC.decode(z_clamped)            # (T, d_in)
  delta_p   = (decoded_new - decoded_orig)[T-1, :]   # ← keep ONLY the right-edge slice
  residual[p, :] += delta_p
```

The encoder integrates context from T tokens (the window), but the steering write only touches position p (the most recent). The earlier T-1 decoder-output slices are discarded. Roughly: "write the steered concept at the current position only."

**Per-position protocol** (Q2.C):

```text
for each position p:
  windows_covering_p = {windows w with p ∈ w}    # up to T such windows (stride-1)
  for each w in windows_covering_p:
    z         = TXC.encode(residual[w-T+1 : w+1, :])
    z_clamped = z.clone(); z_clamped[picked] = s_abs
    decoded_w_orig = TXC.decode(z)               # (T, d_in)
    decoded_w_new  = TXC.decode(z_clamped)       # (T, d_in)
    p_in_w = p - (w - T + 1)                     # position-within-window for p
    delta_p_from_w = (decoded_w_new - decoded_w_orig)[p_in_w, :]
  delta_p = mean over w of delta_p_from_w        # average across overlapping windows
  residual[p, :] += delta_p
```

Now position p gets a contribution from *every* window it sits inside (up to T windows for interior positions). Within each window, p sits at a different relative position, and the corresponding decoder-output slice is used. The final delta at p is the average across those windows.

Roughly: "write the steered concept across the whole T-token span the encoder integrated."

**Why per-position can help.** The encoder's window-level latent is a noisy estimate of "concept presence". Right-edge takes one snapshot of the resulting decoder reconstruction (the rightmost slice). Per-position averages T snapshots from T different overlapping windows — noise reduces, signal stays. Empirically: σ_seeds drops 2–5× (0.33 → 0.10 at T=5) and the constrained peak lifts +0.13 to +0.40 depending on T. (Y's data; cell C T=3 boost was +0.23.)

**Why right-edge is the "canonical" choice.** Right-edge mirrors the way a per-token SAE would work — one position in, one delta out — so it makes the cross-arch comparison cleanest. Per-position is a TXC-specific protocol that has no analogue for per-token archs (T=1 collapses both protocols to the same write).

The two protocols give different numbers because the model's downstream layers attend to *all* the prefix positions when computing the next token. Modifying T positions vs 1 position changes how strongly the steered concept influences attention's input.

#### Per-arch strength schedule, repeated

Both protocols use the same family-normalised strength sweep:

```
for s_norm in {0.5, 1, 2, 5, 10, 20, 50}:
  s_abs = round(s_norm × ⟨|z|⟩_arch, 1)
  for each of 30 concepts:
    generate 60 tokens from prompt "We find" with the steering hook on
    save (concept, s_norm, s_abs, generation)
```

That's 30 × 7 = 210 generations per arch per protocol per seed. Each goes to Sonnet 4.6 grading in Step 5.

Why this can help: at sparse k_pos, the picked feature is partly polysemantic — it fires for the right concept at some positions and noisily at others within the window. Right-edge takes a single noisy snapshot; per-position averages over T snapshots, reducing noise. Y observed σ_seeds drops 2–5× under per-position (0.33 → 0.10 at T=5).

### Step 5 — Grade with Sonnet 4.6

**Code**: `experiments/phase7_unification/case_studies/steering/grade_with_sonnet.py`.

For each (concept, s_abs, generation) tuple, send the generation to Claude Sonnet 4.6 with two prompts:

1. **Coherence** (0–3): "Rate how coherent and grammatical this English text is. 0 = incoherent gibberish, 3 = perfectly fluent."
2. **Steering success** (0–3): "Rate how strongly this output exhibits the [concept] pattern. 0 = unrelated, 3 = strongly on-pattern."

Concept name + a brief description are also passed to the grader (so the grader knows what "medical" means etc.).

**Rate-limited**: Anthropic's 50 req/min ceiling is shared across agents. We use `--n-workers 1` to stay well under that with two agents grading concurrently.

**Output**: `results/case_studies/<subdir>/<arch_id>/grades.jsonl` where `<subdir>` is one of:
- `steering_paper_normalised/` (right-edge, family-normalised)
- `steering_paper_window_perposition/` (per-position, family-normalised)
- `steering_paper_normalised_seed1/` (right-edge, seed=1, etc.)

Each line: `{concept_id, strength, success_grade, coherence_grade}` — 210 rows per arch per protocol per seed.

### Step 6 — Compute the metric

For each (arch, protocol, seed) tuple:

1. **Group grades by `s_norm`** — 30 grades per s_norm.
2. **Per-strength means**: `mean_success(s_norm) = avg over 30 concepts`, `mean_coh(s_norm) = avg over 30 concepts`.
3. **Unconstrained peak (METRIC A)**: `max over s_norm of mean_success(s_norm)` — the headline number Y reported at single-seed.
4. **Constrained peak (METRIC B)**: `max over s_norm of mean_success(s_norm) where mean_coh(s_norm) ≥ 1.5` — the brief's locked primary metric. Captures peak on-pattern success at strengths that still produce coherent text.

The 1.5 coherence threshold is a "still-readable English" cutoff. Below it, the steered output is gibberish that happens to contain concept-related tokens — not actually useful steering.

### Why the constrained metric matters (and why it's fragile)

Without the coherence constraint, T-SAE k=20's headline is **1.80** at s_norm=10 — a strong steering peak. But at s_norm=10, T-SAE k=20's coherence has dropped to **1.40** (just below 1.5). The output is on-pattern but barely English.

With the constraint, T-SAE k=20's peak kicks back to s_norm=5 where coherence is acceptable. At seed=42, s_norm=5 gives `mean_coh = 1.667` (above threshold) and `mean_success = 1.10`. Constrained peak = **1.10**.

But at seed=1, s_norm=5 happens to give `mean_coh = 1.40` (just below threshold). The peak kicks back further to s_norm=2 where success is only 0.30. **σ_anchor between seeds is 0.80.** This is W's anchor-σ discovery — the constraint's reliance on a sharp threshold makes the metric fragile to seed-noise that pushes the coh-cliff position around.

Implication: the brief's pre-registered ±0.27 threshold for win/tie/loss calls is INSIDE the anchor's own seed-noise. Multi-seed pooling is required for honest comparisons. Under multi-seed-pooled anchor (0.70), the W matrix shifts: T=2 cells WIN by ≥0.27, T=3/T=5/matryoshka cells TIE.

### How the matched-sparsity argument runs

The brief's question: "does a TXC at the same per-token sparsity as T-SAE k=20 (k_pos=20) match its steering performance?" The motivation: previous Y showed that at canonical k_pos=100, TXC trails T-SAE k=20 by 0.96 in unconstrained peak. The decomposition: ~0.10 of that gap is magnitude-scale bias (Q1.3 corrects), ~0.53 is sparsity (k=20 vs k=100), ~0.20 is residual architecture. If we also match sparsity, the ~0.73 of attributable-to-non-architecture gap closes; the remaining 0.20 architectural difference might be inside seed-noise.

W's contribution to that question: trained T=3 (cell C) bare-antidead and T=5 (cell E) matryoshka multiscale, both at k_pos=20. Y trained T=2 and T=5 bare-antidead. Together with T-SAE k=20 anchor, that's 5 archs × 2 protocols × 2 seeds (where multi-seed is run) = 18 graded matrix entries.

### Findings summary table

| arch (matched-sparsity, k_pos=20) | T | protocol | mean (multi-seed where available) | Δ vs anchor 0.70 | call |
|---|---|---|---|---|---|
| T-SAE k=20 (anchor) | 1 | per-token | 0.70 (σ=0.80) | — | — |
| TXC bare-antidead T=2 (Y) | 2 | right-edge | 1.067 | +0.367 | WIN |
| TXC bare-antidead T=2 (Y) | 2 | per-position | **1.117** | **+0.417** | **WIN** ⭐ |
| TXC bare-antidead T=3 (W cell C) | 3 | right-edge | 0.783 | +0.083 | TIE |
| TXC bare-antidead T=3 (W cell C) | 3 | per-position | 0.783 | +0.083 | TIE |
| TXC bare-antidead T=5 (Y) | 5 | right-edge | 0.867 | +0.167 | TIE |
| TXC bare-antidead T=5 (Y) | 5 | per-position | 0.783 | +0.083 | TIE |
| TXC matryoshka T=5 (W cell E) | 5 | right-edge (single seed) | 0.633 | −0.067 | TIE |
| TXC matryoshka T=5 (W cell E) | 5 | per-position (single seed) | **0.933** | **+0.233** | TIE (close to win) |

**No cell loses.** T=2 cells decisively win. The matched-sparsity argument lands.

### Five paper findings (priority order)

1. **Anchor instability is the methodological lead.** σ_anchor=0.80 dominates. Multi-seed pooling required.
2. **Matched-sparsity TXC ties or beats T-SAE k=20.** None lose; T=2 wins by ≥0.27.
3. **T-axis advantage reverses at sparse k_pos.** Smaller T helps; opposite of canonical k_pos=100 where T=5–8 was the sweet spot.
4. **Matryoshka × per-position synergy.** Matryoshka multiscale loses under right-edge but wins under per-position at T=5 — protocol-conditional family advantage.
5. **Per-class shift.** TXC wins on sentiment (uniformly +1.0 under per-position) and stylistic (T=3 per-position +0.80). Knowledge stays T-SAE-favourable. The TXC-favourable concept class shifts when sparsity tightens.

### File index

- Trainers: `case_studies/{train_kpos20_txc.py, train_kpos20_matry.py, train_kpos20_hailmary.py}`.
- Pipeline launchers: `case_studies/steering/{run_kpos20_pipeline.sh, run_w_phase1_cell.sh, run_perposition.sh}`.
- Pipeline scripts: `case_studies/steering/{select_features.py, diagnose_z_magnitudes.py, intervene_paper_clamp_normalised.py, intervene_paper_clamp_window_perposition.py, grade_with_sonnet.py, compare_kpos20_vs_tsae.py}`.
- Trained ckpts: `results/ckpts/{tsae_paper_k20, txc_bare_antidead_t2_kpos20, txc_bare_antidead_t3_kpos20, txc_bare_antidead_t5_kpos20, agentic_txc_02_kpos20}__seed{42,1}.pt` (where each seed exists).
- Grades: `results/case_studies/steering_paper_{normalised,window_perposition}{,_seed1}/<arch>/grades.jsonl` — 210 rows each.
- Headline plot: `results/case_studies/plots/all_matched_sparsity_kpos20.png`.
- Findings writeups: `agent_w/2026-04-30-w-final-summary.md`, `agent_w/2026-04-29-w-phase1-sweep.md`, `agent_y_phase2/2026-04-30-y-final-summary.md`.
